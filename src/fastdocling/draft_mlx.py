"""MLX ports of the drafts, so a draft runs inside the target's own process on Apple Silicon.

Two drafts live here and they are different shapes, not variants:

    MLXLatentDraft    one causal block over a window of final states -> ``horizon`` parallel
                      heads, every offset predicted in a single forward pass
    MLXEagle3Draft    one Llama-style decoder layer, autoregressive: propose a token, embed it,
                      feed it back with the drafter's own KV cache, repeat

``LatentDraft`` mirrors ``torch.nn.TransformerEncoderLayer(norm_first=True, activation="gelu")``.
``Eagle3Draft`` mirrors ``fastdocling.eagle3``, which is in turn shaped by what vLLM hosts.
Weights for both are loaded straight from the torch ``state_dict`` saved by ``train.ipynb``.

**The EAGLE-3 draft here runs the function it was trained on, which is not the function vLLM
serves.**  ``Eagle3Layer`` applies no rotary embedding; vLLM's drafter is a real
``LlamaDecoderLayer``, whose ``LlamaAttention`` does ``q, k = self.rotary_emb(positions, q, k)``.
So the weights were fit to un-rotated queries and keys and are then served rotated.  This port
reproduces training, deliberately: it measures what the drafter is worth, and the distance to
what vLLM measures is the cost of that mismatch rather than a property of the draft.  Fixing it
means retraining with rotary applied, not patching this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HIDDEN_SIZE = 576


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(dim, ff)
        self.linear2 = nn.Linear(ff, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        q, k, v = mx.split(self.in_proj(self.norm1(x)), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3) for t in (q, k, v))
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // self.heads) ** -0.5, mask="causal")
        x = x + self.out_proj(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        return x + self.linear2(nn.gelu(self.linear1(self.norm2(x))))


class MLXLatentDraft(nn.Module):
    def __init__(self, context_length: int, in_dim: int = HIDDEN_SIZE, horizon: int = 4,
                 heads: int = 9, ff: int = 1536):
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(in_dim, HIDDEN_SIZE) if in_dim != HIDDEN_SIZE else None
        self.positions = nn.Embedding(context_length, HIDDEN_SIZE)
        self.block = _Block(HIDDEN_SIZE, heads, ff)
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.future_heads = nn.Linear(HIDDEN_SIZE, horizon * HIDDEN_SIZE)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """[B, T, in_dim] -> [B, T, horizon, 576]"""
        T = hidden_states.shape[1]
        x = self.proj(hidden_states) if self.proj is not None else hidden_states
        x = x + self.positions(mx.arange(T))
        x = self.norm(self.block(x))
        return self.future_heads(x).reshape(*x.shape[:2], self.horizon, HIDDEN_SIZE)

    @classmethod
    def from_torch(cls, state_dict: Mapping[str, "object"], context_length: int, horizon: int = 4,
                   in_dim: int | None = None, dtype=mx.float32) -> "MLXLatentDraft":
        sd = {k: mx.array(np.asarray(v.detach().cpu().float().numpy() if hasattr(v, "detach") else v)) for k, v in state_dict.items()}
        if in_dim is None:
            in_dim = sd["proj.weight"].shape[1] if "proj.weight" in sd else HIDDEN_SIZE
        model = cls(context_length, in_dim, horizon)
        # Two torch layouts map onto this one.  The current LatentDraft writes the block out by
        # hand with these very names, so its state_dict needs no translation; checkpoints trained
        # before that used nn.TransformerEncoderLayer, whose keys sit under transformer.layers.0
        # and whose fused QKV is called self_attn.in_proj_weight.  Same tensors either way.
        # MLX keeps the block flat; the torch model splits it into Attention and MLP
        # submodules, and checkpoints trained before that used nn.TransformerEncoderLayer, whose
        # keys sit under transformer.layers.0 with the fused QKV called self_attn.in_proj_weight.
        # Same six tensors and two norms either way -- rename them onto the flat MLX paths.
        if "transformer.layers.0.self_attn.in_proj_weight" in sd:
            p_ = "transformer.layers.0."
            alias = {"block.in_proj": p_ + "self_attn.in_proj", "block.out_proj": p_ + "self_attn.out_proj",
                     "block.linear1": p_ + "linear1", "block.linear2": p_ + "linear2",
                     "block.norm1": p_ + "norm1", "block.norm2": p_ + "norm2"}
            # nn.MultiheadAttention stores the fused QKV as one flat name, not a submodule.
            block = {"block.in_proj.weight": sd[p_ + "self_attn.in_proj_weight"],
                     "block.in_proj.bias": sd[p_ + "self_attn.in_proj_bias"]}
            block.update({f"{dst}.{t}": sd[f"{src_}.{t}"]
                          for dst, src_ in alias.items() if dst != "block.in_proj" for t in ("weight", "bias")})
        else:
            alias = {"block.in_proj": "block.attn.in_proj", "block.out_proj": "block.attn.out_proj",
                     "block.linear1": "block.mlp.linear1", "block.linear2": "block.mlp.linear2",
                     "block.norm1": "block.norm1", "block.norm2": "block.norm2"}
            missing = [f"{src_}.{t}" for src_ in alias.values() for t in ("weight", "bias") if f"{src_}.{t}" not in sd]
            if missing:
                raise KeyError(f"state_dict matches neither layout; missing {missing}")
            block = {f"{dst}.{t}": sd[f"{src_}.{t}"] for dst, src_ in alias.items() for t in ("weight", "bias")}
        params = {
            "positions.weight": sd["positions.weight"],
            **block,
            "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"],
            "future_heads.weight": sd["future_heads.weight"], "future_heads.bias": sd["future_heads.bias"],
        }
        if model.proj is not None:
            params["proj.weight"], params["proj.bias"] = sd["proj.weight"], sd["proj.bias"]
        model.load_weights([(k, v.astype(dtype)) for k, v in params.items()])
        mx.eval(model.parameters())
        return model

    @classmethod
    def from_checkpoint(cls, path: str | Path, dtype=mx.float32) -> "MLXLatentDraft":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        return cls.from_torch(ckpt["state_dict"], ckpt["context_length"], ckpt.get("horizon", 4), dtype=dtype)


def make_draft_fn(draft: MLXLatentDraft, lm_head: nn.Module, context_length: int, *,
                  window: int | None = None, vocab=None, compile: bool = True):
    """Adapter for ``speculative_generate``: features[T, D] -> proposed token ids [horizon] (lazy).

    window: how many recent features the draft sees (default context_length; shorter is cheaper).
    vocab:  optional array of allowed token ids; the head is restricted to those rows
            (DocTags uses ~10k of 100k tokens, making the projection ~4x cheaper).
    The window is right-padded to a fixed length so the graph compiles once; causal attention
    means the output at the last real position ignores the pads.
    """
    W = context_length if window is None else min(window, context_length)
    head_w = lm_head.weight
    ids = None
    if vocab is not None:
        ids = mx.array(np.asarray(vocab, dtype=np.int64))
        head_w = head_w[ids]
        mx.eval(head_w)

    def propose(win: mx.array, last: mx.array) -> mx.array:
        out = draft(win[None])[0]                            # [W, horizon, 576]
        vec = mx.take(out, last, axis=0)                     # [horizon, 576] at the last real position
        best = (vec.astype(head_w.dtype) @ head_w.T).argmax(-1)
        return ids[best] if ids is not None else best

    if compile:
        propose = mx.compile(propose)

    def draft_fn(features: mx.array, tokens: mx.array | None = None) -> mx.array:
        # ``tokens`` is part of the DraftFn contract for EAGLE-3, which embeds the last accepted
        # token; the latent draft reads the target's states alone and ignores it.
        win = features[-W:].astype(mx.float32)
        T = win.shape[0]
        if T < W:
            win = mx.pad(win, ((0, W - T), (0, 0)))
        return propose(win, mx.array(T - 1))
    return draft_fn


# ---------------------------------------------------------------------------------------
# EAGLE-3
# ---------------------------------------------------------------------------------------
# Module and attribute names mirror ``fastdocling.eagle3`` exactly, so a torch state_dict loads
# by its own key names with no rename table -- unlike the latent draft above, which has two
# historical layouts to reconcile.

DRAFT_WINDOW = 128      # positions of history the drafter re-reads each round (training context)


class _Eagle3Layer(nn.Module):
    """One decoder layer with EAGLE-3's doubled attention input; see ``eagle3.Eagle3Layer``."""

    def __init__(self, hidden: int = HIDDEN_SIZE, heads: int = 9, kv_heads: int = 3,
                 head_dim: int = 64, intermediate: int = 1536, eps: float = 1e-5):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.scale = head_dim ** -0.5
        self.input_layernorm = nn.RMSNorm(hidden, eps)
        self.hidden_norm = nn.RMSNorm(hidden, eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps)
        self.q_proj = nn.Linear(2 * hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def attend(self, embeds: mx.array, hidden_states: mx.array, past=None):
        """``(x, (k, v))`` -- the layer output before the final norm, and the KV to cache."""
        residual = hidden_states
        x = mx.concatenate([self.input_layernorm(embeds), self.hidden_norm(hidden_states)], axis=-1)

        B, T, _ = x.shape
        shape = lambda t, n: t.reshape(B, T, n, self.head_dim).transpose(0, 2, 1, 3)
        q = shape(self.q_proj(x), self.heads)
        k, v = shape(self.k_proj(x), self.kv_heads), shape(self.v_proj(x), self.kv_heads)
        if past is not None:
            k, v = mx.concatenate([past[0], k], axis=2), mx.concatenate([past[1], v], axis=2)
        present = (k, v)
        # mx.repeat, not SDPA's own grouped-query path: repeat_interleave is what the torch model
        # trained with, and query head i reading kv head i // rep is the same mapping either way.
        rep = self.heads // self.kv_heads
        kr, vr = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
        # One query with a cache attends to all of it, itself included -- no mask.  More than one
        # means the window prefill, which is causal.
        a = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=self.scale,
                                                 mask="causal" if T > 1 else None)
        hidden_states = residual + self.o_proj(a.transpose(0, 2, 1, 3).reshape(B, T, -1))

        y = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(nn.silu(self.gate_proj(y)) * self.up_proj(y)), present


class MLXEagle3Draft(nn.Module):
    """``fc`` over the target's taps, one decoder layer, a final norm.

    The embedding and the vocabulary projection are deliberately *not* here.  Both are frozen
    copies of the target's own tensors, and the target is already resident in this process --
    holding a second float32 copy of a 100,352 x 576 table would cost more than the rest of the
    draft put together.  ``make_eagle3_draft_fn`` borrows them.
    """

    def __init__(self, in_dim: int = HIDDEN_SIZE * 3, hidden: int = HIDDEN_SIZE, heads: int = 9,
                 kv_heads: int = 3, head_dim: int = 64, intermediate: int = 1536,
                 eps: float = 1e-5):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden, bias=False) if in_dim != hidden else None
        self.layer = _Eagle3Layer(hidden, heads, kv_heads, head_dim, intermediate, eps)
        self.norm = nn.RMSNorm(hidden, eps)

    def combine(self, aux_states: mx.array) -> mx.array:
        """The target's concatenated taps -> ``hidden``.  Applied to target states only: a state
        the drafter fed back to itself is already this wide and must not pass through ``fc``."""
        return self.fc(aux_states) if self.fc is not None else aux_states

    @classmethod
    def from_torch(cls, state_dict: Mapping[str, object], dtype=mx.float32) -> "MLXEagle3Draft":
        sd = {k: mx.array(np.asarray(v.detach().cpu().float().numpy() if hasattr(v, "detach") else v))
              for k, v in state_dict.items()
              if not k.startswith(("embed_tokens", "lm_head"))}
        missing = [k for k in ("fc.weight", "norm.weight", "layer.q_proj.weight") if k not in sd]
        if missing:
            raise KeyError(f"state_dict is not an Eagle3Draft; missing {missing}")
        model = cls(in_dim=sd["fc.weight"].shape[1])
        model.load_weights([(k, v.astype(dtype)) for k, v in sd.items()])
        mx.eval(model.parameters())
        return model


def make_eagle3_draft_fn(draft: MLXEagle3Draft, embed_tokens, lm_head, *, horizon: int,
                         window: int = DRAFT_WINDOW, vocab=None, compile: bool = True):
    """Adapter for ``speculative_generate``: (features[T, D], tokens[T]) -> ids[horizon], lazy.

    Nothing is evaluated here.  The k proposal steps, the target's verification step and the
    acceptance test all land in one graph with a single sync per round, which is the property
    the MLX path exists to preserve -- ``.item()`` anywhere in this loop would cost more than
    the draft does.

    ``vocab`` is the array of target ids the pruned head spans.  The projection is the target's
    own head restricted to those rows -- identical to the draft's frozen ``lm_head``, at no extra
    memory -- and a proposal is mapped back to a target id *before* being embedded, because
    ``embed_tokens`` is indexed by target ids.
    """
    head_w = lm_head.weight if hasattr(lm_head, "weight") else lm_head
    ids = None
    if vocab is not None:
        ids = mx.array(np.asarray(vocab, dtype=np.int64))
        head_w = head_w[ids]
        mx.eval(head_w)

    def propose(features: mx.array, tokens: mx.array) -> mx.array:
        x, past = draft.layer.attend(embed_tokens(tokens)[None].astype(mx.float32),
                                     draft.combine(features[None].astype(mx.float32)))
        out = []
        for _ in range(horizon):
            vec = draft.norm(x[:, -1:]).astype(head_w.dtype)
            best = (vec @ head_w.T).argmax(-1)                  # [1, 1] draft ids
            token = mx.take(ids, best) if ids is not None else best
            out.append(token)
            if len(out) < horizon:
                x, past = draft.layer.attend(embed_tokens(token).astype(mx.float32),
                                             x[:, -1:], past)
        return mx.concatenate(out, axis=-1)[0]

    # Compile only the full-window shape.  A short history cannot simply be padded to it: pads
    # after the last real position would be *before* the queries appended in later steps, which
    # attend to the whole cache, so the drafter would read them.  Early rounds run uncompiled
    # instead, and every round from the window onwards is one fixed shape.
    fast = mx.compile(propose) if compile else propose

    def draft_fn(features: mx.array, tokens: mx.array) -> mx.array:
        if features.shape[0] >= window:
            return fast(features[-window:], tokens[-window:])
        return propose(features, tokens)
    return draft_fn
