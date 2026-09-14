"""MLX port of the notebook's ``LatentDraft`` so the draft runs in the target's process.

Architecture mirrors ``torch.nn.TransformerEncoderLayer(norm_first=True, activation="gelu")``
with one layer, a LayerNorm, learned absolute positions and ``horizon`` parallel heads.  Weights
are loaded straight from the torch ``state_dict`` saved by ``train.ipynb``.
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

    def draft_fn(features: mx.array) -> mx.array:
        win = features[-W:].astype(mx.float32)
        T = win.shape[0]
        if T < W:
            win = mx.pad(win, ((0, W - T), (0, 0)))
        return propose(win, mx.array(T - 1))
    return draft_fn
