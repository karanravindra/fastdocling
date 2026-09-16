"""EAGLE-3 draft for granite-docling: the one drafter shape vLLM hosts without penalty.

**Why this exists.**  The Medusa route works but loses: configuring ``medusa`` makes vLLM disable
async scheduling *and* fall back to the V1 model runner, so a drafted round costs 1.86 ms against
plain vLLM's 1.20 ms, plus ~1.45 ms of fixed V1-proposer overhead that vocabulary pruning barely
touches (measured: 1.7%).  ``eagle3`` is in both ``EagleModelTypes`` and the V2-runner allowlist,
so neither penalty applies.

**What the contract costs us.**  This is not the latent draft ported over; it is a different model,
and the parallel heads do not survive:

    latent draft   one causal block over a *window* of final states -> HORIZON parallel heads,
                   all offsets predicted in a single forward pass
    EAGLE-3        one Llama decoder layer, *autoregressive*: propose a token, embed it, feed it
                   back with the drafter's own KV cache, repeat num_speculative_tokens times

``Eagle3LlamaForCausalLM.forward`` returns ``(hidden_states, aux_output)`` -- one state per
position, never ``[B, T, HORIZON, 576]``.  ``future_heads`` has nowhere to go.  What does survive
is the *window*: EAGLE-3's drafter runs attention over its own KV cache, so the history the latent
draft got from a 64-state window is recovered implicitly, without being passed in.

Two further pieces of the contract are already native to this repo.  The drafter's ``fc`` combines
``num_aux_hidden_states`` target layers (default 3) -- exactly the ``FEATURES="eagle3"`` tap
concatenation of layers 2/14/27 that ``fastdocling.data`` already extracts, 1728 -> 576.  And
``draft_id_to_target_id`` (``d2t`` on disk) is FR-Spec pruning done properly: the draft head spans
``draft_vocab_size`` rows and ``d2t`` carries the *offset* to the real token id, so the pruning
costs nothing at load time and needs no shim (unlike Medusa's broken ``token_map``).

The target must advertise ``SupportsEagle3``; ``backends._vllm_idefics3`` already registers an
Idefics3 subclass that does, through the ``vllm.general_plugins`` entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HIDDEN_SIZE = 576
VOCAB_SIZE = 100_352
AUX_LAYERS = (2, 14, 27)          # the taps fastdocling.data records for FEATURES="eagle3"


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
    return out * weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


class Eagle3Layer(nn.Module):
    """One Llama decoder layer with EAGLE-3's doubled attention input.

    Mirrors vLLM's ``llama_eagle3.LlamaDecoderLayer`` at ``layer_idx == 0``: the token embedding
    and the (fc-combined) target hidden state are concatenated before Q/K/V, so ``q/k/v_proj``
    read ``2 * hidden_size``.  Everything downstream is ordinary Llama.
    """

    def __init__(self, hidden: int = HIDDEN_SIZE, heads: int = 9, kv_heads: int = 3,
                 head_dim: int = 64, intermediate: int = 1536, eps: float = 1e-5):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.input_layernorm = RMSNorm(hidden, eps)
        self.hidden_norm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.q_proj = nn.Linear(2 * hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, embeds: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = torch.cat([self.input_layernorm(embeds), self.hidden_norm(hidden_states)], dim=-1)

        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        # Grouped-query attention: vLLM repeats KV heads internally; do the same here so training
        # and the served model compute the same function.
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        hidden_states = residual + self.o_proj(a.transpose(1, 2).reshape(B, T, -1))

        y = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))


class Eagle3Draft(nn.Module):
    """Trainable EAGLE-3 drafter: fc over the target taps, one decoder layer, a draft LM head.

    ``forward`` is teacher-forced over a whole window -- given the target's aux states and the
    token ids actually generated, it predicts the next token at every position.  That is the
    training-time view of the autoregressive loop vLLM runs one step at a time, and it is why the
    ``[B, T, HORIZON, 576]`` contract of the latent draft is *not* kept: EAGLE-3 emits one state
    per position, and multi-token lookahead comes from iterating, not from parallel heads.
    """

    def __init__(self, in_dim: int = HIDDEN_SIZE * len(AUX_LAYERS), hidden: int = HIDDEN_SIZE,
                 heads: int = 9, kv_heads: int = 3, head_dim: int = 64,
                 intermediate: int = 1536, eps: float = 1e-5,
                 draft_vocab_size: int = VOCAB_SIZE, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.in_dim, self.hidden, self.eps = in_dim, hidden, eps
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.intermediate = intermediate
        self.vocab_size, self.draft_vocab_size = vocab_size, draft_vocab_size
        self.use_aux_hidden_state = in_dim != hidden

        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.fc = nn.Linear(in_dim, hidden, bias=False) if self.use_aux_hidden_state else None
        self.layer = Eagle3Layer(hidden, heads, kv_heads, head_dim, intermediate, eps)
        self.norm = RMSNorm(hidden, eps)
        self.lm_head = nn.Linear(hidden, draft_vocab_size, bias=False)

    def forward(self, aux_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """([B, T, in_dim], [B, T]) -> [B, T, draft_vocab_size] logits for the *next* token."""
        hidden = self.fc(aux_states) if self.fc is not None else aux_states
        x = self.layer(self.embed_tokens(input_ids), hidden)
        return self.lm_head(self.norm(x))


def init_from_target(draft: "Eagle3Draft", vocab: Sequence[int] | np.ndarray | None = None,
                     model_id: str | None = None) -> "Eagle3Draft":
    """Copy the target's embedding and LM head into ``draft``; returns ``draft``.

    Both tensors are frozen during training, so nothing here is learned -- but leaving them at
    their random initialisation is not neutral, it is wrong in two separate ways:

    * ``embed_tokens`` is the drafter's *input*.  ``export_eagle3_checkpoint`` omits it, so vLLM
      binds the target's own table at load time.  A drafter trained against a random table has
      learned a map out of an input space the served model never presents.
    * ``lm_head`` is the drafter's *output*.  With a pruned vocabulary the checkpoint must ship
      its own head (``_should_share`` consults the two independently), so whatever sits in this
      tensor at export time is what vLLM serves -- a random projection, if nothing loads it.

    ``vocab`` is the sorted array of target token ids the pruned head spans; its rows are taken
    from the target's head in that order, which is exactly the order ``d2t`` maps back.
    """
    from .target import MODEL_ID, load_embed_tokens, load_lm_head

    model_id = model_id or MODEL_ID
    with torch.no_grad():
        embed = load_embed_tokens(model_id)
        if embed.shape != draft.embed_tokens.weight.shape:
            raise ValueError(f"target embedding is {tuple(embed.shape)} but the draft's is "
                             f"{tuple(draft.embed_tokens.weight.shape)}")
        draft.embed_tokens.weight.copy_(embed.to(draft.embed_tokens.weight.dtype))

        head = load_lm_head(model_id)
        if vocab is not None:
            ids = torch.as_tensor(np.asarray(vocab), dtype=torch.long)
            if ids.numel() != draft.draft_vocab_size:
                raise ValueError(
                    f"vocab has {ids.numel()} ids but the head has {draft.draft_vocab_size} rows; "
                    "build the model with draft_vocab_size=len(vocab)")
            head = head[ids]
        elif head.shape[0] != draft.draft_vocab_size:
            raise ValueError(f"target head has {head.shape[0]} rows but the draft's has "
                             f"{draft.draft_vocab_size}; pass the pruned `vocab`")
        draft.lm_head.weight.copy_(head.to(draft.lm_head.weight.dtype))
    return draft


def export_eagle3_checkpoint(
    draft: Eagle3Draft,
    out_dir: str | Path,
    *,
    vocab: Sequence[int] | np.ndarray | None = None,
    aux_layers: Sequence[int] = AUX_LAYERS,
    dtype: torch.dtype = torch.float16,
    max_position_embeddings: int = 8192,
    share_embeddings: bool = True,
) -> Path:
    """Write ``draft`` as a directory vLLM loads with ``method="eagle3"``.

    ``share_embeddings`` (default) omits any tensor vLLM can bind from the target instead, which
    is what this project wants -- the target's embedding and vocabulary projection are frozen and
    the draft never trains its own.

    The two are decided **separately**, because they are separate tensors.  vLLM sets
    ``has_own_embed_tokens`` and ``has_own_lm_head`` independently as it reads the checkpoint
    (``process_eagle_weight``) and consults them independently when binding (``_should_share``).
    So a pruned draft vocabulary, which does need its own ``lm_head``, does *not* drag the
    embedding along with it: ``embed_tokens`` stays shared.  Coupling them is expensive -- the
    embedding is [vocab, hidden], 57.8M parameters at 100,352 x 576, which was 72% of an exported
    drafter and cost ~0.6 ms per drafted round for nothing.

    ``vocab`` is the sorted array of token ids the draft may propose.  It is written as ``d2t``,
    the *offset* form vLLM expects (``target_id = draft_id + d2t[draft_id]``), and the head must
    already have ``len(vocab)`` rows -- construct the model with ``draft_vocab_size=len(vocab)``.
    Unlike Medusa's ``token_map`` this path is not broken upstream, so pruning needs no shim.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L, cast = draft.layer, lambda t: t.detach().cpu().to(dtype).contiguous()

    tensors = {
        "norm.weight": cast(draft.norm.weight),
        "layers.0.input_layernorm.weight": cast(L.input_layernorm.weight),
        "layers.0.hidden_norm.weight": cast(L.hidden_norm.weight),
        "layers.0.post_attention_layernorm.weight": cast(L.post_attention_layernorm.weight),
        "layers.0.self_attn.q_proj.weight": cast(L.q_proj.weight),
        "layers.0.self_attn.k_proj.weight": cast(L.k_proj.weight),
        "layers.0.self_attn.v_proj.weight": cast(L.v_proj.weight),
        "layers.0.self_attn.o_proj.weight": cast(L.o_proj.weight),
        "layers.0.mlp.gate_proj.weight": cast(L.gate_proj.weight),
        "layers.0.mlp.up_proj.weight": cast(L.up_proj.weight),
        "layers.0.mlp.down_proj.weight": cast(L.down_proj.weight),
    }
    if draft.fc is not None:
        tensors["fc.weight"] = cast(draft.fc.weight)

    # A tensor absent from the checkpoint is one vLLM binds from the target instead, and it
    # decides that per tensor: process_eagle_weight sets has_own_embed_tokens / has_own_lm_head
    # from the names it sees, and _should_share consults each on its own.
    #
    # embed_tokens is the *input* side -- full vocabulary, frozen, identical to the target's --
    # so it is shared whenever sharing is on, pruned head or not.  At 100,352 x 576 it is 57.8M
    # parameters; shipping a copy alongside a pruned head made one drafter 80.6M instead of 22.8M.
    #
    # lm_head is the *output* side, and pruning changes it: a d2t head has len(vocab) rows and
    # cannot be the target's full-size one, so it must ship. Without pruning it is identical to
    # the target's and is shared.
    if not share_embeddings:
        tensors["embed_tokens.weight"] = cast(draft.embed_tokens.weight)
    if not share_embeddings or vocab is not None:
        tensors["lm_head.weight"] = cast(draft.lm_head.weight)

    if vocab is not None:
        ids = torch.as_tensor(np.asarray(vocab), dtype=torch.long)
        if ids.numel() != draft.draft_vocab_size:
            raise ValueError(
                f"vocab has {ids.numel()} ids but the head has {draft.draft_vocab_size} rows; "
                "build the model with draft_vocab_size=len(vocab)")
        # vLLM computes targets = arange(draft_vocab) + d2t, so store the offset, not the id.
        tensors["d2t"] = ids - torch.arange(ids.numel(), dtype=torch.long)

    save_file(tensors, str(out_dir / "model.safetensors"))
    config = {
        "architectures": ["Eagle3LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": draft.hidden,
        "intermediate_size": draft.intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": draft.heads,
        "num_key_value_heads": draft.kv_heads,
        "head_dim": draft.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": draft.eps,
        "vocab_size": draft.vocab_size,
        "draft_vocab_size": draft.draft_vocab_size,
        "target_hidden_size": draft.hidden,
        "max_position_embeddings": max_position_embeddings,
        "tie_word_embeddings": False,
        "torch_dtype": str(dtype).removeprefix("torch."),
        # Tells the runner which target layers to tap, and the drafter how wide fc's input is.
        # Top level, not just nested under eagle_config: vLLM's
        # get_eagle3_aux_layers_from_config does a plain
        # getattr(hf_config, "eagle_aux_hidden_state_layer_ids"), so a nested copy is invisible
        # to it.  When it finds nothing it silently falls back to the *target's* defaults and
        # logs "Using Eagle3 auxiliary layers from model" -- for granite-docling that is
        # (2, 15, 27) against the (2, 14, 27) these traces were extracted and trained on, so the
        # drafter is fed a layer it never saw.  Check the engine log says "from config".
        "eagle_aux_hidden_state_layer_ids": list(aux_layers),
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": list(aux_layers),
                         "use_aux_hidden_state": draft.use_aux_hidden_state},
        "num_aux_hidden_states": len(aux_layers),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


__all__ = ["HIDDEN_SIZE", "VOCAB_SIZE", "AUX_LAYERS", "RMSNorm", "Eagle3Layer", "Eagle3Draft",
           "init_from_target", "export_eagle3_checkpoint"]
