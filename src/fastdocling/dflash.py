"""DFlash-shaped draft: the one vLLM speculative family that drafts k tokens in *one* forward.

EAGLE-3 is autoregressive -- vLLM runs the drafter once per speculated token -- and on a target
this small that orchestration, not the drafter's arithmetic, is the whole cost: ~1.6 ms added to
a ~1.3 ms round, against ~0.13 ms of actual drafter compute.  DFlash exists to remove exactly
that: ``SpeculativeConfig`` sets ``parallel_drafting = True`` for it automatically
(``config/speculative.py:1404``), and it is in both ``EagleModelTypes`` (so async scheduling
survives) and the V2 model runner's supported list (so the fast runner survives).  It is the only
remaining configuration that keeps all three.

**It is not Qwen3-specific despite the class name.**  ``DFlashQwen3DecoderLayer`` reads every
dimension from the draft's own ``hf_config`` -- hidden size, head counts, head_dim, intermediate
size, norm epsilon, rope parameters -- and "qwen3" names the drafter's layer flavour, not the
target's architecture.  The only genuine Qwen3-ism is an unconditional per-head ``q_norm``/
``k_norm`` (``RMSNorm(head_dim)``, 64 weights each), which a Llama-shaped draft simply carries:
initialised to ones it is the identity, trained it is free capacity.

Weight names follow the same convention as ``fastdocling.eagle3``: anything not ``lm_head`` or
``d2t`` is prefixed with ``model.`` by the loader, and separate ``q_proj``/``k_proj``/``v_proj``
and ``gate_proj``/``up_proj`` are fused by the ``packed_modules_mapping`` inherited from
``Qwen3ForCausalLM``.  Omitting ``embed_tokens`` binds the target's, as for EAGLE-3.

``mask_hidden`` must NOT be shipped -- ``DFlashQwen3ForCausalLM.load_weights`` asserts against it,
because DFlash marks drafted slots with ``mask_token_id`` in the vocabulary rather than with a
separate hidden vector (that is a P-Eagle mechanism).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

HIDDEN_SIZE = 576
VOCAB_SIZE = 100_352
AUX_LAYERS = (2, 14, 27)


class DFlashDraft(nn.Module):
    """Weight container matching vLLM's ``DFlashQwen3Model`` parameter-for-parameter.

    This holds the tensors and their shapes; it deliberately does not reimplement DFlash's
    forward (non-causal block attention over the target's context K/V), because nothing here
    needs to run it -- vLLM does.  Training will need a faithful forward; see the module note.
    """

    def __init__(self, hidden: int = HIDDEN_SIZE, heads: int = 9, kv_heads: int = 3,
                 head_dim: int = 64, intermediate: int = 1536, num_layers: int = 1,
                 draft_vocab_size: int = VOCAB_SIZE, aux_layers: Sequence[int] = AUX_LAYERS,
                 eps: float = 1e-5):
        super().__init__()
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, head_dim
        self.intermediate, self.num_layers = intermediate, num_layers
        self.draft_vocab_size, self.aux_layers, self.eps = draft_vocab_size, tuple(aux_layers), eps

        # Target hidden states from len(aux_layers) taps, projected down to the draft width.
        self.fc = nn.Linear(hidden * len(aux_layers), hidden, bias=False)
        self.layers = nn.ModuleList(nn.ModuleDict({
            "q_proj": nn.Linear(hidden, heads * head_dim, bias=False),
            "k_proj": nn.Linear(hidden, kv_heads * head_dim, bias=False),
            "v_proj": nn.Linear(hidden, kv_heads * head_dim, bias=False),
            "o_proj": nn.Linear(heads * head_dim, hidden, bias=False),
            # Qwen3-flavour per-head QK norms; ones = identity for a Llama-shaped draft.
            "q_norm": nn.RMSNorm(head_dim, eps=eps),
            "k_norm": nn.RMSNorm(head_dim, eps=eps),
            "gate_proj": nn.Linear(hidden, intermediate, bias=False),
            "up_proj": nn.Linear(hidden, intermediate, bias=False),
            "down_proj": nn.Linear(intermediate, hidden, bias=False),
            "input_layernorm": nn.RMSNorm(hidden, eps=eps),
            "post_attention_layernorm": nn.RMSNorm(hidden, eps=eps),
        }) for _ in range(num_layers))
        self.hidden_norm = nn.RMSNorm(hidden, eps=eps)
        self.norm = nn.RMSNorm(hidden, eps=eps)
        self.lm_head = nn.Linear(hidden, draft_vocab_size, bias=False)


def export_dflash_checkpoint(
    draft: DFlashDraft,
    out_dir: str | Path,
    *,
    vocab: Sequence[int] | np.ndarray | None = None,
    num_lookahead_tokens: int = 2,
    mask_token_id: int = VOCAB_SIZE - 1,
    causal: bool | None = True,
    dtype: torch.dtype = torch.float32,
    max_position_embeddings: int = 8192,
    rope_theta: float = 10_000.0,
) -> Path:
    """Write ``draft`` as a directory vLLM loads with ``method="dflash"``.

    ``causal=True`` is the conservative default: DFlash's "standard" flavour is non-causal within
    the drafted block, which needs an attention backend advertising non-causal support, and SM120
    is stuck on FlashAttention-2.  Pass ``causal=None`` to leave it unset and take vLLM's own
    resolution.

    ``vocab`` writes ``d2t`` (offset form, ``target_id = draft_id + d2t[draft_id]``); the head must
    already have ``len(vocab)`` rows.  ``embed_tokens`` is never written -- the target's is bound.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cast = lambda t: t.detach().cpu().to(dtype).contiguous()

    tensors: dict[str, torch.Tensor] = {
        "fc.weight": cast(draft.fc.weight),
        "hidden_norm.weight": cast(draft.hidden_norm.weight),
        "norm.weight": cast(draft.norm.weight),
        "lm_head.weight": cast(draft.lm_head.weight),
    }
    for i, layer in enumerate(draft.layers):
        for src, dst in (("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"),
                         ("v_proj", "self_attn.v_proj"), ("o_proj", "self_attn.o_proj"),
                         ("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm"),
                         ("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj"),
                         ("down_proj", "mlp.down_proj"),
                         ("input_layernorm", "input_layernorm"),
                         ("post_attention_layernorm", "post_attention_layernorm")):
            tensors[f"layers.{i}.{dst}.weight"] = cast(layer[src].weight)

    if vocab is not None:
        ids = torch.as_tensor(np.asarray(vocab), dtype=torch.long)
        if ids.numel() != draft.draft_vocab_size:
            raise ValueError(f"vocab has {ids.numel()} ids but the head has "
                             f"{draft.draft_vocab_size} rows")
        tensors["d2t"] = ids - torch.arange(ids.numel(), dtype=torch.long)

    save_file(tensors, str(out_dir / "model.safetensors"))

    dflash_config: dict = {"use_aux_hidden_state": True, "mask_token_id": mask_token_id}
    if causal is not None:
        dflash_config["causal"] = bool(causal)
    config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": draft.hidden,
        "intermediate_size": draft.intermediate,
        "num_hidden_layers": draft.num_layers,
        "num_attention_heads": draft.heads,
        "num_key_value_heads": draft.kv_heads,
        "head_dim": draft.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": draft.eps,
        "attention_bias": False,
        "vocab_size": VOCAB_SIZE,
        "draft_vocab_size": draft.draft_vocab_size,
        "target_hidden_size": draft.hidden,
        "max_position_embeddings": max_position_embeddings,
        "tie_word_embeddings": False,
        "torch_dtype": str(dtype).removeprefix("torch."),
        "rope_theta": rope_theta,
        "rope_parameters": {"rope_type": "default", "rope_theta": rope_theta},
        # Top level, where get_eagle3_aux_layers_from_config actually looks.
        "eagle_aux_hidden_state_layer_ids": list(draft.aux_layers),
        "num_aux_hidden_states": len(draft.aux_layers),
        "num_lookahead_tokens": num_lookahead_tokens,
        "mask_token_id": mask_token_id,
        "dflash_config": dflash_config,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


__all__ = ["HIDDEN_SIZE", "VOCAB_SIZE", "AUX_LAYERS", "DFlashDraft", "export_dflash_checkpoint"]
