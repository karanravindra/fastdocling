"""Medusa-shaped draft: parallel residual heads that vLLM can host in-engine.

``LatentDraft`` reads a *window* of target hidden states through a causal transformer block.
vLLM's Medusa proposer hands a drafter ``[num_reqs, hidden_size]`` -- the current position only --
so the window cannot cross that interface.  This is the model that fits it: ``horizon`` independent
residual MLP heads, each mapping the target's final hidden state to the state of one future token.

The architecture is not a choice, it is vLLM's ``model_executor/models/medusa.py`` copied exactly,
because the weights trained here are loaded by that class:

    ResidualBlock.layers = [Linear(h, h, bias=medusa_fc_bias)] * num_hidden_layers
    ResidualBlock(x)     = for layer in layers: x = x + SiLU(layer(x))
    Medusa(hidden)       = [block(hidden) for block in blocks]     # one per head

One property of that class makes the fit better than it first looks: ``original_lm_head=True``
gives every head a *single shared* ``lm_head``, which is already how this project works -- the
target's frozen vocabulary projection turns 576-wide outputs into logits, and the draft learns no
vocabulary of its own.  The FR-Spec pruning, however, does *not* survive the trip: vLLM 0.29
cannot load a Medusa checkpoint carrying a ``token_map`` at all.  See
``export_medusa_checkpoint`` for why, and for what that costs.

**What it costs.**  Configuring ``medusa`` makes vLLM disable async scheduling *and* fall back to
the V1 model runner (it is absent from both ``EagleModelTypes`` and the V2 allowlist).  Measured on
an RTX 5070 Ti that is a 1.92 ms target step against 1.20 ms for plain vLLM, so beating plain vLLM
needs roughly 2.03 tokens/step -- while dropping the window is precisely what lowers acceptance.
``eagle3`` keeps both and needs only ~1.1.  Read ``decode/vllm_decoder.py`` before assuming this
route pays; it is built because it is the one that *fits*, not the one that is fastest.

``forward`` keeps ``LatentDraft``'s contract -- ``[B, T, in_dim] -> [B, T, horizon, 576]`` -- so the
notebook's loss, acceptance metric and ``TorchDecoder.attach_draft`` all work unchanged.
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


class ResidualBlock(nn.Module):
    """One Medusa head: ``num_layers`` residual SiLU projections, no normalisation.

    Mirrors vLLM's ``ResidualBlock`` exactly, including the absence of a norm and the bias flag
    being off by default -- ``medusa_fc_bias`` in the exported config must agree with ``bias``.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_layers: int = 1, bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=bias) for _ in range(num_layers)])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x


class MedusaDraft(nn.Module):
    """``horizon`` parallel residual heads over the target's final hidden state.

    Head ``j`` predicts offset ``FIRST_OFFSET + j``, the same convention the windowed draft and
    the training targets use, so the acceptance numbers stay comparable between the two models.
    """

    def __init__(self, horizon: int = 4, in_dim: int = HIDDEN_SIZE, num_layers: int = 1,
                 bias: bool = False):
        super().__init__()
        if in_dim != HIDDEN_SIZE:
            raise ValueError(
                f"Medusa reads the target's final hidden state, so in_dim must be {HIDDEN_SIZE}, "
                f"got {in_dim}.  The EAGLE-3 tap concatenation (FEATURES='eagle3') has nowhere to "
                "enter vLLM's Medusa interface -- train this model with FEATURES='last'.")
        self.horizon = horizon
        self.num_layers = num_layers
        self.bias = bias
        self.blocks = nn.ModuleList(
            [ResidualBlock(HIDDEN_SIZE, num_layers, bias) for _ in range(horizon)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, T, 576] -> [B, T, horizon, 576], matching ``LatentDraft``'s output contract.

        Every position is independent -- there is no attention and no positional term -- which is
        the whole point: it is what lets vLLM call this with a single hidden state per request.
        """
        return torch.stack([block(hidden_states) for block in self.blocks], dim=-2)


def _validate_vocab(
    vocab: Sequence[int] | np.ndarray | None, full_vocab: int
) -> torch.Tensor | None:
    """Turn ``vocab`` into the int64 token_map vLLM expects, or reject it with a reason.

    The size check is the load-bearing one.  vLLM installs the map only when
    ``truncated_vocab_size < vocab_size``; hand it a *complete* vocabulary in some other order and
    it drops the map, leaves the reordered head in place, and reports each head's argmax -- an
    index into the reordered rows -- as a token id.  Nothing raises and every proposal is wrong,
    so refuse the export instead.
    """
    if vocab is None:
        return None
    ids = torch.as_tensor(np.asarray(vocab)).to(torch.int64)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError(
            f"vocab must be a non-empty 1-D sequence of token ids, got shape {tuple(ids.shape)}.")
    if int(ids.min()) < 0 or int(ids.max()) >= full_vocab:
        raise ValueError(
            f"vocab holds token ids outside the lm_head's {full_vocab} rows "
            f"(min {int(ids.min())}, max {int(ids.max())}).")
    if int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError(
            "vocab contains duplicate token ids; the token_map must be a one-to-one map from "
            "pruned index to vocabulary id.")
    if int(ids.numel()) >= full_vocab:
        raise ValueError(
            f"vocab keeps all {full_vocab} tokens, which vLLM cannot express: it installs a "
            "token_map only when truncated_vocab_size < vocab_size, so it would silently ignore "
            "the map, keep the reordered lm_head, and emit pruned indices as token ids.  Omit "
            "vocab= for a full-vocabulary head.")
    return ids


def export_medusa_checkpoint(
    draft: MedusaDraft,
    lm_head: torch.Tensor,
    out_dir: str | Path,
    *,
    vocab: Sequence[int] | np.ndarray | None = None,
    dtype: torch.dtype = torch.float16,
    allow_token_map: bool = False,
) -> Path:
    """Write ``draft`` as a directory vLLM can load with ``method="medusa"``.

    ``lm_head`` is the target's frozen [vocab, 576] projection.  Returns the directory, ready to
    pass as ``speculative_config={"model": <dir>, ...}``.

    **On the vocabulary pruning.**  Stock vLLM 0.29 cannot load a Medusa checkpoint carrying a
    ``token_map`` at all: ``Medusa.load_weights`` registers the parameter but omits the name from
    the set it returns, so the loader reports it uninitialised.  ``backends._vllm_medusa`` fixes
    that with a registered subclass, which is why ``vocab`` works here.  Temper expectations,
    though: measured on an RTX 5070 Ti, pruning 100,352 -> 35,878 tokens moved the drafted round
    from 3.327 ms to 3.306 ms -- 1.7%.  The head's ~1.45 ms is overwhelmingly fixed per-round
    overhead in vLLM's V1 Medusa proposer, not the vocabulary projection, so this is worth taking
    (it is free) but it is not a lever.

    **On ``dtype``.**  vLLM ignores ``torch_dtype`` here and casts the draft to the *target's*
    dtype as it loads, so this only sets the fidelity of what is written.  The fp16 default costs
    ~2.6e-3 max-abs on the head outputs (measured against the fp32 model, trained checkpoint,
    horizon 4) and rounds twice when the target runs bf16; ``dtype=torch.float32`` reproduces the
    torch model exactly, at twice the file size.
    """
    if vocab is not None and not allow_token_map:
        raise ValueError(
            "writing a token_map needs the loader shim in fastdocling.backends._vllm_medusa, "
            "which is registered through the vllm.general_plugins entry point in pyproject.  "
            "Pass allow_token_map=True once you have confirmed the plugin is installed "
            "(`uv sync`), or omit vocab= for a full-vocabulary head.")
    if lm_head.ndim != 2 or int(lm_head.shape[1]) != HIDDEN_SIZE:
        raise ValueError(
            f"lm_head must be the target's [vocab, {HIDDEN_SIZE}] projection, got "
            f"{tuple(lm_head.shape)} -- pass the weight, not its transpose.")
    full_vocab = int(lm_head.shape[0])

    ids = _validate_vocab(vocab, full_vocab)
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    for h, block in enumerate(draft.blocks):
        for i, layer in enumerate(block.layers):
            tensors[f"blocks.{h}.layers.{i}.weight"] = layer.weight.detach().cpu().to(dtype)
            if layer.bias is not None:
                tensors[f"blocks.{h}.layers.{i}.bias"] = layer.bias.detach().cpu().to(dtype)

    head = lm_head.detach().cpu()
    if ids is not None:
        head = head[ids]
        # vLLM reads this key by name and keeps it as a non-trainable parameter; with it, a head
        # that is still full-sized gets indexed down at load time instead of rejected.
        tensors["token_map"] = ids
        truncated = int(ids.numel())
    else:
        truncated = full_vocab
    # original_lm_head=True means one shared head; vLLM accepts it under the name lm_heads.0.weight.
    tensors["lm_heads.0.weight"] = head.to(dtype).contiguous()

    save_file(tensors, str(out_dir / "model.safetensors"))
    config = {
        "architectures": ["MedusaModel"],
        "model_type": "medusa",
        "hidden_size": HIDDEN_SIZE,
        "num_heads": draft.horizon,
        "num_hidden_layers": draft.num_layers,
        "vocab_size": full_vocab,
        "truncated_vocab_size": truncated,
        "original_lm_head": True,
        "medusa_fc_bias": bool(draft.bias),
        "torch_dtype": str(dtype).removeprefix("torch."),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


__all__ = ["HIDDEN_SIZE", "VOCAB_SIZE", "ResidualBlock", "MedusaDraft", "export_medusa_checkpoint"]
