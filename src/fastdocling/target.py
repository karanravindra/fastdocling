"""Cheap access to the granite-docling target.

Training only needs the target's LM head (to turn 576-d draft outputs into token logits); reading
it straight out of the cached safetensors takes ~0.1 s, versus several seconds and ~600 MB of RAM
for a full ``mlx_vlm.load``.  The full model is loaded lazily, once per process, by ``load_target``
when the speculative-decoding cells need it.
"""

from __future__ import annotations

from functools import lru_cache

import torch

MODEL_ID = "ibm-granite/granite-docling-258M-mlx"
LM_HEAD_KEY = "language_model.lm_head.weight"


def load_lm_head(model_id: str = MODEL_ID) -> torch.Tensor:
    """The target's ``[vocab, 576]`` LM head as float32, without instantiating the model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(model_id, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(LM_HEAD_KEY).float()


@lru_cache(maxsize=1)
def load_target(model_id: str = MODEL_ID):
    """``(model, processor, config)`` loaded once per process; repeated calls return the same objects."""
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(model_id)
    return model, processor, load_config(model_id)
