"""Decode-backend selection for the throughput cells in ``train.ipynb``.

``auto`` picks the best backend that can actually run the speculative loop here: MLX on Apple
Silicon, transformers everywhere else.  ``vllm`` is never chosen automatically because it
measures the baseline target rate only -- ask for it by name.  See ``base`` for why.
"""

from __future__ import annotations

import importlib
import platform
from typing import Literal

from .base import (  # re-exported: the stable public surface
    FEATURE_KEYS_EAGLE3,
    FEATURE_KEYS_LAST,
    HF_MODEL_ID,
    PROMPT,
    Decoder,
    SpecResult,
    cached_generate,
)

Name = Literal["auto", "mlx", "transformers", "vllm"]
_CLASSES = {
    "mlx": (".mlx_decoder", "MLXDecoder"),
    "transformers": (".torch_decoder", "TorchDecoder"),
    "vllm": (".vllm_decoder", "VLLMDecoder"),
}
_EXTRA = {"mlx": "uv sync --extra mlx", "vllm": "uv sync --extra cuda",
          "transformers": "uv sync"}
# The adapters defer their heavy imports into __init__, so importing an adapter proves
# nothing about the machine; probe the runtime it wraps instead.
_RUNTIME = {"mlx": "mlx_vlm", "transformers": "transformers", "vllm": "vllm"}


def available() -> list[str]:
    """Backend names whose runtime is installed here."""
    out = []
    for name, runtime in _RUNTIME.items():
        try:
            importlib.import_module(runtime)
            out.append(name)
        except Exception:
            pass
    return out


def detect() -> str:
    """Best *speculative-capable* backend for this machine."""
    if (platform.system() == "Darwin" and platform.machine() == "arm64"
            and "mlx" in available()):
        return "mlx"
    return "transformers"


def get_decoder(name: Name = "auto", **kwargs) -> Decoder:
    """Instantiate a decode backend, with an actionable message when its extra is missing."""
    if name == "auto":
        name = detect()
    if name not in _CLASSES:
        raise ValueError(f"unknown decode backend {name!r}; choose from {list(_CLASSES)}")
    mod, cls = _CLASSES[name]
    try:
        module = importlib.import_module(mod, __package__)
    except ImportError as e:
        raise ImportError(f"decode backend {name!r} is not installed here ({e}); "
                          f"run `{_EXTRA[name]}`") from e
    return getattr(module, cls)(**kwargs)


__all__ = ["Name", "Decoder", "SpecResult", "FEATURE_KEYS_LAST", "FEATURE_KEYS_EAGLE3",
           "HF_MODEL_ID", "PROMPT", "available", "detect", "get_decoder", "cached_generate"]
