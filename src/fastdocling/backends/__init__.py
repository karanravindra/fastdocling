"""Backend selection.

``auto`` picks whichever backend this machine can actually run: MLX on Apple Silicon,
vLLM where CUDA is present.  The two are installed by mutually-exclusive extras
(``--extra mlx`` / ``--extra cuda``), so in practice only one ever imports cleanly.
"""

from __future__ import annotations

import importlib
import platform
from typing import Literal

from .base import (  # re-exported: the stable public surface
    END_TAG,
    GEN_BATCH_SIZE,
    MAX_NEW_TOKENS,
    PROMPT,
    Backend,
    Trace,
    eagle3_taps,
    finalize_trace,
    trace_name,
)

Name = Literal["auto", "mlx", "vllm"]
_CLASSES = {"mlx": (".mlx_backend", "TraceExtractor"), "vllm": (".vllm_backend", "VLLMBackend")}


def available() -> list[str]:
    """Backend names whose runtime imports succeed here."""
    out = []
    for name, (mod, _) in _CLASSES.items():
        try:
            importlib.import_module(mod, __package__)
            out.append(name)
        except Exception:
            pass
    return out


def detect() -> str:
    """Best backend for this machine, by hardware rather than by what imports."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    try:
        import torch

        if torch.cuda.is_available():
            return "vllm"
    except Exception:
        pass
    return "mlx"


def get_backend(name: Name = "auto", **kwargs) -> Backend:
    """Instantiate a backend, with an actionable message when its extra is missing."""
    if name == "auto":
        name = detect()
    if name not in _CLASSES:
        raise ValueError(f"unknown backend {name!r}; choose from {', '.join(_CLASSES)} or auto")
    mod_name, cls_name = _CLASSES[name]
    try:
        mod = importlib.import_module(mod_name, __package__)
    except ImportError as e:
        extra = "mlx" if name == "mlx" else "cuda"
        raise ImportError(
            f"the {name!r} backend is not installed here ({e}). "
            f"Install it with:  uv sync --extra {extra}\n"
            f"(the extras are mutually exclusive -- vllm pins torch==2.13.0, mlx wants 2.14)"
        ) from e
    return getattr(mod, cls_name)(**kwargs)


__all__ = [
    "Backend", "Trace", "Name", "get_backend", "detect", "available",
    "eagle3_taps", "finalize_trace", "trace_name",
    "PROMPT", "MAX_NEW_TOKENS", "GEN_BATCH_SIZE", "END_TAG",
]
