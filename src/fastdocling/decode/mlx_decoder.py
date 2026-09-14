"""Apple Silicon backend: the mlx-vlm target with the draft ported into MLX.

A thin adapter over ``fastdocling.spec.speculative_generate``, which stays the reference
implementation of the loop: the draft returns a *lazy* array, so the draft call, the target's
verification step and the acceptance test fuse into one graph with a single sync per round.
That is why the MLX path is not merged with the torch one -- the laziness is the whole point,
and routing it through eager tensors would change the numbers it exists to measure.

Install with ``uv sync --extra mlx``.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..target import MODEL_ID
from .base import FEATURE_KEYS_LAST, PROMPT, SpecResult


class MLXDecoder:
    """Speculative and baseline decoding of granite-docling through mlx-vlm."""

    name = "mlx"
    supports_draft = True
    accepts_latent_draft = True   # attach_draft installs the trained draft

    def __init__(self, model_id: str = MODEL_ID, prompt: str = PROMPT, *, extractor=None):
        """``extractor``: reuse an already-built ``TraceExtractor`` instead of loading the target."""
        from ..backends.mlx_backend import TraceExtractor
        from ..target import load_target

        self.model_id = model_id
        if extractor is None:
            model, processor, config = load_target(model_id)
            extractor = TraceExtractor(model=model, processor=processor, config=config, prompt=prompt)
        self.extractor = extractor
        self.processor = extractor.processor
        self.tokenizer = extractor.processor.tokenizer
        self._draft_fn = None

    def attach_draft(self, draft: Any, lm_head: Any = None, *, context_length: int, horizon: int,
                     window: int | None = None, vocab: Any = None) -> None:
        """Port the trained torch draft into MLX and build the lazy proposal function.

        ``lm_head`` is ignored: the MLX draft projects with the target's own head, already
        resident in this process, rather than a second float32 copy.
        """
        from ..draft_mlx import MLXLatentDraft, make_draft_fn

        mlx_draft = MLXLatentDraft.from_torch(draft.state_dict(), context_length, horizon)
        self._draft_fn = make_draft_fn(mlx_draft, self.extractor.lm.lm_head, context_length,
                                       window=window, vocab=vocab)

    def generate(self, image, *, use_draft: bool, horizon: int = 2,
                 feature_keys: Sequence[str] = FEATURE_KEYS_LAST,
                 history_limit: int | None = 256, max_tokens: int = 8192) -> SpecResult:
        from ..spec import speculative_generate

        if use_draft and self._draft_fn is None:
            raise RuntimeError("call attach_draft() before generating with use_draft=True")
        result = speculative_generate(
            self.extractor, image, self._draft_fn if use_draft else None, horizon=horizon,
            max_tokens=max_tokens, feature_keys=feature_keys, history_limit=history_limit)
        result.backend = self.name
        return result
