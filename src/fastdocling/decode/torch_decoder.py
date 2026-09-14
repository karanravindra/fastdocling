"""Portable backend: the HF target plus the trained torch draft, in one process.

Runs wherever torch does -- CUDA, MPS, CPU -- on a plain ``uv sync``, which makes it the only
backend that can measure a speculative speedup on the training box.  The loop mirrors
``fastdocling.spec.speculative_generate`` step for step so the two are comparable:

    prefill the page -> round: forward ``[last accepted token] + proposals`` in one step,
    accept the longest prefix matching the target's greedy choices, append the target's own
    next token, crop the KV cache over the rejected tail.

With greedy verification the output is token-identical to plain decoding.  The baseline
(``use_draft=False``) is this same loop with an empty proposal block, so both sides pay the
same Python, cache and hidden-state overhead and the ratio between them is fair.

**Reading the absolute numbers.** HF eager decoding of a 258M model is launch-bound, not
compute-bound: ~57 tok/s on an RTX 5070 Ti, where vLLM reaches several hundred.  Thirty layers
of small kernels per token means the target step is dominated by fixed overhead that the tiny
draft does not pay, so the measured speedup here is *optimistic* relative to a target served by
an optimized runtime.  Trust the acceptance rate and ``tokens_per_round`` across backends;
compare wall-clock only within one.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import torch

from .base import (
    FEATURE_KEYS_LAST,
    HF_MODEL_ID,
    PROMPT,
    SpecResult,
    accepted_prefix,
    stop_ids,
    truncate_at_stop,
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _hidden_index(key: str) -> int:
    """Feature name -> index into HF's ``hidden_states`` tuple.

    HF emits ``(embeddings, layer_0_out, ..., layer_n-2_out, final_norm)``: entry ``i + 1`` is
    decoder layer ``i``'s output, and the last entry is *after* the final norm (verified: the
    model's own logits are exactly ``lm_head(hidden_states[-1])``).  That is the same convention
    ``backends.base.eagle3_taps`` documents, so ``layer_2`` -> 3, ``layer_14`` -> 15, and
    ``last_hidden_state`` -> -1, matching what the MLX backend records into traces.
    """
    return -1 if key == "last_hidden_state" else int(key.split("_")[1]) + 1


class TorchDecoder:
    """Speculative and baseline decoding of granite-docling through transformers."""

    name = "transformers"
    supports_draft = True
    accepts_latent_draft = True   # attach_draft installs the trained draft

    def __init__(self, model_id: str = HF_MODEL_ID, prompt: str = PROMPT, *,
                 device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.device = torch.device(device) if device is not None else pick_device()
        self.dtype = dtype or (torch.float32 if self.device.type == "cpu" else torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )
        self.stop = stop_ids(self.tokenizer)
        self._draft = None
        self._draft_fn = None

    # -- draft ---------------------------------------------------------------------------
    def attach_draft(self, draft: Any, lm_head: torch.Tensor, *, context_length: int,
                     horizon: int, window: int | None = None, vocab: Any = None) -> None:
        """Install the trained torch draft (used in place, no conversion).

        ``vocab``: optional array of allowed token ids.  The head is restricted to those rows,
        which makes the projection ~4x cheaper because DocTags uses ~19k of 100k tokens; tokens
        outside the set are never proposed but the target still produces them.
        """
        W = context_length if window is None else min(window, context_length)
        self._draft = draft.to(self.device).eval()
        head = lm_head.to(self.device, torch.float32)
        ids = None
        if vocab is not None:
            ids = torch.as_tensor(vocab, dtype=torch.long, device=self.device)
            head = head[ids]

        @torch.inference_mode()
        def draft_fn(history: torch.Tensor) -> list[int]:
            out = self._draft(history[-W:].to(torch.float32)[None])[0, -1]   # [horizon, 576]
            best = (out @ head.T).argmax(-1)
            return (ids[best] if ids is not None else best).tolist()[:horizon]

        self._draft_fn = draft_fn

    # -- forward helpers -----------------------------------------------------------------
    def _features(self, hidden_states: tuple, keys: Sequence[str]) -> torch.Tensor:
        """[T, D] features for one sequence, concatenated over ``keys`` (EAGLE-3 tap fusion)."""
        return torch.cat([hidden_states[_hidden_index(k)][0] for k in keys], dim=-1)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    # -- the loop ------------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, image, *, use_draft: bool, horizon: int = 2,
                 feature_keys: Sequence[str] = FEATURE_KEYS_LAST,
                 history_limit: int | None = 256, max_tokens: int = 8192) -> SpecResult:
        from transformers import DynamicCache

        if use_draft and self._draft_fn is None:
            raise RuntimeError("call attach_draft() before generating with use_draft=True")
        draft_fn = self._draft_fn if use_draft and horizon > 0 else None

        t0 = perf_counter()
        inputs = self.processor(images=[image], text=self.prompt, return_tensors="pt").to(self.device)
        cache = DynamicCache()
        out = self.model(**inputs, past_key_values=cache, use_cache=True, output_hidden_states=True)
        history = self._features(out.hidden_states, feature_keys)[-1:]   # state that predicted token 1
        tokens = [int(out.logits[0, -1].argmax(-1))]
        self._sync()
        prefill = perf_counter() - t0

        rounds = accepted_total = 0
        draft_seconds = 0.0
        hist: list[int] = []

        t0 = perf_counter()
        while tokens[-1] not in self.stop and len(tokens) < max_tokens:
            if draft_fn is not None:
                td = perf_counter()
                proposals = draft_fn(history)          # .tolist() forces the sync we time
                draft_seconds += perf_counter() - td
                block = [tokens[-1], *proposals]
            else:
                proposals, block = [], [tokens[-1]]
            k = len(proposals)
            out = self.model(input_ids=torch.tensor([block], device=self.device),
                             past_key_values=cache, use_cache=True, output_hidden_states=True)
            greedy = out.logits[0].argmax(-1).tolist()          # target's choice after each position
            feats = self._features(out.hidden_states, feature_keys)
            n_acc = accepted_prefix(proposals, greedy[:k])
            # accepted proposals == greedy[:n_acc], plus the target's own bonus token
            new, n_acc = truncate_at_stop(greedy[: n_acc + 1], n_acc, self.stop)
            if k > n_acc:                                        # drop the rejected tail
                cache.crop(-(k - n_acc))                         # negative == "remove this many"
            history = torch.cat([history, feats[: 1 + n_acc]], dim=0)
            if draft_fn is not None and history_limit and history.shape[0] > 2 * history_limit:
                history = history[-history_limit:]
            tokens.extend(new)
            rounds += 1
            accepted_total += n_acc
            hist.append(n_acc)
        self._sync()
        decode = perf_counter() - t0
        return SpecResult(tokens, prefill, decode, rounds, accepted_total, draft_seconds, hist,
                          backend=self.name)
