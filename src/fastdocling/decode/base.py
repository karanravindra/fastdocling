"""Shared contract for end-to-end decode backends.

``fastdocling.backends`` turns pages into *traces* (an offline, throughput-bound job).  This
package is the other half: running the target **one page at a time** to measure latency, which
is what speculative decoding actually improves.  Three runtimes are supported:

    mlx           Apple Silicon, mlx-vlm         full speculative loop   (uv sync --extra mlx)
    transformers  CUDA / MPS / CPU, HF eager     full speculative loop   (plain uv sync)
    vllm          CUDA, vLLM                     baseline only           (uv sync --extra cuda)

**Why vLLM is baseline-only.** The loop below needs to push a block of ``1 + horizon`` tokens
through the target, compare against its greedy choices and then *roll the KV cache back* over
the rejected tail.  vLLM owns its KV cache inside the scheduler and exposes no rollback, so a
custom draft cannot be driven from the Python API.  What vLLM measures instead is the honest
single-stream decode rate of the target on this machine -- the ``TARGET_DECODE_TPS`` that the
teacher-forced estimate in ``train.ipynb`` divides by.  Use ``mlx`` or ``transformers`` for a
measured speedup, ``vllm`` for a trustworthy baseline.

Every decoder returns the same ``SpecResult``, and ``cached_generate`` stores it keyed by
(everything that determines the output) + the page bytes, so re-running a notebook after a
kernel restart re-reads rather than re-decodes.
"""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from ..backends.base import END_OF_UTTERANCE_ID, END_TOKEN_ID, PROMPT

HF_MODEL_ID = "ibm-granite/granite-docling-258M"   # torch/vLLM read the original weights

FEATURE_KEYS_LAST = ("last_hidden_state",)
FEATURE_KEYS_EAGLE3 = ("layer_2", "layer_14", "layer_27")


@dataclass
class SpecResult:
    """One page decoded end to end.

    ``rounds`` is target decode steps: with a draft a round emits ``1 + accepted`` tokens, without
    one it emits exactly one, so ``tokens_per_round`` is the speculative gain before draft cost.
    """

    tokens: list[int]
    prefill_seconds: float
    decode_seconds: float
    rounds: int                      # target decode steps (verifications)
    accepted: int                    # draft tokens accepted
    draft_seconds: float = 0.0
    accepted_hist: list[int] = field(default_factory=list)
    backend: str = ""

    # Aggregate drafting statistics.  The in-process loops fill ``accepted_hist`` per round and
    # leave these at their defaults; vLLM reports only engine-wide counters, so it fills these
    # and leaves ``accepted_hist`` empty.  ``drafts`` can be below ``rounds``: a drafter that
    # sometimes has nothing to propose (ngram finds no matching suffix) still costs a round.
    gpu_util: float = float("nan")   # mean device utilisation % over the timed decode, if sampled
    drafts: int = 0                  # rounds in which the drafter actually proposed
    draft_tokens: int = 0            # proposals made across those rounds
    accepted_per_pos: list[int] = field(default_factory=list)   # accepted count by proposal slot

    @property
    def decode_tps(self) -> float:
        return len(self.tokens) / self.decode_seconds

    @property
    def tokens_per_round(self) -> float:
        return len(self.tokens) / self.rounds

    @property
    def draft_acceptance(self) -> float:
        """Share of proposed draft tokens the target accepted (0.0 when nothing was proposed)."""
        return self.accepted / self.draft_tokens if self.draft_tokens else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpecResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@runtime_checkable
class Decoder(Protocol):
    """What ``train.ipynb``'s throughput cells need from a runtime."""

    name: str

    supports_draft: bool
    """Whether this runtime can run a speculative loop, or only a plain greedy baseline."""

    accepts_latent_draft: bool
    """Whether ``attach_draft`` installs *our* trained draft.

    False for vLLM, which can speculate but only with a drafter of its own (ngram and friends):
    a speedup measured there says what the runtime gives away for free, not what the trained
    draft is worth.  Branch on this, not on ``supports_draft``, before calling ``attach_draft``.
    See ``vllm_decoder`` for why the latent draft cannot be handed to vLLM -- it is the proposer
    interface, not KV rollback, which vLLM handles internally for drafters it hosts.
    """

    def attach_draft(self, draft: Any, lm_head: Any, *, context_length: int, horizon: int,
                     window: int | None = None, vocab: Any = None) -> None:
        """Install the trained torch draft, converting it to the runtime's framework if needed."""
        ...

    def generate(self, image, *, use_draft: bool, horizon: int = 2,
                 feature_keys: Sequence[str] = FEATURE_KEYS_LAST,
                 history_limit: int | None = 256, max_tokens: int = 8192) -> SpecResult:
        """Decode one page: plain greedy when ``use_draft`` is false, speculative otherwise."""
        ...


def stop_ids(tokenizer) -> set[int]:
    """Token ids that end a page: the DocTags terminator, EOS, and the chat end-of-utterance."""
    return {END_TOKEN_ID, tokenizer.eos_token_id, END_OF_UTTERANCE_ID}


def accepted_prefix(proposals: Sequence[int], greedy: Sequence[int]) -> int:
    """How many leading proposals match the target's own greedy choices."""
    n = 0
    while n < len(proposals) and proposals[n] == greedy[n]:
        n += 1
    return n


def truncate_at_stop(new: list[str] | list[int], n_acc: int, stop: set[int]) -> tuple[list[int], int]:
    """Cut a round's emitted tokens at the first stop token, and shrink the accept count with it."""
    for j, t in enumerate(new):
        if t in stop:
            return new[: j + 1], min(n_acc, j)
    return new, n_acc


def _file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def cached_generate(decoder: Decoder, image_path: str | Path, *, use_draft: bool,
                    cache_dir: str | Path, key: dict, refresh: bool = False,
                    **generate_kwargs) -> tuple[SpecResult, bool]:
    """``decoder.generate`` on the page at ``image_path``, reusing a stored result when possible.

    Decoding is greedy, so for a fixed (backend, target, draft, page) the token output is
    deterministic and re-running it after a kernel restart is pure waste (~5-60 s per page).
    Results are stored as JSON under ``cache_dir`` named by the sha1 of ``key`` + the image
    bytes.  ``key`` must identify everything the output depends on -- **including the backend**,
    since MLX, HF and vLLM resolve near-tied bf16 argmaxes differently and their timings are not
    comparable at all.  Timings are stored as measured by whichever session ran the decode
    (``measured_at``/``host`` are kept in the file); pass ``refresh=True`` to re-run and re-time.

    Returns ``(result, hit)`` where ``hit`` says whether the result came from the cache.
    """
    from transformers.image_utils import load_image

    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    digest = hashlib.sha1(json.dumps({**key, "image": _file_sha1(image_path)},
                                     sort_keys=True, default=str).encode()).hexdigest()
    path = cache_dir / f"{digest}.json"
    if path.exists() and not refresh:
        return SpecResult.from_dict(json.loads(path.read_text())["result"]), True
    result = decoder.generate(load_image(str(image_path)), use_draft=use_draft, **generate_kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "key": key, "image_path": str(image_path),
        "kwargs": {k: str(v) for k, v in generate_kwargs.items()},
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(), "result": result.to_dict(),
    }))
    tmp.replace(path)
    return result, False


__all__ = ["PROMPT", "HF_MODEL_ID", "FEATURE_KEYS_LAST", "FEATURE_KEYS_EAGLE3", "SpecResult", "Decoder",
           "stop_ids", "accepted_prefix", "truncate_at_stop", "cached_generate"]
