"""Speculative decoding of granite-docling with a latent draft model.

The MLX target does prefill and verification with its KV cache; a caller-supplied ``draft_fn``
proposes ``horizon`` tokens from the target's recent hidden states.  Each round forwards
``[last accepted token] + proposals`` in one target step, accepts the longest prefix that matches
the target's greedy choices, appends the target's own next token, and rolls the cache back over
the rejected tail.  With greedy verification the output is token-identical to plain decoding.

``draft_fn(features: mx.array[T, D]) -> mx.array | list[int]`` receives the target's features
for every accepted position so far (final normed state, or the EAGLE-3 tap concat) and returns
up to ``horizon`` token ids.  An MLX draft (see ``draft_mlx``) returns a lazy array, so the
draft, the target step and the acceptance test form one graph with a single eval per round.
With ``draft_fn=None`` the loop degenerates to plain greedy decoding, which is the fair baseline.
"""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import KVCache

from .extract import END_OF_UTTERANCE_ID, END_TOKEN_ID, TraceExtractor

DraftFn = Callable[[mx.array], "mx.array | list[int]"]


@dataclass
class SpecResult:
    tokens: list[int]
    prefill_seconds: float
    decode_seconds: float
    rounds: int                      # target decode steps (verifications)
    accepted: int                    # draft tokens accepted
    draft_seconds: float = 0.0
    accepted_hist: list[int] = field(default_factory=list)

    @property
    def decode_tps(self) -> float:
        return len(self.tokens) / self.decode_seconds

    @property
    def tokens_per_round(self) -> float:
        return len(self.tokens) / self.rounds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpecResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _features(taps: dict[str, mx.array], keys: Sequence[str]) -> mx.array:
    return mx.concatenate([taps[k][0] for k in keys], axis=-1)  # [n, D]


def speculative_generate(
    ex: TraceExtractor,
    image,
    draft_fn: DraftFn | None,
    horizon: int = 4,
    max_tokens: int = 8192,
    feature_keys: Sequence[str] = ("last_hidden_state",),
    history_limit: int | None = 256,
) -> SpecResult:
    """``history_limit``: keep only the most recent features the draft can use (its context window)."""
    lm = ex.lm
    stop = {END_TOKEN_ID, ex.processor.tokenizer.eos_token_id, END_OF_UTTERANCE_ID}

    t0 = perf_counter()
    ids, emb = ex.encode_prompt(image)
    caches = [KVCache() for _ in lm.layers]
    taps, next_id = ex._step(emb.astype(lm.norm.weight.dtype), caches, "causal")
    feats = _features(taps, feature_keys)[-1:]          # state that predicted the first token
    mx.eval(next_id, feats)
    prefill = perf_counter() - t0

    tokens: list[int] = [int(next_id.item())]
    history = feats                                      # [T, D] target features of accepted positions
    rounds = accepted_total = 0
    draft_seconds = 0.0
    hist: list[int] = []

    t0 = perf_counter()
    while tokens[-1] not in stop and len(tokens) < max_tokens:
        if draft_fn is not None and horizon > 0:
            td = perf_counter()
            proposals = draft_fn(history)
            proposals = mx.array(proposals, dtype=mx.int64)[:horizon] if not isinstance(proposals, mx.array) else proposals[:horizon].astype(mx.int64)
            k = proposals.shape[0]
            block = mx.concatenate([mx.array([tokens[-1]], dtype=mx.int64), proposals])[None]
            draft_seconds += perf_counter() - td          # graph construction only; compute is fused below
        else:
            k = 0
            block = mx.array([[tokens[-1]]])
        taps, _ = ex._step(lm.embed_tokens(block), caches, "causal" if k else None)
        greedy = lm.lm_head(taps["last_hidden_state"]).argmax(-1)[0]   # [1+k]: target's choice after each position
        feats = _features(taps, feature_keys)
        # One sync per round: the target's choices and the per-position match flags.
        if k:
            greedy_l, match = mx.eval(greedy, feats) or (greedy.tolist(), (proposals == greedy[:k]).tolist())
            n_acc = 0
            while n_acc < k and match[n_acc]:
                n_acc += 1
        else:
            greedy_l, n_acc = (mx.eval(greedy, feats) or greedy.tolist()), 0
        new = greedy_l[: n_acc + 1]                       # accepted proposals == greedy[:n_acc], plus the bonus token
        for j, t in enumerate(new):                       # a stop token inside the block ends the output there
            if t in stop:
                new, n_acc = new[: j + 1], min(n_acc, j)
                break
        for c in caches:                                  # roll back the rejected tail of this step
            c.trim(k - n_acc)
        history = mx.concatenate([history, feats[: 1 + n_acc]], axis=0)
        if draft_fn is not None and history_limit and history.shape[0] > 2 * history_limit:
            history = history[-history_limit:]
        tokens.extend(new)
        rounds += 1
        accepted_total += n_acc
        hist.append(n_acc)
    decode = perf_counter() - t0
    return SpecResult(tokens, prefill, decode, rounds, accepted_total, draft_seconds, hist)


def _file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def cached_generate(
    ex: TraceExtractor,
    image_path: str | Path,
    draft_fn: DraftFn | None,
    *,
    cache_dir: str | Path,
    key: dict,
    refresh: bool = False,
    **generate_kwargs,
) -> tuple[SpecResult, bool]:
    """``speculative_generate`` on the page at ``image_path``, reusing a stored result when possible.

    Decoding is greedy, so for a fixed (target, draft, page) the token output is deterministic and
    re-running it after a kernel restart is pure waste (~5-60 s per page).  Results are stored as
    JSON under ``cache_dir`` named by the sha1 of ``key`` + the image bytes.  ``key`` must identify
    everything the output depends on: model id for a baseline; plus draft weights, horizon, window
    and vocabulary for a speculative run.  Timings are stored as measured in whichever session ran
    the decode (``measured_at``/``host`` are kept in the file); pass ``refresh=True`` to re-run.

    Returns ``(result, hit)`` where ``hit`` says whether the result came from the cache.
    """
    from transformers.image_utils import load_image

    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    digest = hashlib.sha1(json.dumps({**key, "image": _file_sha1(image_path)}, sort_keys=True, default=str).encode()).hexdigest()
    path = cache_dir / f"{digest}.json"
    if path.exists() and not refresh:
        return SpecResult.from_dict(json.loads(path.read_text())["result"]), True
    result = speculative_generate(ex, load_image(str(image_path)), draft_fn, **generate_kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "key": key, "image_path": str(image_path), "kwargs": {k: str(v) for k, v in generate_kwargs.items()},
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": socket.gethostname(),
        "result": result.to_dict(),
    }))
    tmp.replace(path)
    return result, False
