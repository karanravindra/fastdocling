"""Speculative decoding of granite-docling with a latent draft model.

The MLX target does prefill and verification with its KV cache; a caller-supplied ``draft_fn``
proposes ``horizon`` tokens from the target's recent hidden states.  Each round forwards
``[last accepted token] + proposals`` in one target step, accepts the longest prefix that matches
the target's greedy choices, appends the target's own next token, and rolls the cache back over
the rejected tail.  With greedy verification the output is token-identical to plain decoding.

``draft_fn(features: mx.array[T, D], tokens: mx.array[T]) -> mx.array | list[int]`` receives the
target's features for every accepted position so far (final normed state, or the EAGLE-3 tap
concat), the token each of those positions produced, and returns up to ``horizon`` token ids.
The two are aligned: ``features[j]`` is the state whose projection emitted ``tokens[j]``, which
is the "state at P, token at P+1" pairing both drafts were trained on.  A latent draft ignores
``tokens``; an EAGLE-3 draft embeds the last one to start its autoregressive loop.

An MLX draft (see ``draft_mlx``) returns a lazy array, so the draft, the target step and the
acceptance test form one graph with a single eval per round.  With ``draft_fn=None`` the loop
degenerates to plain greedy decoding, which is the fair baseline.

``SpecResult`` and ``cached_generate`` now live in ``fastdocling.decode.base`` and are shared with
the transformers and vLLM decoders; they are re-exported here so existing imports keep working.
Prefer ``fastdocling.decode.get_decoder`` for new code.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Sequence

import mlx.core as mx
from mlx_vlm.models.cache import KVCache

from .backends.base import END_OF_UTTERANCE_ID, END_TOKEN_ID
from .backends.mlx_backend import TraceExtractor
from .decode.base import SpecResult, cached_generate  # re-exported: these moved to `decode`

DraftFn = Callable[[mx.array, mx.array], "mx.array | list[int]"]


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
    hist_tokens = mx.array(tokens, dtype=mx.int64)       # the token each of those positions emitted
    rounds = accepted_total = drafts = draft_tokens = 0
    draft_seconds = 0.0
    hist: list[int] = []

    t0 = perf_counter()
    while tokens[-1] not in stop and len(tokens) < max_tokens:
        if draft_fn is not None and horizon > 0:
            td = perf_counter()
            proposals = draft_fn(history, hist_tokens)
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
        # ``new`` is always 1 + n_acc long -- the stop-token cut above shortens both together --
        # so the two stay aligned position for position, including when they are trimmed.
        history = mx.concatenate([history, feats[: 1 + n_acc]], axis=0)
        hist_tokens = mx.concatenate([hist_tokens, mx.array(new, dtype=mx.int64)], axis=0)
        if draft_fn is not None and history_limit and history.shape[0] > 2 * history_limit:
            history, hist_tokens = history[-history_limit:], hist_tokens[-history_limit:]
        tokens.extend(new)
        rounds += 1
        accepted_total += n_acc
        drafts += k > 0
        draft_tokens += k
        hist.append(n_acc)
    decode = perf_counter() - t0
    return SpecResult(tokens, prefill, decode, rounds, accepted_total, draft_seconds, hist,
                      backend="mlx", drafts=drafts, draft_tokens=draft_tokens)
