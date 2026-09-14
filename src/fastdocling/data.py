"""Streaming dataset over traces written by ``fastdocling-extract``.

Each trace is one page: ``token_ids[N]``, ``prompt_len``, ``state_offset`` and hidden states for
positions ``state_offset..N-1`` (the final normed state; layer_2/14/27 taps only if extracted
with ``--keep-taps``).  Files are loaded one at a time and cut into fixed-length windows.

Alignment: hidden state ``i`` is the target model's state *after reading* token ``i``; the
target's own LM head turned it into token ``i+1``.  In speculative decoding that token is
therefore already known when the draft runs, so the draft heads predict the tokens *after* it:

    inputs        = hidden[s : s+ctx]                            (what the draft sees)
    target_tokens = tokens[s+k : s+ctx+k]   k = 2..horizon+1     (head j predicts offset j+1)
    target_hidden = hidden[s+k : s+ctx+k]                        (their states)

``first_offset`` controls the first predicted offset (2 by default; 1 would make head 1 duplicate
the target's LM head and shift every live proposal by one token).

Windows only cover the generated DocTags: they start at ``prompt_len - 1`` (the state that
predicted the first DocTags token), never inside the image/prompt prefix.
"""

from __future__ import annotations

import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import torch

HIDDEN_SIZE = 576
FIRST_OFFSET = 2
EAGLE3_TAPS = ("layer_2", "layer_14", "layer_27")
Features = Literal["last", "eagle3"]


@dataclass(frozen=True)
class TraceInfo:
    path: Path
    n_tokens: int
    prompt_len: int
    state_offset: int = 0      # position of the first stored hidden state
    has_taps: bool = False

    @property
    def completion_states(self) -> int:
        """Usable positions: states from prompt_len-1 to the end (the DocTags region)."""
        return self.n_tokens - self.prompt_len + 1


def _read_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def scan_traces(root: Path | str, min_completion: int = 1) -> list[TraceInfo]:
    """Index every trace under ``root`` by reading only safetensors headers (fast, no tensor IO)."""
    infos = []
    for p in sorted(Path(root).glob("*.safetensors")):
        h = _read_header(p)
        if "token_ids" not in h or "prompt_len" not in h:
            continue
        n = h["token_ids"]["shape"][0]
        # prompt_len is a 1-element tensor; read it from the file body cheaply
        def scalar(key: str) -> int:
            off = h[key]["data_offsets"]
            with open(p, "rb") as fh:
                (hlen,) = struct.unpack("<Q", fh.read(8))
                fh.seek(8 + hlen + off[0])
                raw = fh.read(off[1] - off[0])
            dtype = {"I32": "<i4", "I64": "<i8", "U32": "<u4"}[h[key]["dtype"]]
            return int(np.frombuffer(raw, dtype=dtype)[0])

        prompt_len = scalar("prompt_len")
        state_offset = scalar("state_offset") if "state_offset" in h else 0
        info = TraceInfo(p, n, prompt_len, state_offset, has_taps="layer_2" in h)
        if info.completion_states >= min_completion:
            infos.append(info)
    return infos


def load_trace(info: TraceInfo, features: Features = "last") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (inputs[T, D_in], target_states[T, 576], tokens[T]) for the DocTags region of one page."""
    from safetensors.torch import load_file

    t = load_file(str(info.path))
    s = info.prompt_len - 1 - info.state_offset          # index into the stored states
    last = t["last_hidden_state"][s:].float()
    if features == "last":
        inputs = last
    elif features == "eagle3":
        if not info.has_taps:
            raise ValueError(f"{info.path.name} has no layer taps; re-extract with --keep-taps for eagle3 features")
        inputs = torch.cat([t[k][s:].float() for k in EAGLE3_TAPS], dim=-1)
    else:
        raise ValueError(features)
    return inputs, last, t["token_ids"][info.prompt_len - 1 :].long()


def input_dim(features: Features) -> int:
    return HIDDEN_SIZE if features == "last" else HIDDEN_SIZE * len(EAGLE3_TAPS)


@dataclass
class Plan:
    """How a (context_length, batch_size) choice maps onto the corpus."""

    pages: int
    tokens: int                # usable positions across the corpus
    context_length: int
    horizon: int
    batch_size: int
    windows_per_epoch: int     # non-overlapping windows the corpus yields
    steps_per_epoch: int
    tokens_per_step: int

    def epochs_for_steps(self, steps: int) -> float:
        return steps / self.steps_per_epoch

    def steps_for_epochs(self, epochs: float) -> int:
        return int(round(epochs * self.steps_per_epoch))

    def describe(self, steps: int | None = None) -> str:
        lines = [
            f"{self.pages:,} pages, {self.tokens:,} usable DocTags positions",
            f"context={self.context_length} horizon={self.horizon} batch={self.batch_size} "
            f"-> {self.tokens_per_step:,} tokens/step",
            f"{self.windows_per_epoch:,} windows/epoch -> {self.steps_per_epoch:,} steps/epoch",
        ]
        if steps is not None:
            lines.append(f"{steps:,} steps = {self.epochs_for_steps(steps):.2f} epochs")
        return "\n".join(lines)


def plan(infos: Sequence[TraceInfo], context_length: int, batch_size: int, horizon: int = 4,
         first_offset: int = FIRST_OFFSET) -> Plan:
    window = context_length + horizon + first_offset - 1
    usable = sum(i.completion_states for i in infos)
    windows = sum(i.completion_states // window for i in infos)
    return Plan(
        pages=len(infos), tokens=usable, context_length=context_length, horizon=horizon,
        batch_size=batch_size, windows_per_epoch=windows,
        steps_per_epoch=max(1, windows // batch_size), tokens_per_step=context_length * batch_size,
    )


def iterate_windows(
    infos: Sequence[TraceInfo],
    context_length: int,
    horizon: int = 4,
    features: Features = "last",
    seed: int = 0,
    shuffle_buffer_files: int = 32,
    first_offset: int = FIRST_OFFSET,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One epoch of windows: (inputs[ctx, D_in], target_states[ctx, horizon, 576], target_tokens[ctx, horizon]).

    Head ``j`` (0-based) is paired with offset ``first_offset + j`` from the input position.

    Files are visited in a shuffled order; each is cut into non-overlapping windows from a random
    start offset (so successive epochs see different boundaries); windows from a buffer of files
    are shuffled together before being yielded.
    """
    rng = random.Random(seed)
    order = list(infos)
    rng.shuffle(order)
    window = context_length + horizon + first_offset - 1
    offsets = range(first_offset, first_offset + horizon)
    buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def flush():
        rng.shuffle(buffer)
        yield from buffer
        buffer.clear()

    for n, info in enumerate(order, 1):
        x, h, tok = load_trace(info, features)
        T = x.shape[0]
        if T < window:
            continue
        start = rng.randrange(0, T - (T // window) * window + 1)
        for s in range(start, T - window + 1, window):
            inp = x[s : s + context_length]
            th = torch.stack([h[s + k : s + context_length + k] for k in offsets], dim=1)
            tt = torch.stack([tok[s + k : s + context_length + k] for k in offsets], dim=1)
            buffer.append((inp, th, tt))
        if n % shuffle_buffer_files == 0:
            yield from flush()
    yield from flush()


def iterate_batches(
    infos: Sequence[TraceInfo],
    context_length: int,
    batch_size: int,
    horizon: int = 4,
    features: Features = "last",
    seed: int = 0,
    device: torch.device | str = "cpu",
    drop_last: bool = True,
    first_offset: int = FIRST_OFFSET,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One epoch of batches: inputs[B, ctx, D_in], target_states[B, ctx, H, 576], target_tokens[B, ctx, H]."""
    batch: list = []
    for item in iterate_windows(infos, context_length, horizon, features, seed, first_offset=first_offset):
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(torch.stack(z).to(device) for z in zip(*batch))
            batch = []
    if batch and not drop_last:
        yield tuple(torch.stack(z).to(device) for z in zip(*batch))


def split(infos: Sequence[TraceInfo], holdout_fraction: float = 0.02, seed: int = 0) -> tuple[list[TraceInfo], list[TraceInfo]]:
    """Deterministic train/holdout split by page."""
    order = list(infos)
    random.Random(seed).shuffle(order)
    k = max(1, int(len(order) * holdout_fraction))
    return order[k:], order[:k]


def output_vocab(infos: Sequence[TraceInfo], cache: Path | None = None) -> np.ndarray:
    """Sorted token ids that appear in the generated DocTags of ``infos`` (for a pruned draft head).

    DocTags output uses ~10% of the 100k vocabulary; a draft head restricted to it is ~4x cheaper.
    Tokens outside the set can never be proposed (they are still produced by the target), so
    build it from as many pages as possible.  ``cache`` stores the result as .npy.
    """
    if cache is not None and Path(cache).exists():
        return np.load(cache)
    from safetensors.numpy import load_file

    ids: set[int] = set()
    for info in infos:
        toks = load_file(str(info.path))["token_ids"][info.prompt_len :]
        ids.update(np.unique(toks).tolist())
    vocab = np.array(sorted(ids), dtype=np.int64)
    if cache is not None:
        np.save(cache, vocab)
    return vocab
