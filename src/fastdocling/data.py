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

Two loaders read this corpus.  ``iterate_batches`` streams it a file at a time and re-derives
every batch each epoch; ``packed_batches`` reads a ``PackedCorpus`` built once by ``pack_traces``,
which is an order of magnitude cheaper per batch and is what the EAGLE-3 loop uses.  See the
"Packed corpus" section below.

Windows only cover the generated DocTags: they start at ``prompt_len - 1`` (the state that
predicted the first DocTags token), never inside the image/prompt prefix.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
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
    source: str = ""           # name of the root this trace was scanned from

    @property
    def completion_states(self) -> int:
        """Usable positions: states from prompt_len-1 to the end (the DocTags region)."""
        return self.n_tokens - self.prompt_len + 1


def _read_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def scan_traces(root: Path | str | Sequence[Path | str], min_completion: int = 1) -> list[TraceInfo]:
    """Index every trace under ``root`` by reading only safetensors headers (fast, no tensor IO).

    ``root`` may be one directory or several.  Several is how a run mixes corpora of different
    provenance -- ``data/traces`` holds states recorded against the target's own greedy DocTags,
    ``data/traces_prefill`` states recorded against labels borrowed from a Hub dataset -- while
    keeping them separable: each info records the ``source`` root it came from, so a caller can
    weight, split or report the mix (see ``source_counts``).

    Raises rather than returning an empty corpus.  ``Path.glob`` on a directory that does not
    exist yields nothing at all, so a mistyped path -- or a notebook run from a subdirectory,
    where ``data/traces`` resolves relative to the *notebook*, not the repo -- used to surface
    only much later as a ZeroDivisionError inside the acceptance metric, once the training loop
    had silently iterated over nothing.
    """
    roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(r) for r in root]
    if not roots:
        raise ValueError("scan_traces needs at least one root")
    for r in roots:
        if not r.is_dir():
            raise FileNotFoundError(
                f"no trace directory at {r.resolve()} (cwd {Path.cwd()}); paths are relative to "
                "the working directory, so a notebook run from a subdirectory needs an absolute root")
    infos = []
    for r in roots:
        for p in sorted(r.glob("*.safetensors")):
            h = _read_header(p)
            if "token_ids" not in h or "prompt_len" not in h:
                continue
            n = h["token_ids"]["shape"][0]
            # prompt_len is a 1-element tensor; read it from the file body cheaply
            def scalar(key: str, p=p, h=h) -> int:
                off = h[key]["data_offsets"]
                with open(p, "rb") as fh:
                    (hlen,) = struct.unpack("<Q", fh.read(8))
                    fh.seek(8 + hlen + off[0])
                    raw = fh.read(off[1] - off[0])
                dtype = {"I32": "<i4", "I64": "<i8", "U32": "<u4"}[h[key]["dtype"]]
                return int(np.frombuffer(raw, dtype=dtype)[0])

            prompt_len = scalar("prompt_len")
            state_offset = scalar("state_offset") if "state_offset" in h else 0
            info = TraceInfo(p, n, prompt_len, state_offset, has_taps="layer_2" in h, source=r.name)
            if info.completion_states >= min_completion:
                infos.append(info)
    if not infos:
        files = sum(len(list(r.glob("*.safetensors"))) for r in roots)
        where = ", ".join(str(r.resolve()) for r in roots)
        raise ValueError(
            f"no usable traces in {where}: {files} safetensors file(s) found, none with "
            f"at least min_completion={min_completion} generated tokens"
            + ("" if files else " -- run `fastdocling-extract` first"))
    return infos


def source_counts(infos: Sequence[TraceInfo]) -> dict[str, int]:
    """Pages per source root, for reporting what a mixed corpus is actually made of."""
    out: dict[str, int] = {}
    for i in infos:
        out[i.source] = out.get(i.source, 0) + 1
    return out


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
    want_states: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]:
    """One epoch of windows: (inputs[ctx, D_in], target_states[ctx, horizon, 576], target_tokens[ctx, horizon]).

    Head ``j`` (0-based) is paired with offset ``first_offset + j`` from the input position.

    Files are visited in a shuffled order; each is cut into non-overlapping windows from a random
    start offset (so successive epochs see different boundaries); windows from a buffer of files
    are shuffled together before being yielded.

    ``want_states=False`` yields ``None`` in the middle slot, for a token-only objective that
    would otherwise pay to stack and copy a [B, ctx, horizon, 576] float32 tensor per step and
    then discard it (5.3 of the 9.4 ms each batch spent in stack+H2D, when the EAGLE-3 loop still
    streamed).  Medusa distills against the states, so the default keeps them.  The EAGLE-3 loop
    now reads ``packed_batches``, which never builds them at all -- this flag is what the
    streaming path offers a caller that does not want them.
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
            th = (torch.stack([h[s + k : s + context_length + k] for k in offsets], dim=1)
                  if want_states else None)
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
    want_states: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]:
    """One epoch of batches: inputs[B, ctx, D_in], target_states[B, ctx, H, 576], target_tokens[B, ctx, H].

    With ``want_states=False`` the middle element is ``None`` and is neither stacked nor copied.
    """

    def collate(items: list) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        cols = list(zip(*items))
        return tuple(None if c[0] is None else torch.stack(c).to(device) for c in cols)

    batch: list = []
    for item in iterate_windows(infos, context_length, horizon, features, seed,
                                first_offset=first_offset, want_states=want_states):
        batch.append(item)
        if len(batch) == batch_size:
            yield collate(batch)
            batch = []
    if batch and not drop_last:
        yield collate(batch)


# ---------------------------------------------------------------------------------------
# Packed corpus
#
# The streaming loader above re-derives every batch from scratch: open the safetensors, slice
# three taps out of it, concatenate them, upcast to float32, cut windows, stack.  Measured on the
# eagle3 loop that was 22 ms per batch with the traces in page cache and ~290 ms without -- 52%
# to 91% of the step, against ~20 ms of compute.  None of that work depends on the epoch, so it
# should not be paid once per epoch.
#
# Packing does it once into a single contiguous float16 array.  A window is then a contiguous
# slice, a batch is B memcpys, and the per-batch cost drops to ~3 ms of CPU that a prefetch thread
# hides behind the GPU entirely (measured 0.6 ms of wait, 6% of the step).  The pack is ~10.3 GB
# for 2.98M positions at 1728-d, which the page cache holds comfortably; building it took 16 s.
#
# float16, not bfloat16, for two reasons -- and *not* for its extra mantissa bits, which the
# eagle3 loop throws away the moment it casts the batch to bfloat16 for autocast.  First, .npy has
# no bfloat16 dtype, so a memmapped numpy array cannot hold one.  Second, it is what the traces
# already store, so packing is a copy rather than a rounding.  Neither dtype moves fewer bytes;
# both are two.  The taps peak at |544| against float16's 65504 ceiling (checked over 40 pages,
# no inf/nan), so the range bfloat16 would buy is range this corpus never uses.
# ---------------------------------------------------------------------------------------

PACK_VERSION = 2


def page_key(info: TraceInfo) -> str:
    """Identity of a page within a pack: ``<source root>/<file name>``.

    The bare file name is not enough once ``scan_traces`` takes several roots.  ``trace_name``
    builds it from the image path *relative to its own root*, so the same page extracted into
    ``data/traces`` and into ``data/traces_prefill`` gets the same name in both -- and a pack keyed
    on that name would quietly map two different pages onto one row.  Pages scanned before
    ``source`` existed fall back to the bare name.
    """
    return f"{info.source}/{info.path.name}" if info.source else info.path.name


@dataclass(frozen=True, eq=False)   # ndarray fields: == would be ambiguous, hash would raise
class PackedCorpus:
    """Every page's EAGLE-3 taps concatenated into one memmapped float16 array.

    Page ``p`` (named ``names[p]``) occupies rows ``starts[p]:starts[p + 1]``, the same
    DocTags-only region ``load_trace`` returns -- position 0 is the state at ``prompt_len - 1``.
    ``tokens`` is indexed identically, so ``tokens[starts[p] + i]`` is the token the target read
    at that state.  Windows never cross a page boundary.
    """

    root: Path
    taps: np.ndarray          # memmap [positions, dim] float16
    tokens: np.ndarray        # [positions] int32
    starts: np.ndarray        # [pages + 1] int64
    names: np.ndarray         # [pages] page_key of each packed page
    key: str                  # corpus_key of the packed pages
    features: Features = "eagle3"   # which tensors were packed, hence `dim`
    content: str = ""         # _content_key of the trace files at build time

    @property
    def dim(self) -> int:
        return int(self.taps.shape[1])

    @property
    def positions(self) -> int:
        return int(self.taps.shape[0])

    def rows_for(self, infos: Sequence[TraceInfo]) -> np.ndarray:
        """Row indices into ``starts``/``names`` for ``infos``, in the order given."""
        where = {n: i for i, n in enumerate(self.names.tolist())}
        missing = [page_key(i) for i in infos if page_key(i) not in where]
        if missing:
            raise KeyError(
                f"{len(missing)} page(s) are not in the pack at {self.root} (e.g. {missing[0]}); "
                "the pack was built from a different trace directory -- rebuild it with "
                "`pack_traces(scan_traces(TRACES), out_dir)`")
        return np.array([where[page_key(i)] for i in infos], dtype=np.int64)


def _pack_meta(out_dir: Path) -> Path:
    return Path(out_dir) / "pack.json"


def _content_key(infos: Sequence[TraceInfo]) -> str:
    """Fingerprint of the trace *files*, not just which pages they are.

    ``corpus_key`` answers "same page set?", which is the right guard for a vocabulary cache but
    not for a pack: re-running ``fastdocling-extract`` over the same images rewrites every file
    under the same name, and a pack keyed on names alone would keep serving the previous
    extraction's tap values.  Size and mtime are enough to notice that without re-reading 13 GB.
    """
    parts = []
    for i in sorted(infos, key=page_key):
        st = i.path.stat()
        parts.append(f"{page_key(i)}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def pack_traces(infos: Sequence[TraceInfo], out_dir: Path | str, features: Features = "eagle3",
                progress=None) -> PackedCorpus:
    """Write ``infos`` into a packed corpus under ``out_dir`` and return it.

    ``progress`` is called as ``progress(done, total)`` if given.  Overwrites any existing pack.
    """
    from safetensors import safe_open

    out_dir = Path(out_dir)
    keys = EAGLE3_TAPS if features == "eagle3" else ("last_hidden_state",)
    dim = input_dim(features)
    total = sum(i.completion_states for i in infos)
    # Both keys first, before a byte is written: corpus_key refuses a page name that appears in
    # two roots, and computing it at the end meant discovering that after a 19 s, 10.3 GB build.
    key, content = corpus_key(infos), _content_key(infos)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build into temporaries and rename into place.  Writing taps.npy directly would truncate the
    # file a previously returned PackedCorpus still has mapped, and the next read through that
    # mapping faults with SIGBUS; a rename leaves the old inode alive for as long as it is mapped.
    # Dropping the metadata first means a build that dies halfway leaves no loadable pack behind.
    _pack_meta(out_dir).unlink(missing_ok=True)
    staging = out_dir / ".building"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    taps = None
    try:
        taps = np.lib.format.open_memmap(staging / "taps.npy", mode="w+", dtype=np.float16,
                                         shape=(total, dim))
        tokens = np.zeros(total, dtype=np.int32)
        starts = np.zeros(len(infos) + 1, dtype=np.int64)
        off = 0
        for n, info in enumerate(infos):
            if features == "eagle3" and not info.has_taps:
                raise ValueError(f"{info.path.name} has no layer taps; re-extract with --keep-taps")
            # safe_open, not load_file: the eagle3 pack wants three of the four stored tensors,
            # and materialising last_hidden_state too read ~33% more bytes than the build needs.
            with safe_open(str(info.path), framework="np") as f:
                first = info.prompt_len - 1 - info.state_offset
                block = np.concatenate([f.get_slice(k)[first:] for k in keys], axis=-1)
                tok = f.get_slice("token_ids")[info.prompt_len - 1 :]
            # completion_states is derived from the header, so a disagreement here means the file
            # itself is malformed.  Clamping instead would leave off < total, and a pack whose row
            # count disagrees with its metadata is one load_packed rejects forever -- the failure
            # would be a permanently poisoned cache rather than a named bad file.
            if not block.shape[0] == tok.shape[0] == info.completion_states:
                raise ValueError(
                    f"{info.path.name} is malformed: {block.shape[0]} stored states and "
                    f"{tok.shape[0]} tokens in the DocTags region, but its header implies "
                    f"{info.completion_states}; re-extract this page")
            starts[n] = off
            taps[off : off + block.shape[0]] = block
            tokens[off : off + tok.shape[0]] = tok
            off += block.shape[0]
            if progress is not None:
                progress(n + 1, len(infos))
        starts[len(infos)] = off

        taps.flush()
        taps = None                       # drop the mapping before renaming the file under it
        np.save(staging / "tokens.npy", tokens[:off])
        np.save(staging / "starts.npy", starts)
        np.save(staging / "names.npy", np.array([page_key(i) for i in infos]))
        for name in ("taps", "tokens", "starts", "names"):
            os.replace(staging / f"{name}.npy", out_dir / f"{name}.npy")
    finally:
        # Without this a failed build leaves the full-size .building/taps.npy behind -- 10.3 GB on
        # the real corpus -- plus a live mapping of it held by the traceback frame.
        del taps
        shutil.rmtree(staging, ignore_errors=True)

    _pack_meta(out_dir).write_text(json.dumps(
        {"version": PACK_VERSION, "features": features, "dim": dim, "positions": int(off),
         "pages": len(infos), "key": key, "content": content}, indent=2))
    return load_packed(out_dir)


def load_packed(out_dir: Path | str) -> PackedCorpus:
    """Open an existing pack read-only (the taps stay memmapped, never read into RAM)."""
    out_dir = Path(out_dir)
    meta_path = _pack_meta(out_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"no pack at {out_dir.resolve()} -- build one with pack_traces()")
    meta = json.loads(meta_path.read_text())
    if meta.get("version") != PACK_VERSION:
        raise ValueError(f"pack at {out_dir} is version {meta.get('version')}, expected "
                         f"{PACK_VERSION}; rebuild it")
    taps = np.load(out_dir / "taps.npy", mmap_mode="r")
    tokens = np.load(out_dir / "tokens.npy")
    starts = np.load(out_dir / "starts.npy")
    names = np.load(out_dir / "names.npy")
    if (taps.shape[0] != meta["positions"] or starts[-1] != meta["positions"]
            or taps.shape[1] != meta["dim"] or len(names) + 1 != len(starts)):
        raise ValueError(f"pack at {out_dir} is inconsistent with its metadata "
                         f"({taps.shape} vs {meta['positions']}x{meta['dim']}); rebuild it")
    return PackedCorpus(out_dir, taps, tokens, starts, names, meta["key"],
                        meta["features"], meta.get("content", ""))


def ensure_packed(infos: Sequence[TraceInfo], out_dir: Path | str, features: Features = "eagle3",
                  progress=None) -> PackedCorpus:
    """Load the pack at ``out_dir`` if it already covers ``infos``, else build it.

    The guard is ``corpus_key``, so adding or removing traces rebuilds and a re-run does not.
    """
    out_dir = Path(out_dir)
    if _pack_meta(out_dir).exists():
        try:
            pack = load_packed(out_dir)
        except (ValueError, FileNotFoundError, KeyError):
            pack = None
        # All three have to match.  The page set alone is not enough: `features` decides the width
        # of every row, and `_content_key` catches a re-extraction that rewrote the same file
        # names with different values -- which would otherwise train on the previous extraction.
        if (pack is not None and pack.features == features
                and pack.key == corpus_key(infos) and pack.content == _content_key(infos)):
            return pack
    return pack_traces(infos, out_dir, features, progress)


def packed_batches(
    pack: PackedCorpus,
    infos: Sequence[TraceInfo],
    context_length: int,
    batch_size: int,
    horizon: int = 4,
    seed: int = 0,
    first_offset: int = FIRST_OFFSET,
    drop_last: bool = True,
    pin: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """One epoch of ``(inputs[B, ctx, dim] float16, target_tokens[B, ctx, horizon] int64)``.

    Same windowing as ``iterate_windows`` -- non-overlapping, from a per-page random offset that
    moves with ``seed`` -- but every window in the epoch is shuffled together rather than within a
    32-file buffer, and inputs stay float16 for the caller to cast on the device.  Target states
    are not produced at all; EAGLE-3 does not train on them.

    ``pin`` allocates the batch in pinned memory and fills it in place, so a CUDA consumer gets an
    async-copyable buffer without the extra host-to-host copy a later ``.pin_memory()`` would make
    (measured 3.3 ms per batch, against 2.8 ms for the rest of the producer put together).  It
    needs a CUDA context, so it stays off by default.
    """
    window = context_length + horizon + first_offset - 1
    rng = np.random.default_rng(seed)
    pages = pack.rows_for(infos)
    spans = []
    for p in pages:
        a, b = int(pack.starts[p]), int(pack.starts[p + 1])
        length = b - a
        if length < window:
            continue
        # Random phase, then non-overlapping windows -- successive epochs cut different boundaries.
        start = a + int(rng.integers(0, length - (length // window) * window + 1))
        spans.append(np.arange(start, b - window + 1, window, dtype=np.int64))
    if not spans:
        raise ValueError(f"no page in the pack has {window} usable positions for "
                         f"context_length={context_length}, horizon={horizon}")
    idx = np.concatenate(spans)
    rng.shuffle(idx)

    offsets = np.arange(first_offset, first_offset + horizon, dtype=np.int64)
    taps, tokens, dim = pack.taps, pack.tokens, pack.dim
    limit = len(idx) if not drop_last else len(idx) - len(idx) % batch_size
    rows = np.arange(context_length, dtype=np.int64)[None, :, None]
    for s in range(0, limit, batch_size):
        chunk = idx[s : s + batch_size]
        n = len(chunk)
        # Fill the torch tensor through its numpy view, so `pin` costs nothing extra: one
        # contiguous memcpy per window, straight into the buffer the H2D copy will read.
        x = torch.empty((n, context_length, dim), dtype=torch.float16, pin_memory=pin)
        xv = x.numpy()
        for j, start in enumerate(chunk):
            xv[j] = taps[start : start + context_length]
        tok = torch.empty((n, context_length, horizon), dtype=torch.int64, pin_memory=pin)
        tok.numpy()[:] = tokens[chunk[:, None, None] + rows + offsets]
        yield x, tok


def prefetch_to_device(batches: Iterator, device: torch.device | str, depth: int = 4) -> Iterator:
    """Run ``batches`` on a worker thread and hand the consumer device tensors.

    The training loop is otherwise strictly serial: nothing reads the corpus while the GPU works.
    One thread is enough -- safetensors, numpy and torch all drop the GIL for the copies -- and it
    is what turns the packed loader's ~3 ms of per-batch CPU into ~0.6 ms of observed wait.

    Batches are pinned on the worker thread, where the copy is free to the consumer, and then
    copied with ``non_blocking=True``.  Pinning needs a CUDA context and buys nothing for a CPU
    destination, so it is skipped unless the target device is CUDA -- and skipped again for a
    producer that already handed back pinned memory (``packed_batches(pin=True)`` does).  On exit,
    normal or not, the queue is drained so the worker cannot be left blocked on ``put``.
    """
    import queue
    import threading

    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")   # 0 means *unbounded* to Queue
    want_pin = torch.device(device).type == "cuda"
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    DONE = object()

    def pinned(t: torch.Tensor) -> torch.Tensor:
        return t.pin_memory() if want_pin and not t.is_pinned() else t

    def work():
        try:
            for b in batches:
                if stop.is_set():
                    break
                q.put(tuple(pinned(t) for t in b))
        except BaseException as exc:                      # surface it on the consumer thread
            q.put(exc)
        finally:
            # Close the upstream generator here rather than leaving it suspended until GC, which
            # in a notebook can be a long time -- it holds the memmap slices of a whole batch.
            close = getattr(batches, "close", None)
            if close is not None:
                close()
            q.put(DONE)

    worker = threading.Thread(target=work, name="fastdocling-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = q.get()
            if item is DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield tuple(t.to(device, non_blocking=True) for t in item)
    finally:
        stop.set()
        while worker.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                worker.join(timeout=0.05)
        worker.join()


def split(infos: Sequence[TraceInfo], holdout_fraction: float = 0.02, seed: int = 0) -> tuple[list[TraceInfo], list[TraceInfo]]:
    """Deterministic train/holdout split by page."""
    order = list(infos)
    if len(order) < 2:
        raise ValueError(f"need at least 2 pages to split, got {len(order)}")
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


def corpus_key(infos: Sequence[TraceInfo]) -> str:
    """Short fingerprint of a page set (by file name), for caches that depend on which pages were used.

    Names, not paths, so the fingerprint survives moving a corpus between machines -- which means
    two roots holding a same-named page would fingerprint as one.  Extraction names pages after
    their source (``ml_papers__…``, ``doclingmatix__…``) so that cannot happen by accident; if it
    ever does, say so rather than hand back a key that silently aliases two different corpora.
    """
    names = sorted(i.path.name for i in infos)
    dupes = {n for n, m in zip(names, names[1:]) if n == m}
    if dupes:
        raise ValueError(
            f"{len(dupes)} page name(s) appear in more than one trace root, e.g. "
            f"{sorted(dupes)[:3]}; rename one corpus -- a cache key cannot tell them apart")
    return hashlib.sha1("\n".join(names).encode()).hexdigest()[:10]


def key_of(**parts) -> str:
    """Stable sha1 of a dict of identifying values (JSON, sorted keys); non-JSON values are str()'d."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    """``fastdocling-pack <traces> <out>`` -- build a packed corpus ahead of a training run."""
    import argparse

    ap = argparse.ArgumentParser(description="Pack traces into one contiguous float16 array.")
    ap.add_argument("traces", type=Path, nargs="+", help="one or more directories of traces")
    ap.add_argument("out", type=Path, help="where to write the pack")
    ap.add_argument("--features", default="eagle3", choices=("eagle3", "last"))
    ap.add_argument("--min-completion", type=int, default=66,
                    help="drop pages with fewer usable DocTags positions than this; it must be at "
                         "least the training window, context_length + horizon + first_offset - 1 "
                         "(default 66 = the eagle3 loop's 64 + 2 + 1 - 1)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the pack is current")
    args = ap.parse_args(argv)

    infos = scan_traces(args.traces, min_completion=args.min_completion)
    total = sum(i.completion_states for i in infos)
    print(f"{len(infos):,} pages, {total:,} positions -> "
          f"{total * input_dim(args.features) * 2 / 1e9:.2f} GB")
    if len(args.traces) > 1:
        print("  from " + ", ".join(f"{k}: {v:,}" for k, v in sorted(source_counts(infos).items())))
    build = pack_traces if args.force else ensure_packed
    last = -1

    def progress(done: int, n: int) -> None:
        nonlocal last
        pct = done * 100 // n
        if pct != last:
            last = pct
            print(f"\r  packing {pct:3d}%  ({done:,}/{n:,})", end="", flush=True)

    pack = build(infos, args.out, args.features, progress)
    print(f"\r{pack.positions:,} positions at {pack.dim}-d in {pack.root}          ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
