"""Turn a Hub dataset that already carries DocTags into a prefill manifest.

``fastdocling-extract`` in manifest mode needs labels, not just images, and costs one
teacher-forced prefill per page instead of a full decode.  This script produces that manifest
from a dataset whose rows hold ``images`` plus a ``texts`` conversation ending in a
"Convert this page to docling." turn -- the shape shared by DoclingMatix and the Synth* sets
granite-docling was trained on.

    uv run python scripts/fetch_doctags.py --shards 5
    uv run --extra cuda fastdocling-extract data/prefill/doclingmatix/manifest.jsonl \\
        data/traces_prefill --keep-taps

Shards are whole parquet files, downloaded with ``hf_hub_download`` and read locally with
pyarrow.  That is the fast path, and it is fast for two reasons: the download is a plain
sequential CDN fetch (measured at ~97 MB/s, so a 800 MB shard lands in ~8 s), and the page
images are written out as the *encoded bytes already sitting in the parquet* rather than being
decoded to pixels and re-encoded.  ``--stream`` keeps the old row-by-row reader for a dataset
with no parquet conversion; it is far slower and caches nothing.

Writes, under ``--out``:

    images/<name>.jpg     one page each, in whatever format the dataset stored
    labels/<name>.dt      that page's DocTags, re-wrapped as a standalone <doctag> document
    manifest.jsonl        {"image": ..., "doctags_path": ..., "name": ..., "shard": ..., ...}

Names are ``<tag>__s<shard>__r<row>__p<page>``, so shards can be fetched in any order or
re-fetched without renaming anything, and a name always says where its page came from.

These labels are *not* the target model's own greedy output: on DoclingMatix they agree with it
for 95.4% of label tokens, with about half the disagreement in ``<loc_N>`` coordinate digits.
That is why the traces belong in their own directory -- see the README.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = "Convert this page to docling."
OPEN, CLOSE, BREAK = "<doctag>", "</doctag>", "<page_break>"
MAGIC = [(b"\xff\xd8\xff", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png"), (b"GIF8", ".gif"),
         (b"RIFF", ".webp"), (b"II*\x00", ".tif"), (b"MM\x00*", ".tif")]


def pages_of(row: dict) -> list[tuple[object, str]] | None:
    """Split one dataset row into ``(image, doctags)`` pairs, or None if it cannot be aligned.

    A row is a whole document: ``images`` is its pages and the DocTags for all of them are one
    string joined by ``<page_break>``.  The pipeline works a page at a time, so each segment is
    re-wrapped as a standalone document.  A row whose segment count does not match its image
    count cannot be aligned page-to-page and is dropped -- silently pairing them would shift
    every label in the row onto the wrong picture.

    The image objects are passed through untouched, so this works for both readers: pyarrow
    hands back ``{"bytes": ..., "path": ...}`` dicts, ``datasets`` hands back PIL images.
    """
    turns = [t for t in row.get("texts", []) if (t.get("user") or "").strip() == PROMPT]
    if not turns:
        return None
    segs = turns[0]["assistant"].split(BREAK)
    images = row.get("images") or []
    if len(segs) != len(images):
        return None
    out = []
    for img, seg in zip(images, segs):
        seg = seg.strip()
        if not seg.startswith(OPEN):
            seg = OPEN + seg
        if not seg.endswith(CLOSE):
            seg = seg + CLOSE
        out.append((img, seg))
    return out


def ext_of(data: bytes) -> str:
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    return ".png"


def write_image(img, stem: Path, max_side: int | None) -> Path:
    """Write one page, re-encoding only when we actually have to.

    A dataset image arrives already encoded.  Copying those bytes out is the whole reason the
    fast path is fast -- decoding to pixels and re-encoding as PNG costs far more per page than
    everything else here put together, and buys nothing: the extractor's image loader reads
    whatever the dataset stored.  Only ``--max-side`` forces a real decode.
    """
    raw = img.get("bytes") if isinstance(img, dict) else None
    if raw is not None and not max_side:
        path = stem.with_suffix(ext_of(raw))
        path.write_bytes(raw)
        return path
    from PIL import Image

    if raw is not None:
        import io

        img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        w, h = img.size
        s = max_side / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    path = stem.with_suffix(".jpg")
    img.save(path, quality=92)
    return path


def shard_rows(path: str, columns=("images", "texts")):
    """Yield rows of a local parquet shard, one row group at a time."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=list(columns))
        yield from table.to_pylist()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="HuggingFaceM4/DoclingMatix")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default=None, help="name prefix (default: the dataset's basename, lowercased)")
    ap.add_argument("--out", type=Path, default=None, help="default: data/prefill/<tag>")
    ap.add_argument("--shards", type=int, default=1, help="how many parquet shards to take")
    ap.add_argument("--shard-start", type=int, default=0, help="first shard index")
    ap.add_argument("--pages", type=int, default=None, help="also stop once the corpus holds this many pages")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale pages to this longest side. Off by default, which keeps the "
                         "dataset's own resolution (it varies per row, and differs from the 144 "
                         "dpi `fastdocling-prep render` produces) -- and keeps the fast path, "
                         "since resizing forces a decode and re-encode of every page")
    ap.add_argument("--workers", type=int, default=8, help="threads for downloads and page writes")
    ap.add_argument("--stream", action="store_true",
                    help="row-by-row reader for datasets with no parquet conversion (slow)")
    a = ap.parse_args(argv)

    tag = a.tag or a.dataset.split("/")[-1].lower()
    out = a.out or Path("data/prefill") / tag
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    have = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()] if manifest.exists() else []
    done = {r["name"] for r in have}
    print(f"{tag}: {len(have)} pages already in {manifest}", file=sys.stderr)

    from tqdm import tqdm

    written = unaligned = 0
    bar = tqdm(total=a.pages, initial=len(have), unit="page", dynamic_ncols=True)
    pool = ThreadPoolExecutor(max_workers=a.workers)
    fh = manifest.open("a")

    def emit(name, img, doctags, meta):
        img_path = write_image(img, out / "images" / name, a.max_side)
        dt_path = out / "labels" / f"{name}.dt"
        dt_path.write_text(doctags)
        return json.dumps({"image": str(img_path), "doctags_path": str(dt_path),
                           "name": name, "dataset": a.dataset, **meta}) + "\n"

    def consume(rows, shard_label, shard_idx):
        nonlocal written, unaligned
        pending = []
        for row_idx, row in enumerate(rows):
            pages = pages_of(row)
            if pages is None:
                unaligned += 1
                continue
            for page_idx, (img, doctags) in enumerate(pages):
                name = f"{tag}__{shard_label}__r{row_idx:05d}__p{page_idx:03d}"
                if name in done:
                    continue
                pending.append(pool.submit(emit, name, img, doctags,
                                           {"shard": shard_idx, "row": row_idx, "page": page_idx}))
                if a.pages is not None and len(have) + written + len(pending) >= a.pages:
                    break
            if len(pending) >= 256 or (a.pages is not None and len(have) + written + len(pending) >= a.pages):
                for f in pending:
                    fh.write(f.result())
                    written += 1
                    bar.update()
                fh.flush()
                pending = []
                if a.pages is not None and len(have) + written >= a.pages:
                    return True
        for f in pending:
            fh.write(f.result())
            written += 1
            bar.update()
        fh.flush()
        return a.pages is not None and len(have) + written >= a.pages

    if a.stream:
        from datasets import load_dataset

        consume(load_dataset(a.dataset, split=a.split, streaming=True), "stream", -1)
    else:
        from huggingface_hub import hf_hub_download

        idxs = list(range(a.shard_start, a.shard_start + a.shards))
        names = [f"{a.split}/{i:04d}.parquet" for i in idxs]
        print(f"downloading {len(names)} shard(s) with {a.workers} workers...", file=sys.stderr)
        paths = list(ThreadPoolExecutor(max_workers=a.workers).map(
            lambda n: hf_hub_download(a.dataset, f"default/{n}", repo_type="dataset",
                                      revision="refs/convert/parquet"), names))
        for i, p in zip(idxs, paths):
            if consume(shard_rows(p), f"s{i:04d}", i):
                break

    fh.close()
    bar.close()
    pool.shutdown(wait=True)
    print(f"{tag}: wrote {written} pages ({len(have) + written} total); "
          f"{unaligned} rows dropped (page/image count mismatch)", file=sys.stderr)
    print(f"next: fastdocling-extract {manifest} data/traces_prefill --keep-taps", file=sys.stderr)
    # Every page is already on disk (the manifest is flushed as it goes), but the streaming
    # reader keeps HTTP workers alive that nothing shuts down -- the process would otherwise
    # sit there until something kills it and look like a failure.
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
