"""Extract EAGLE-3 training traces from granite-docling.

For every page we record the target's residual stream at each generated position, which
``fastdocling.data`` cuts into draft-training windows.  Labels are the model's *own* greedy
DocTags (so the draft learns the target's distribution, as in EAGLE); in the default
image-directory mode the states are recorded *during* decoding, so no second pass is needed.

The work is done by a backend -- ``mlx`` on Apple Silicon, ``vllm`` on CUDA -- chosen
automatically.  Both write the same trace format, so a corpus can be extracted on either.

Usage:
    fastdocling-extract data/images data/traces          # generate DocTags + traces for every PNG
    fastdocling-extract manifest.jsonl data/traces       # use provided labels instead
    fastdocling-extract data/images data/traces --backend vllm
    # manifest lines: {"image": "data/images/x/0001.png", "doctags": "<doctag>...</doctag>"}
    # or               {"image": ..., "doctags_path": "labels/x/0001.dt"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from .backends import END_TAG, MAX_NEW_TOKENS, available, detect, get_backend, trace_name


def __getattr__(name: str):
    """Lazily re-export the MLX extractor so ``train.ipynb``'s import still works.

    Kept lazy because importing it on a CUDA box raises ImportError, which would otherwise
    take the whole CLI down with it.
    """
    if name == "TraceExtractor":
        from .backends.mlx_backend import TraceExtractor

        return TraceExtractor
    raise AttributeError(name)


def _read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if "doctags" not in r:
            r["doctags"] = Path(r["doctags_path"]).read_text()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="directory of page PNGs, or a .jsonl manifest with labels")
    ap.add_argument("out", type=Path)
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "vllm"],
                    help="target runtime (default: auto; see --list-backends)")
    ap.add_argument("--model", default=None, help="override the target model id (default: per-backend)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="pages per teacher-forced sweep (default: per-backend)")
    ap.add_argument("--gen-batch", type=int, default=None,
                    help="pages handed to the backend per call (default: per-backend; "
                         "on vLLM this bounds disk and crash-loss, not GPU batching)")
    ap.add_argument("--chunk", type=int, default=16, help="pages per prefill sweep (manifest mode)")
    ap.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--keep-truncated", action="store_true", help="keep pages that never emitted </doctag>")
    ap.add_argument("--keep-taps", action="store_true", help="also store EAGLE-3 layer taps (layer_2/14/27); 4x larger")
    ap.add_argument("--list-backends", action="store_true", help="show which backends import here, then exit")
    args = ap.parse_args(argv)

    if args.list_backends:
        print(f"detected: {detect()}   importable: {', '.join(available()) or 'none'}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    kw = {"keep_taps": args.keep_taps}
    if args.model:
        kw["model_id"] = args.model
    ex = get_backend(args.backend, **kw)
    backend = args.backend if args.backend != "auto" else detect()

    if args.source.suffix == ".jsonl":
        rows = _read_manifest(args.source)
        for r in rows:
            r["name"] = Path(r["image"]).stem
    else:
        rows = [{"image": str(p), "name": trace_name(p, args.source)}
                for p in sorted(args.source.rglob("*.png"))]
    total = len(rows)
    rows = [r for r in rows if not (args.out / f"{r['name']}.safetensors").exists()]
    skipped = total - len(rows)
    print(f"backend={backend} taps={ex.taps} pages={len(rows)} (skipping {skipped} already extracted)",
          file=sys.stderr)

    # Generation dominates runtime, so the bar advances per page once its label is ready;
    # the batched prefill at the end of each chunk is a fraction of a second per page.
    bar = tqdm(total=len(rows), unit="page", dynamic_ncols=True, smoothing=0.05)
    truncated = 0
    if args.source.suffix == ".jsonl":
        from transformers.image_utils import load_image

        # labels supplied: one teacher-forced prefill per page
        for c0 in range(0, len(rows), args.chunk):
            chunk = rows[c0 : c0 + args.chunk]
            items = [(load_image(r["image"]), r["doctags"]) for r in chunk]
            for i, trace in (ex.extract(items) if args.batch_size is None
                             else ex.extract(items, args.batch_size)):
                ex.save(args.out / f"{chunk[i]['name']}.safetensors", trace)
                bar.update()
    else:
        # no labels: greedy-decode, recording taps during decoding.
        from transformers.image_utils import load_image

        if ex.needs_length_sorted_input:
            # MLX prefills a refill group unpadded, so a group must share a prompt length --
            # which depends only on image size.  vLLM schedules per request and gains
            # nothing, so skip a full pass of PIL opens over the corpus.
            from PIL import Image

            rows.sort(key=lambda r: Image.open(r["image"]).size)
        images = (load_image(r["image"]) for r in rows)
        gen_args = () if args.gen_batch is None else (args.gen_batch,)
        for i, tr in ex.generate_traces(images, *gen_args, max_tokens=args.max_tokens):
            r = rows[i]
            (args.out / f"{r['name']}.dt").write_text(tr.pop("doctags"))
            if tr.pop("truncated") and not args.keep_truncated:
                truncated += 1
            else:
                ex.save(args.out / f"{r['name']}.safetensors", tr)
            bar.set_postfix_str(r["name"][-30:], refresh=False)
            bar.update()
    bar.close()
    if truncated:
        print(f"{truncated} pages skipped: no {END_TAG} within --max-tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
