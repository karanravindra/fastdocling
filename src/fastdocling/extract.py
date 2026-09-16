"""Extract EAGLE-3 training traces from granite-docling.

For every page we record the target's residual stream at each generated position, which
``fastdocling.data`` cuts into draft-training windows.  Labels are the model's *own* greedy
DocTags (so the draft learns the target's distribution, as in EAGLE); in the default
image-directory mode the states are recorded *during* decoding, so no second pass is needed.

Manifest mode supplies the labels instead, and costs one teacher-forced prefill per page
rather than a full decode -- the difference between a corpus that takes minutes and one that
takes hours.  It buys that with label quality: states recorded against somebody else's
DocTags are off-policy wherever those DocTags disagree with what the target would have said.
Measured against ``HuggingFaceM4/DoclingMatix`` (whose labels come from the Docling pipeline,
and which granite-docling was trained on), agreement is 95.4% of label tokens -- so keep
prefilled traces in their own directory, away from the generated ones.  See the README.

The work is done by a backend -- ``mlx`` on Apple Silicon, ``vllm`` on CUDA -- chosen
automatically.  Both write the same trace format, so a corpus can be extracted on either.

Usage:
    fastdocling-extract data/images data/traces          # generate DocTags + traces for every PNG
    fastdocling-extract manifest.jsonl data/traces       # use provided labels instead
    fastdocling-extract data/images data/traces --backend vllm
    # manifest lines: {"image": "data/images/x/0001.png", "doctags": "<doctag>...</doctag>"}
    # or               {"image": ..., "doctags_path": "labels/x/0001.dt"}
    # an explicit {"name": ...} sets the trace filename; otherwise it is derived from the
    # image path relative to the manifest's directory, so nested pages stay collision-free.
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
    """Rows of ``{image, doctags | doctags_path, name?}``, with labels read in and names filled.

    ``name`` becomes the trace filename, so it has to be unique across the manifest.  A row
    that does not carry one gets the same flat, collision-free name the directory mode builds
    (``<dir>__<dir>__<page>``), relative to the manifest's own directory -- the bare image stem
    would collapse ``a/0001.png`` and ``b/0001.png`` onto one trace.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    root = path.parent
    for r in rows:
        if "doctags" not in r:
            r["doctags"] = Path(r["doctags_path"]).read_text()
        if not r.get("name"):
            r["name"] = trace_name(Path(r["image"]), root)
    seen = {}
    for i, r in enumerate(rows):
        if r["name"] in seen:
            raise ValueError(
                f"{path}: duplicate trace name {r['name']!r} on lines {seen[r['name']] + 1} "
                f"and {i + 1}; give the rows explicit distinct \"name\" fields")
        seen[r["name"]] = i
    return rows


def _drop_overlong(rows: list[dict], ex, limit: int) -> tuple[list[dict], int]:
    """Drop manifest pages whose image prefix plus label exceeds the model's context window.

    Borrowed labels come with no length guarantee: a dense page rendered large enough turns into
    a long vision prefix, and a long label on top of that overruns granite-docling's 8192
    positions.  The engine rejects such a request outright, which kills the whole run -- one bad
    page out of ten thousand should cost that page, not the corpus.

    The vision prefix depends only on the image's *size* (Idefics3 tiles by dimensions), so the
    measurement is cached per size and costs one processor call per distinct page geometry --
    a handful for a whole corpus -- rather than one per page.  ``Image.open`` reads the header
    without decoding pixels, so the scan itself is cheap.
    """
    tok = getattr(ex, "tokenizer", None)
    prompt_len_of = getattr(ex, "_prompt_token_len", None)
    if tok is None or prompt_len_of is None or not limit:
        return rows, 0            # backend cannot tell us; let the engine decide

    from PIL import Image

    by_size: dict[tuple[int, int], int] = {}
    kept, dropped = [], 0
    for r in rows:
        with Image.open(r["image"]) as im:
            size = im.size
        if size not in by_size:
            by_size[size] = prompt_len_of(Image.new("RGB", size))
        total = by_size[size] + len(tok.encode(r["doctags"], add_special_tokens=False))
        if total > limit:
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


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
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="vLLM only: fraction of the GPU to claim (default 0.85). Lower it when "
                         "something else already holds memory on the card, or the engine will "
                         "fail to allocate its KV cache")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="vLLM only: skip CUDA graph capture. Slower per step, but it frees the "
                         "1-2 GB the capture reserves -- often the difference between fitting on "
                         "a shared card and not fitting at all")
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
    if args.gpu_memory_utilization is not None:
        kw["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.enforce_eager:
        kw["enforce_eager"] = True
    ex = get_backend(args.backend, **kw)
    backend = args.backend if args.backend != "auto" else detect()

    if args.source.suffix == ".jsonl":
        rows = _read_manifest(args.source)
    else:
        rows = [{"image": str(p), "name": trace_name(p, args.source)}
                for p in sorted(args.source.rglob("*.png"))]
    total = len(rows)
    rows = [r for r in rows if not (args.out / f"{r['name']}.safetensors").exists()]
    skipped = total - len(rows)
    overlong = 0
    if args.source.suffix == ".jsonl":
        rows, overlong = _drop_overlong(rows, ex, getattr(ex, "max_model_len", 0))
    print(f"backend={backend} taps={ex.taps} pages={len(rows)} (skipping {skipped} already extracted"
          + (f", {overlong} too long for the context window" if overlong else "") + ")",
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
                # Mirror the generated mode and drop the label next to the trace, so a trace
                # directory can always be read back without its manifest.
                (args.out / f"{chunk[i]['name']}.dt").write_text(chunk[i]["doctags"])
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
