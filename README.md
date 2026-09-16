# fastdocling

Speculative decoding for [granite-docling](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)
with a small latent draft model.

This repository holds **code only**. Everything else lives in private datasets on the Hugging Face
Hub under `karanravindra`:

| Hub dataset | Contents | Local path |
|---|---|---|
| `karanravindra/fastdocling-data` | `docs/` source PDFs, `ood/` out-of-domain PDFs + renders, `images/` rendered pages, `prefill/` borrowed-label pages + manifests, `checkpoints/` trained drafts, `cache/` trained runs and decodes | `data/docs`, `data/ood`, `data/images`, `data/prefill`, `checkpoints`, `data/cache` |
| `karanravindra/fastdocling-traces` | per-page hidden-state traces (`.safetensors`) and DocTags (`.dt`) | `data/traces` |
| `karanravindra/fastdocling-traces-prefill` | same format, but recorded against *borrowed* labels -- see "Borrowed labels" below | `data/traces_prefill` |

## Setup

Extraction runs the target model, so it needs a backend, and the two are mutually exclusive
extras -- vLLM pins `torch==2.13.0` while MLX is happier on 2.14:

```sh
uv sync --extra mlx     # Apple Silicon (mlx-vlm, torch 2.14)
uv sync --extra cuda    # NVIDIA        (vllm,    torch 2.13)

hf auth login                                   # as karanravindra
uv run python scripts/sync_data.py pull         # or: pull docs traces checkpoints
```

Everything except `fastdocling-extract` works with a plain `uv sync`. Note that `uv run`
without the extra re-syncs the environment back to the base set, so pass it every time:
`uv run --extra cuda fastdocling-extract ...`.

`sync_data.py push [target ...]` uploads new or changed files. Run `sync_data.py ls` to see targets.

## Pipeline

```sh
uv run fastdocling-prep render data/docs data/images          # PDFs -> PNGs  (needs ghostscript)
uv run --extra cuda fastdocling-extract data/images data/traces   # PNGs -> DocTags + traces
# train / evaluate the draft in train.ipynb
```

`fastdocling-extract` picks a backend automatically (`--list-backends` shows what this
machine has; `--backend mlx|vllm` forces one). Both write the same trace format, so a
corpus can be extracted on either machine and pushed to the same Hub dataset.

Add `--keep-taps` to store the EAGLE-3 layer taps (`layer_2/14/27`) alongside the final
state -- needed for `features="eagle3"` in `data.py`, and roughly 4x the bytes.

Rough cost on one RTX 5070 Ti: ~3.7 pages/s after a ~45 s engine start, so the 5,379-page
corpus is ~25 min and ~6 GB of traces (~25 GB with `--keep-taps`).

On a card someone else is already using, `--gpu-memory-utilization` (a fraction of *total* VRAM,
so it must stay below what is actually free) and `--enforce-eager` (skips CUDA-graph capture,
which reserves 1-2 GB) are the two knobs that decide whether the engine starts at all.

## Borrowed labels

Generating DocTags is what makes extraction slow, and some Hub datasets already ship them. Point
`fetch_doctags.py` at one and `fastdocling-extract` costs a single teacher-forced prefill per
page instead of a full decode:

```sh
uv run python scripts/fetch_doctags.py --shards 5             # -> data/prefill/doclingmatix/
uv run --extra cuda fastdocling-extract \
    data/prefill/doclingmatix/manifest.jsonl data/traces_prefill --keep-taps
```

Measured on one box: 5 shards is **10,727 pages in 38 s** (29 s of download at ~97 MB/s, 9 s of
conversion), then ~20 min of extraction at 7-9 pages/s. The fetch is fast because it downloads
whole parquet shards and copies the page images out *already encoded* -- decoding to pixels and
re-encoding cost more than everything else combined. `--max-side` forces a real decode, so it is
off by default. `HF_XET_HIGH_PERFORMANCE` does nothing here: the `refs/convert/parquet` branch is
plain LFS, not Xet-backed.

Sizing, if you are choosing a shard count: a shard is 1,152 rows, ~1.92 pages each, so **~2,200
pages** and ~2.2M label tokens per shard. Traces cost 1.13 KB/token without taps and 4.51 with
(4.95 MB/page), so 5 shards with `--keep-taps` is ~53 GB. DoclingMatix has 1,104 shards in total,
about 2.4B label tokens.

The default source is [`HuggingFaceM4/DoclingMatix`](https://hf.co/datasets/HuggingFaceM4/DoclingMatix),
one of the four datasets granite-docling was trained on, whose rows carry a
`"Convert this page to docling."` turn -- the same prompt `backends/base.py` uses. Rows are whole
documents, so the script splits each one on `<page_break>`, re-wraps every page as a standalone
`<doctag>` document, and drops any row whose segment count does not match its image count rather
than risk pairing a label with the wrong page.

**These labels are not the target's own output, which is why they live in their own directory.**
Measured on 20 DoclingMatix pages (19,571 label tokens), granite-docling's greedy argmax agrees
with the stored label for **95.4%** of positions:

| token class | agreement | share of label |
|---|---:|---:|
| `<`,`loc`,`_`,`>` scaffolding | 1.000 | 18.2% |
| coordinate digits | 0.674 | 7.3% |
| structural tags | 0.926 | 4.4% |
| page text | 0.973 | 70.2% |

About half the disagreement is coordinate digits -- the label's boxes come from Docling's layout
model, not from granite-docling. That 4.6% is off-policy supervision, but the draft's own
per-token error is far larger (the EAGLE-3 draft accepts 0.72-0.87 tokens per step at k=2, so
roughly 0.5 per token), so label noise is not the binding constraint today. Revisit if acceptance
ever climbs past ~0.85 per token.

Keep the two corpora apart on disk and mix them explicitly, so a run can always be reproduced
with the clean corpus alone:

```python
from fastdocling.data import scan_traces, source_counts
infos = scan_traces(["data/traces", "data/traces_prefill"])
source_counts(infos)      # {'traces': 5200, 'traces_prefill': 12}
```

Pass `--keep-taps` if these pages are meant for `features="eagle3"`; without it they carry only
the final normed state and `load_trace` will refuse them.

Borrowed labels come with no length guarantee, and granite-docling's context is a hard 8192
(`max_position_embeddings`), so a long enough label overruns it. vLLM rejects such a request at
submission and the exception takes the whole run down -- one 20,816-token DoclingMatix label once
killed a 10,727-page extraction at page 512. Manifest mode now measures every page first and drops
the ones that cannot fit, reporting the count at startup (24 of 10,727 here, 0.22%). The
measurement is cheap: `Image.open` reads the header without decoding, and the vision prefix turns
out to be a constant 1,142 tokens for every page -- Idefics3 normalizes its tiling, so page
dimensions do not matter and only the label length does.

### Where the data lives

The corpora outgrew the repo disk, so `data/traces`, `data/traces_taps`, `data/traces_prefill`,
`data/prefill` and `data/cache` are symlinks into `/mnt/ai/data/fastdocling/`. Every path in the
notebooks and in `sync_data.py` works unchanged through them. Two things to keep in mind: the
EAGLE-3 *pack* is ~3.5 KB per position (1728-d float16), so a merged corpus runs to tens of GB and
has to be on that disk; and `HF_HOME` already points at `/mnt/ai/data/hf`, so the parquet cache
lands there too.

## Measuring the draft

Extraction is throughput-bound and batches everything it can; measuring speculative decoding is
the opposite job -- one page at a time, latency only. `fastdocling.decode` has its own backends
for that, selected by `DECODE_BACKEND` in `train.ipynb`:

| backend | runtime | install | speculative loop |
|---|---|---|---|
| `mlx` | mlx-vlm | `uv sync --extra mlx` | yes |
| `transformers` | HF eager | plain `uv sync` (CUDA / MPS / CPU) | yes |
| `vllm` | vLLM | `uv sync --extra cuda` | no -- baseline only |

`auto` picks `mlx` on Apple Silicon and `transformers` everywhere else, so the draft can be
measured on the training box without an MLX install. vLLM is never picked automatically: it owns
its KV cache inside the scheduler and exposes no rollback, so the draft cannot propose into it.
What it does measure is the target's honest single-stream decode rate, which is the right
denominator for a speedup. The two are far apart -- on one RTX 5070 Ti, same model and page,
`transformers` decodes at 57 tok/s (17.5 ms/step) and `vllm` at ~1,150 tok/s (0.87 ms/step),
because HF eager on a 258M model is launch-bound rather than compute-bound. A speedup quoted over
the slow baseline flatters the draft. Compare acceptance rates across backends; compare wall-clock
only within one.
