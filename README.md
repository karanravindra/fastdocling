# fastdocling

Speculative decoding for [granite-docling](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)
with a small latent draft model.

This repository holds **code only**. Everything else lives in private datasets on the Hugging Face
Hub under `karanravindra`:

| Hub dataset | Contents | Local path |
|---|---|---|
| `karanravindra/fastdocling-data` | `docs/` source PDFs, `ood/` out-of-domain PDFs + renders, `images/` rendered pages, `checkpoints/` trained drafts, `cache/` trained runs and decodes | `data/docs`, `data/ood`, `data/images`, `checkpoints`, `data/cache` |
| `karanravindra/fastdocling-traces` | per-page hidden-state traces (`.safetensors`) and DocTags (`.dt`) | `data/traces` |

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

Rough cost on one RTX 5070 Ti: ~1 page/s after a ~45 s engine start, so the 5,379-page
corpus is ~90 min and ~7 GB of traces (~27 GB with `--keep-taps`).
