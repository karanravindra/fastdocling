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

```sh
uv sync
hf auth login                                   # as karanravindra
uv run python scripts/sync_data.py pull         # or: pull docs traces checkpoints
```

`sync_data.py push [target ...]` uploads new or changed files. Run `sync_data.py ls` to see targets.

## Pipeline

```sh
uv run fastdocling-prep render data/docs data/images     # PDFs -> PNGs
uv run fastdocling-extract data/images data/traces       # PNGs -> DocTags + traces
# train / evaluate the draft in train.ipynb
```
