"""Cache extracted traces on the Hugging Face Hub so nobody recomputes them.

    uv run python scripts/sync_traces.py push [--repo USER/fastdocling-traces]   # upload new/changed files
    uv run python scripts/sync_traces.py pull [--repo ...]                        # download into data/traces

Push uses ``upload_large_folder``: resumable, parallel, skips files already on the Hub, and
safe to re-run while ``fastdocling-extract`` is still producing pages (it only uploads files
that exist and are unchanged between hash and upload).  The repo is created private if missing.
Login once with ``hf auth login`` (or set HF_TOKEN).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_REPO = "karan-wattsai/fastdocling-traces"
TRACES = Path("data/traces")


def push(repo: str, folder: Path, workers: int) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    n = sum(1 for _ in folder.glob("*.safetensors"))
    print(f"uploading {folder} ({n} pages) -> hf://datasets/{repo}", file=sys.stderr)
    api.upload_large_folder(
        repo_id=repo, repo_type="dataset", folder_path=str(folder),
        allow_patterns=["*.safetensors", "*.dt", "*.npy"], num_workers=workers, print_report=True,
    )


def pull(repo: str, folder: Path, workers: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(folder), max_workers=workers)
    print(f"{sum(1 for _ in folder.glob('*.safetensors'))} pages in {folder}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["push", "pull"])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--folder", type=Path, default=TRACES)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    (push if a.command == "push" else pull)(a.repo, a.folder, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
