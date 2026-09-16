"""Keep everything that is not code on the Hugging Face Hub (account: karanravindra).

GitHub holds only source, notebooks and the lockfile.  Every data folder lives in a private
Hub dataset and is synced with this script:

    uv run python scripts/sync_data.py push [TARGET ...]     # upload new/changed files
    uv run python scripts/sync_data.py pull [TARGET ...]     # download into data/ and checkpoints/
    uv run python scripts/sync_data.py ls                    # list targets

Targets (default: all of them):

    docs         data/docs        source PDFs                 -> datasets/karanravindra/fastdocling-data/docs
    ood          data/ood         out-of-domain PDFs + PNGs   -> .../fastdocling-data/ood
    images       data/images      rendered pages              -> .../fastdocling-data/images
    checkpoints  checkpoints      trained draft models        -> .../fastdocling-data/checkpoints
    cache        data/cache       trained runs, greedy decodes-> .../fastdocling-data/cache
    prefill      data/prefill     borrowed-label pages+manifest-> .../fastdocling-data/prefill
    traces       data/traces      hidden-state traces (flat)  -> datasets/karanravindra/fastdocling-traces
    traces_prefill data/traces_prefill  teacher-forced traces -> datasets/karanravindra/fastdocling-traces-prefill

``traces`` and ``traces_prefill`` are separate repos on purpose: the first holds states recorded
against the target's own greedy DocTags, the second against labels borrowed from a Hub dataset,
which agree with the target for ~95% of tokens.  Mixing them in one repo would make it
impossible to pull only the clean corpus.

Push uses ``upload_folder`` (hub >= 1.x): files are hashed and chunk-uploaded via Xet, committed
in batches, and a re-run resumes by skipping already-committed files.  Safe to re-run while
``fastdocling-extract`` or a training cell is still producing files.  Repos are created private
if missing.  Login once with ``hf auth login`` (or set HF_TOKEN).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

HF_USER = "karanravindra"
DATA_REPO = f"{HF_USER}/fastdocling-data"
TRACES_REPO = f"{HF_USER}/fastdocling-traces"
PREFILL_TRACES_REPO = f"{HF_USER}/fastdocling-traces-prefill"


@dataclass(frozen=True)
class Target:
    name: str
    folder: Path
    repo: str
    subdir: str  # "" = repo root
    patterns: tuple[str, ...]


TARGETS = {
    t.name: t
    for t in [
        Target("docs", Path("data/docs"), DATA_REPO, "docs", ("**/*.pdf",)),
        Target("ood", Path("data/ood"), DATA_REPO, "ood", ("**/*.pdf", "**/*.png")),
        Target("images", Path("data/images"), DATA_REPO, "images", ("**/*.png",)),
        Target("checkpoints", Path("checkpoints"), DATA_REPO, "checkpoints", ("**/*.pt", "**/*.npz", "**/*.safetensors", "**/*.json")),
        Target("cache", Path("data/cache"), DATA_REPO, "cache", ("**/*",)),
        Target("prefill", Path("data/prefill"), DATA_REPO, "prefill",
               ("**/*.png", "**/*.dt", "**/*.jsonl")),
        Target("traces", Path("data/traces"), TRACES_REPO, "", ("*.safetensors", "*.dt", "*.npy")),
        Target("traces_prefill", Path("data/traces_prefill"), PREFILL_TRACES_REPO, "",
               ("*.safetensors", "*.dt", "*.npy")),
    ]
}


def hub_patterns(patterns: tuple[str, ...]) -> list[str]:
    """fnmatch-style patterns for the Hub: ``**/*.pdf`` there needs a slash, so add the bare form too."""
    out: list[str] = []
    for p in patterns:
        out.append(p)
        if p.startswith("**/"):
            out.append(p[3:])
    return out


def count(folder: Path, patterns: tuple[str, ...]) -> int:
    return len({f for p in patterns for f in folder.glob(p) if f.is_file()})


def push(api: HfApi, t: Target) -> None:
    if not t.folder.is_dir() or count(t.folder, t.patterns) == 0:
        print(f"[{t.name}] nothing to upload in {t.folder}", file=sys.stderr)
        return
    api.create_repo(t.repo, repo_type="dataset", private=True, exist_ok=True)
    n = count(t.folder, t.patterns)
    dest = f"hf://datasets/{t.repo}" + (f"/{t.subdir}" if t.subdir else "")
    print(f"[{t.name}] uploading {t.folder} ({n} files) -> {dest}", file=sys.stderr)
    api.upload_folder(
        repo_id=t.repo, repo_type="dataset", folder_path=str(t.folder),
        path_in_repo=t.subdir or None, allow_patterns=hub_patterns(t.patterns),
        ignore_patterns=[".DS_Store"], commit_message=f"{t.name}: {n} files",
    )


def pull(api: HfApi, t: Target, workers: int) -> None:
    t.folder.mkdir(parents=True, exist_ok=True)
    if t.subdir:
        # snapshot_download keeps the repo layout, so land it in a sibling then move the subdir in.
        allow = [f"{t.subdir}/{p}" for p in hub_patterns(t.patterns)]
        staging = t.folder.parent / f".{t.folder.name}.hf"
        snapshot_download(repo_id=t.repo, repo_type="dataset", local_dir=str(staging),
                          allow_patterns=allow, max_workers=workers)
        src = staging / t.subdir
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    out = t.folder / f.relative_to(src)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    f.replace(out)
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
    else:
        snapshot_download(repo_id=t.repo, repo_type="dataset", local_dir=str(t.folder),
                          allow_patterns=hub_patterns(t.patterns), max_workers=workers)
    print(f"[{t.name}] {count(t.folder, t.patterns)} files in {t.folder}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["push", "pull", "ls"])
    ap.add_argument("targets", nargs="*", choices=[*TARGETS, []], help="default: all")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    names = a.targets or list(TARGETS)
    if a.command == "ls":
        for t in TARGETS.values():
            local = count(t.folder, t.patterns) if t.folder.is_dir() else 0
            print(f"{t.name:12s} {str(t.folder):14s} {local:6d} local  -> {t.repo}/{t.subdir}")
        return 0
    api = HfApi()
    for name in names:
        (push(api, TARGETS[name]) if a.command == "push" else pull(api, TARGETS[name], a.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
