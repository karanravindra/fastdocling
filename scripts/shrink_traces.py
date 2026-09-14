"""Losslessly shrink existing traces in place (drop prompt-position states and, by default, layer taps).

    uv run python scripts/shrink_traces.py data/traces [--keep-taps]

Files written in the last 60 s are skipped (an extractor may still be writing them); each file is
rewritten atomically (tmp + rename).  Already-shrunk files are left alone.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from tqdm import tqdm

from fastdocling.extract import finalize_trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, nargs="?", default=Path("data/traces"))
    ap.add_argument("--keep-taps", action="store_true")
    a = ap.parse_args()
    files = sorted(a.root.glob("*.safetensors"))
    before = after = 0
    skipped = 0
    for f in tqdm(files, unit="file"):
        if time.time() - f.stat().st_mtime < 60:
            skipped += 1
            continue
        t = mx.load(str(f))
        has_taps = any(k.startswith("layer_") for k in t)
        if "state_offset" in t and (a.keep_taps or not has_taps):
            continue
        size = f.stat().st_size
        pl = int(t["prompt_len"].item())
        if "state_offset" in t:                        # already offset; only taps to drop
            off = int(t["state_offset"].item())
            out = {k: v for k, v in t.items() if a.keep_taps or not k.startswith("layer_")}
            assert off == pl - 1
        else:
            out = finalize_trace(t, pl, a.keep_taps)
        mx.eval(*out.values())
        tmp = f.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(tmp), out)
        tmp.replace(f)
        before += size
        after += f.stat().st_size
    print(f"rewrote {len(files) - skipped} files: {before/1e9:.2f} GB -> {after/1e9:.2f} GB; skipped {skipped} recent", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
