"""Standardize PDFs and render page images with Ghostscript.

Usage:
    fastdocling-prep compress data/docs            # downsample images to 150 dpi, in place
    fastdocling-prep render   data/docs data/images  # PNGs at docling scale 2.0 (144 dpi)
    fastdocling-prep all      data/docs data/images  # both steps
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DOCLING_SCALE = 2.0  # docling renders pages at 72 dpi * scale
DOCLING_DPI = int(72 * DOCLING_SCALE)


def find_pdfs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf")


def _gs(args: list[str]) -> bool:
    proc = subprocess.run(
        ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def compress_pdf(pdf: Path, dpi: int) -> tuple[Path, bool]:
    tmp = pdf.with_suffix(".gs_tmp.pdf")
    ok = _gs(
        [
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            "-dDownsampleColorImages=true",
            "-dDownsampleGrayImages=true",
            "-dDownsampleMonoImages=true",
            f"-dColorImageResolution={dpi}",
            f"-dGrayImageResolution={dpi}",
            f"-dMonoImageResolution={dpi}",
            "-dColorImageDownsampleThreshold=1.0",
            "-dGrayImageDownsampleThreshold=1.0",
            "-dMonoImageDownsampleThreshold=1.0",
            f"-sOutputFile={tmp}",
            str(pdf),
        ]
    )
    if ok and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(pdf)
        return pdf, True
    tmp.unlink(missing_ok=True)
    return pdf, False


def render_pdf(pdf: Path, docs_root: Path, out_root: Path, dpi: int) -> tuple[Path, bool]:
    rel = pdf.relative_to(docs_root) if pdf.is_relative_to(docs_root) else Path(pdf.name)
    out_dir = out_root / rel.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = _gs(
        [
            "-sDEVICE=png16m",
            f"-r{dpi}",
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            f"-sOutputFile={out_dir / '%04d.png'}",
            str(pdf),
        ]
    )
    return pdf, ok


def _run(label: str, jobs, fn, workers: int) -> int:
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            pdf, ok = fut.result()
            status = "ok  " if ok else "FAIL"
            failed += not ok
            print(f"[{label}] {i}/{len(futures)} {status} {pdf.name}", file=sys.stderr)
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["compress", "render", "all"])
    ap.add_argument("docs", type=Path, help="PDF file or directory (searched recursively)")
    ap.add_argument("images", type=Path, nargs="?", help="output root for PNGs (render/all)")
    ap.add_argument("--compress-dpi", type=int, default=150, help="image resolution for compress (default 150)")
    ap.add_argument("--render-dpi", type=int, default=DOCLING_DPI, help=f"render resolution (default {DOCLING_DPI} = docling scale {DOCLING_SCALE})")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="parallel Ghostscript processes")
    args = ap.parse_args(argv)

    if shutil.which("gs") is None:
        ap.error("ghostscript (gs) not found on PATH")
    if args.command in ("render", "all") and args.images is None:
        ap.error("images output directory is required for render/all")

    pdfs = find_pdfs(args.docs)
    if not pdfs:
        ap.error(f"no PDFs found under {args.docs}")
    docs_root = args.docs if args.docs.is_dir() else args.docs.parent

    failed = 0
    if args.command in ("compress", "all"):
        failed += _run("compress", [(p, args.compress_dpi) for p in pdfs], compress_pdf, args.jobs)
    if args.command in ("render", "all"):
        failed += _run("render", [(p, docs_root, args.images, args.render_dpi) for p in pdfs], render_pdf, args.jobs)

    print(f"{len(pdfs)} PDFs processed, {failed} failures", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
