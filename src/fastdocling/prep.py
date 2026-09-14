"""Standardize PDFs and render page images with Ghostscript.

Usage:
    fastdocling-prep compress data/docs            # downsample images to 150 dpi, in place
    fastdocling-prep render   data/docs data/images  # PNGs at docling scale 2.0 (144 dpi); skips rendered docs
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


def compress_pdf(pdf: Path, dpi: int) -> tuple[Path, str]:
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
        return pdf, "ok"
    tmp.unlink(missing_ok=True)
    return pdf, "FAIL"


def render_pdf(pdf: Path, docs_root: Path, out_root: Path, dpi: int, force: bool = False) -> tuple[Path, str]:
    """Render one PDF to ``out_root/<rel>/%04d.png``.

    A document whose output directory already holds PNGs is skipped (re-running ``render`` over a
    corpus then only touches new PDFs).  Pages are rendered into a sibling temp directory that is
    renamed into place on success, so a directory that exists is always a complete render.
    Returns ``(pdf, status)`` with status ``"ok"``, ``"skip"`` or ``"FAIL"``.
    """
    rel = pdf.relative_to(docs_root) if pdf.is_relative_to(docs_root) else Path(pdf.name)
    out_dir = out_root / rel.with_suffix("")
    if not force and out_dir.is_dir() and any(out_dir.glob("*.png")):
        return pdf, "skip"
    tmp = out_dir.with_name(out_dir.name + ".rendering")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    ok = _gs(
        [
            "-sDEVICE=png16m",
            f"-r{dpi}",
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            f"-sOutputFile={tmp / '%04d.png'}",
            str(pdf),
        ]
    )
    ok = ok and any(tmp.glob("*.png"))   # gs can exit 0 without emitting a page; never replace output with nothing
    if ok:
        shutil.rmtree(out_dir, ignore_errors=True)
        tmp.replace(out_dir)
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return pdf, "ok" if ok else "FAIL"


def _run(label: str, jobs, fn, workers: int) -> int:
    failed = skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            pdf, status = fut.result()
            failed += status == "FAIL"
            skipped += status == "skip"
            print(f"[{label}] {i}/{len(futures)} {status:4s} {pdf.name}", file=sys.stderr)
    if skipped:
        print(f"[{label}] {skipped} already done, skipped (use --force to redo)", file=sys.stderr)
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["compress", "render", "all"])
    ap.add_argument("docs", type=Path, help="PDF file or directory (searched recursively)")
    ap.add_argument("images", type=Path, nargs="?", help="output root for PNGs (render/all)")
    ap.add_argument("--compress-dpi", type=int, default=150, help="image resolution for compress (default 150)")
    ap.add_argument("--render-dpi", type=int, default=DOCLING_DPI, help=f"render resolution (default {DOCLING_DPI} = docling scale {DOCLING_SCALE})")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="parallel Ghostscript processes")
    ap.add_argument("--force", action="store_true", help="re-render PDFs whose page images already exist")
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
        failed += _run("render", [(p, docs_root, args.images, args.render_dpi, args.force) for p in pdfs], render_pdf, args.jobs)

    print(f"{len(pdfs)} PDFs processed, {failed} failures", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
