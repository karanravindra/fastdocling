"""Download public out-of-domain PDFs for evaluating the draft model beyond ML papers.

Usage:  uv run python scripts/fetch_ood.py [data/ood/docs]
Then:   uv run fastdocling-prep render data/ood/docs data/ood/images

Each entry is (category, name, url). Categories follow Docling's typical workloads:
forms, tax/legal text, regulation, standards & RFCs, finance/policy minutes, lecture notes,
government resolutions, and a non-ML scientific paper.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

SOURCES = [
    ("forms",       "irs_f1040",              "https://www.irs.gov/pub/irs-pdf/f1040.pdf"),
    ("tax_guide",   "irs_i1040_instructions", "https://www.irs.gov/pub/irs-pdf/i1040gi.pdf"),
    ("legal",       "scotus_opinion",         "https://www.supremecourt.gov/opinions/23pdf/22-451_7m58.pdf"),
    ("legal",       "uk_legislation",         "https://www.legislation.gov.uk/ukpga/2018/12/pdfs/ukpga_20180012_en.pdf"),
    ("standards",   "rfc9114_http3",          "https://www.rfc-editor.org/rfc/rfc9114.pdf"),
    ("standards",   "nist_sp800_63",          "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63-3.pdf"),
    ("finance",     "fed_fomc_minutes",       "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20240612.pdf"),
    ("lecture",     "slides_cs229_notes",     "https://cs229.stanford.edu/notes2022fall/main_notes.pdf"),
    ("government",  "un_resolution",          "https://documents.un.org/doc/undoc/gen/n15/291/89/pdf/n1529189.pdf"),
    ("science",     "arxiv_2401_00001",       "https://arxiv.org/pdf/2401.00001"),
]


def main(root: Path) -> int:
    failed = 0
    for category, name, url in SOURCES:
        dest = root / category / f"{name}.pdf"
        if dest.exists():
            print(f"skip {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("not a PDF")
            dest.write_bytes(data)
            print(f"ok   {dest} ({len(data)/1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ood/docs")))
