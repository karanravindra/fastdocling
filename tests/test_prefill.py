"""Borrowed-label extraction: manifest naming, page splitting, and corpus separation.

The failure these guard against is silent.  A manifest names its traces, and two pages whose
image files share a stem (``a/0001.png``, ``b/0001.png`` -- the shape every rendered corpus has)
used to land on one trace file, so the second overwrote the first and the corpus quietly shrank.
The same goes for splitting a multi-page dataset row: pairing N labels with M images when N != M
puts every label after the mismatch on the wrong picture, and nothing downstream can tell.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from fastdocling.data import TraceInfo, corpus_key, scan_traces, source_counts
from fastdocling.extract import _read_manifest

_spec = importlib.util.spec_from_file_location(
    "fetch_doctags", Path(__file__).resolve().parent.parent / "scripts" / "fetch_doctags.py")
fetch_doctags = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_doctags)


def write_manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def write_trace(path: Path, n_tokens: int = 8, prompt_len: int = 3) -> None:
    """A minimal trace with the keys ``scan_traces`` reads out of the header."""
    states = np.zeros((n_tokens - prompt_len + 1, 4), dtype=np.float16)
    save_file({
        "token_ids": np.arange(n_tokens, dtype=np.int32),
        "prompt_len": np.array([prompt_len], dtype=np.int32),
        "state_offset": np.array([prompt_len - 1], dtype=np.int32),
        "last_hidden_state": states,
    }, str(path))


class ManifestNamingTest(unittest.TestCase):
    def test_same_stem_in_different_directories_gets_distinct_names(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for sub in ("a", "b"):
                (root / sub).mkdir()
                (root / sub / "0001.dt").write_text(f"<doctag>{sub}</doctag>")
            mf = write_manifest(root, [
                {"image": str(root / sub / "0001.png"), "doctags_path": str(root / sub / "0001.dt")}
                for sub in ("a", "b")
            ])
            names = [r["name"] for r in _read_manifest(mf)]
            self.assertEqual(names, ["a__0001", "b__0001"])
            self.assertEqual(len(set(names)), 2, "two pages must not share one trace file")

    def test_explicit_name_is_honoured(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mf = write_manifest(root, [
                {"image": str(root / "x.png"), "doctags": "<doctag/>", "name": "corpus__000__001"}])
            self.assertEqual(_read_manifest(mf)[0]["name"], "corpus__000__001")

    def test_duplicate_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mf = write_manifest(root, [
                {"image": str(root / "a.png"), "doctags": "<doctag/>", "name": "dup"},
                {"image": str(root / "b.png"), "doctags": "<doctag/>", "name": "dup"},
            ])
            with self.assertRaisesRegex(ValueError, "duplicate trace name"):
                _read_manifest(mf)

    def test_labels_are_read_from_disk_when_inline_text_is_absent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "p.dt").write_text("<doctag>from disk</doctag>")
            mf = write_manifest(root, [{"image": str(root / "p.png"), "doctags_path": str(root / "p.dt")}])
            self.assertEqual(_read_manifest(mf)[0]["doctags"], "<doctag>from disk</doctag>")


class PageSplittingTest(unittest.TestCase):
    def row(self, assistant: str, images: list[str]) -> dict:
        return {"texts": [{"user": "Convert this page to docling.", "assistant": assistant}],
                "images": images}

    def test_multi_page_row_splits_and_rewraps_each_page(self):
        pages = fetch_doctags.pages_of(self.row(
            "<doctag><text>a</text><page_break><text>b</text></doctag>", ["I1", "I2"]))
        self.assertEqual(pages, [("I1", "<doctag><text>a</text></doctag>"),
                                 ("I2", "<doctag><text>b</text></doctag>")])

    def test_row_whose_counts_disagree_is_dropped(self):
        self.assertIsNone(fetch_doctags.pages_of(
            self.row("<doctag><text>a</text></doctag>", ["I1", "I2"])))

    def test_row_without_a_conversion_turn_is_dropped(self):
        self.assertIsNone(fetch_doctags.pages_of(
            {"texts": [{"user": "What is this?", "assistant": "a form"}], "images": ["I1"]}))

    def test_single_page_row_is_left_intact(self):
        pages = fetch_doctags.pages_of(self.row("<doctag><text>a</text></doctag>", ["I1"]))
        self.assertEqual(pages, [("I1", "<doctag><text>a</text></doctag>")])


class CorpusSeparationTest(unittest.TestCase):
    def test_multiple_roots_load_together_but_stay_attributable(self):
        with tempfile.TemporaryDirectory() as d:
            clean, borrowed = Path(d) / "traces", Path(d) / "traces_prefill"
            clean.mkdir(); borrowed.mkdir()
            for i in range(3):
                write_trace(clean / f"ml_papers__doc__{i:04d}.safetensors")
            write_trace(borrowed / "doclingmatix__0000000__000.safetensors")

            infos = scan_traces([clean, borrowed])
            self.assertEqual(len(infos), 4)
            self.assertEqual(source_counts(infos), {"traces": 3, "traces_prefill": 1})
            self.assertEqual(len(scan_traces(clean)), 3, "one root must still load alone")

    def test_a_missing_root_is_reported_not_silently_empty(self):
        with tempfile.TemporaryDirectory() as d:
            clean = Path(d) / "traces"
            clean.mkdir()
            write_trace(clean / "ml_papers__doc__0001.safetensors")
            with self.assertRaises(FileNotFoundError):
                scan_traces([clean, Path(d) / "does_not_exist"])

    def test_corpus_key_refuses_to_alias_same_named_pages_from_two_roots(self):
        shared = [TraceInfo(Path("a/page.safetensors"), 8, 3, source="traces"),
                  TraceInfo(Path("b/page.safetensors"), 8, 3, source="traces_prefill")]
        with self.assertRaisesRegex(ValueError, "more than one trace root"):
            corpus_key(shared)

    def test_corpus_key_is_stable_for_a_single_corpus(self):
        infos = [TraceInfo(Path(f"t/p{i}.safetensors"), 8, 3, source="traces") for i in range(3)]
        self.assertEqual(corpus_key(infos), corpus_key(list(reversed(infos))))



class OverlongFilterTest(unittest.TestCase):
    """One page too long for the context window must cost that page, not the run.

    vLLM rejects an over-length request at submission and the exception takes the whole process
    with it -- which is how a single 20,816-token DoclingMatix label killed a 10,727-page
    extraction at page 512.  Borrowed labels carry no length guarantee, so the filter is what
    makes the manifest path survive them.
    """

    class FakeBackend:
        """Stands in for a backend: a whitespace tokenizer and a fixed vision prefix."""

        max_model_len = 100

        class _Tok:
            def encode(self, text, add_special_tokens=True):
                return text.split()

        def __init__(self, prefix=10):
            self.tokenizer = self._Tok()
            self.prefix = prefix
            self.calls = 0

        def _prompt_token_len(self, image):
            self.calls += 1
            return self.prefix

    def rows_with(self, root: Path, lengths: list[int]) -> list[dict]:
        from PIL import Image

        rows = []
        for i, n in enumerate(lengths):
            img = root / f"p{i}.png"
            Image.new("RGB", (32, 32)).save(img)
            rows.append({"image": str(img), "name": f"p{i}", "doctags": " ".join(["t"] * n)})
        return rows

    def test_pages_over_the_window_are_dropped_and_the_rest_kept(self):
        from fastdocling.extract import _drop_overlong

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ex = self.FakeBackend(prefix=10)
            # budget is 100: 10 of prefix leaves 90 label tokens
            rows = self.rows_with(root, [50, 89, 90, 91, 500])
            kept, dropped = _drop_overlong(rows, ex, ex.max_model_len)
            self.assertEqual([r["name"] for r in kept], ["p0", "p1", "p2"])
            self.assertEqual(dropped, 2)

    def test_vision_prefix_is_measured_once_per_distinct_size(self):
        from fastdocling.extract import _drop_overlong

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ex = self.FakeBackend(prefix=10)
            _drop_overlong(self.rows_with(root, [1] * 20), ex, ex.max_model_len)
            self.assertEqual(ex.calls, 1, "same-sized pages must not re-measure the prefix")

    def test_a_backend_that_cannot_measure_is_left_alone(self):
        from fastdocling.extract import _drop_overlong

        class Bare:
            pass

        rows = [{"image": "x.png", "name": "x", "doctags": "t"}]
        self.assertEqual(_drop_overlong(rows, Bare(), 100), (rows, 0))
        self.assertEqual(_drop_overlong(rows, self.FakeBackend(), 0), (rows, 0))

if __name__ == "__main__":
    unittest.main()
