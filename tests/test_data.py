"""The packed corpus has to agree with the streaming loader it replaces.

``packed_batches`` is a second implementation of the windowing ``iterate_windows`` does, reading
a pre-built array instead of re-deriving each batch.  Two implementations of the same alignment
is exactly the situation that drifts silently: a window shifted by one position still trains, it
just trains the drafter to predict the wrong token.  These tests pin the agreement.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fastdocling.data import (EAGLE3_TAPS, FIRST_OFFSET, TraceInfo, ensure_packed,
                              iterate_batches, iterate_windows, load_packed, load_trace,
                              pack_traces, packed_batches, page_key, prefetch_to_device,
                              scan_traces)

CTX, HORIZON, FIRST_OFFSET_1 = 8, 2, 1
WINDOW = CTX + HORIZON + FIRST_OFFSET_1 - 1


def write_trace(path: Path, n_tokens: int, prompt_len: int, seed: int) -> None:
    """A synthetic trace in the on-disk format, with EAGLE-3 taps."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    state_offset = prompt_len - 1
    m = n_tokens - state_offset
    tensors = {
        "token_ids": rng.integers(0, 1000, n_tokens).astype(np.int32),
        "prompt_len": np.array([prompt_len], dtype=np.int32),
        "state_offset": np.array([state_offset], dtype=np.int32),
        "last_hidden_state": rng.standard_normal((m, 576)).astype(np.float16),
    }
    for k in EAGLE3_TAPS:
        tensors[k] = rng.standard_normal((m, 576)).astype(np.float16)
    save_file(tensors, str(path))


class PackedCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.traces = root / "traces"
        cls.traces.mkdir()
        # Deliberately ragged: pages that yield 1, 2 and 5 windows, plus one too short to use.
        for i, (n, plen) in enumerate([(60, 10), (40, 10), (120, 10), (14, 10)]):
            write_trace(cls.traces / f"page__{i:04d}.safetensors", n, plen, seed=i)
        cls.infos = scan_traces(cls.traces, min_completion=CTX + 2)
        cls.pack_dir = root / "pack"
        cls.pack = pack_traces(cls.infos, cls.pack_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_pack_holds_exactly_what_load_trace_returns(self):
        for p, info in enumerate(self.infos):
            x, _, tok = load_trace(info, "eagle3")
            a, b = int(self.pack.starts[p]), int(self.pack.starts[p + 1])
            self.assertEqual(b - a, x.shape[0], info.path.name)
            np.testing.assert_array_equal(self.pack.taps[a:b], x.half().numpy())
            np.testing.assert_array_equal(self.pack.tokens[a:b], tok.numpy().astype(np.int32))

    def test_window_count_matches_the_streaming_loader(self):
        packed = sum(1 for _ in packed_batches(self.pack, self.infos, CTX, 1, horizon=HORIZON,
                                               seed=0, first_offset=FIRST_OFFSET_1))
        stream = sum(1 for _ in iterate_batches(self.infos, CTX, 1, horizon=HORIZON,
                                                features="eagle3", seed=0,
                                                first_offset=FIRST_OFFSET_1))
        self.assertEqual(packed, stream)

    def test_labels_are_the_tokens_that_follow_the_window(self):
        """tokens[:, :, j] must be offset first_offset + j from the input position."""
        x, tok = next(packed_batches(self.pack, self.infos, CTX, 4, horizon=HORIZON, seed=0,
                                     first_offset=FIRST_OFFSET_1))
        self.assertEqual(tuple(x.shape), (4, CTX, 576 * len(EAGLE3_TAPS)))
        self.assertEqual(tuple(tok.shape), (4, CTX, HORIZON))
        self.assertEqual(x.dtype, torch.float16)
        for j in range(x.shape[0]):
            start = self._locate(x[j])
            for h in range(HORIZON):
                want = self.pack.tokens[start + FIRST_OFFSET_1 + h :
                                        start + FIRST_OFFSET_1 + h + CTX]
                np.testing.assert_array_equal(tok[j, :, h].numpy().astype(np.int32), want)

    def test_windows_never_cross_a_page_boundary(self):
        """A window spanning two pages would train on a discontinuity that cannot occur at serve
        time, where the drafter's KV cache is reset per request."""
        for x, _ in packed_batches(self.pack, self.infos, CTX, 2, horizon=HORIZON, seed=1,
                                   first_offset=FIRST_OFFSET_1, drop_last=False):
            for j in range(x.shape[0]):
                start = self._locate(x[j])
                page = int(np.searchsorted(self.pack.starts, start, side="right") - 1)
                self.assertGreaterEqual(start, int(self.pack.starts[page]))
                self.assertLessEqual(start + WINDOW, int(self.pack.starts[page + 1]))

    def _locate(self, window: torch.Tensor) -> int:
        """Row in the pack where ``window`` starts (the windows are distinct by construction)."""
        first = window[0].numpy()
        for c in np.nonzero(self.pack.taps[:, 0] == first[0])[0]:
            if np.array_equal(self.pack.taps[c : c + CTX], window.numpy()):
                return int(c)
        self.fail("window is not a slice of the pack")

    def test_alignment_holds_at_both_first_offsets(self):
        """The default FIRST_OFFSET=2 path is otherwise untested; the eagle3 loop uses 1.

        Not compared window-for-window against ``iterate_windows``: the two draw their per-page
        phase from different generators (``random.Random`` after shuffling the page order, versus
        ``np.random.default_rng`` per page), so the same seed picks different -- equally legal --
        phases.  What has to agree is the count, and that every window is a contiguous in-page
        slice whose labels sit at the right offsets.
        """
        for first_offset in (1, FIRST_OFFSET):
            with self.subTest(first_offset=first_offset):
                window = CTX + HORIZON + first_offset - 1
                kw = dict(horizon=HORIZON, seed=5, first_offset=first_offset)
                packed = list(packed_batches(self.pack, self.infos, CTX, 1, drop_last=False, **kw))
                stream = list(iterate_windows(self.infos, CTX, features="eagle3",
                                              want_states=False, **kw))
                self.assertEqual(len(packed), len(stream))
                for x, tok in packed:
                    start = self._locate(x[0])
                    page = int(np.searchsorted(self.pack.starts, start, side="right") - 1)
                    self.assertLessEqual(start + window, int(self.pack.starts[page + 1]))
                    for h in range(HORIZON):
                        lo = start + first_offset + h
                        np.testing.assert_array_equal(
                            tok[0, :, h].numpy().astype(np.int32),
                            self.pack.tokens[lo : lo + CTX])

    def test_seed_changes_the_window_phase(self):
        a = next(packed_batches(self.pack, self.infos, CTX, 2, horizon=HORIZON, seed=0,
                                first_offset=FIRST_OFFSET_1))[0]
        b = next(packed_batches(self.pack, self.infos, CTX, 2, horizon=HORIZON, seed=7,
                                first_offset=FIRST_OFFSET_1))[0]
        self.assertFalse(torch.equal(a, b))

    def test_ensure_packed_reuses_and_rebuilds_on_a_changed_page_set(self):
        same = ensure_packed(self.infos, self.pack_dir)
        self.assertEqual(same.key, self.pack.key)
        fewer = ensure_packed(self.infos[:2], self.pack_dir)
        self.assertNotEqual(fewer.key, self.pack.key)
        # Rebuilding renames a fresh file into place rather than truncating the mapped one, so an
        # already-open PackedCorpus keeps working instead of faulting with SIGBUS.
        self.assertEqual(self.pack.taps.shape[0], int(self.pack.starts[-1]))
        float(self.pack.taps[-1, 0])
        ensure_packed(self.infos, self.pack_dir)   # restore for the other tests

    def test_rows_for_rejects_pages_outside_the_pack(self):
        stranger = TraceInfo(Path("nowhere/page__9999.safetensors"), 60, 10, 9, True)
        with self.assertRaises(KeyError):
            self.pack.rows_for([stranger])

    def test_pages_are_keyed_by_source_root_not_bare_name(self):
        """scan_traces takes several roots, and trace_name is relative to each one, so the same
        page in two corpora shares a file name.  Keyed on the bare name they collide onto one row."""
        a = TraceInfo(self.traces / "page__0000.safetensors", 60, 10, 9, True, source="traces")
        b = TraceInfo(Path("elsewhere/page__0000.safetensors"), 60, 10, 9, True, source="prefill")
        self.assertNotEqual(page_key(a), page_key(b))
        with self.assertRaises(KeyError):
            self.pack.rows_for([b])

    def test_a_colliding_page_name_is_refused_before_anything_is_written(self):
        """corpus_key rejects a name that appears in two roots.  Computing it at the *end* of the
        build meant finding that out after writing the whole pack -- 10.3 GB on the real corpus."""
        same = [
            TraceInfo(self.traces / "page__0000.safetensors", 60, 10, 9, True, source="a"),
            TraceInfo(self.traces / "page__0000.safetensors", 60, 10, 9, True, source="b"),
        ]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pack"
            with self.assertRaisesRegex(ValueError, "more than one trace root"):
                pack_traces(same, out)
            self.assertFalse(out.exists(), "the output directory should not even be created")


class PackGuardTest(unittest.TestCase):
    """What ensure_packed is allowed to hand back, and what pack_traces must refuse."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.traces = self.root / "traces"
        self.traces.mkdir()
        for i in range(3):
            write_trace(self.traces / f"page__{i:04d}.safetensors", 60, 10, seed=200 + i)
        self.infos = scan_traces(self.traces, min_completion=CTX + 2)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_pack_of_different_features_is_not_reused(self):
        """`features` sets the width of every row, and corpus_key cannot see it."""
        wide = ensure_packed(self.infos, self.root / "p", features="eagle3")
        self.assertEqual(wide.dim, 576 * len(EAGLE3_TAPS))
        narrow = ensure_packed(self.infos, self.root / "p", features="last")
        self.assertEqual(narrow.dim, 576)

    def test_a_re_extraction_under_the_same_names_is_not_reused(self):
        """Same page set, new tap values: keyed on names alone this served the old extraction."""
        before = ensure_packed(self.infos, self.root / "p")
        first_row = np.array(before.taps[0])
        write_trace(self.traces / "page__0000.safetensors", 60, 10, seed=999)
        rescanned = scan_traces(self.traces, min_completion=CTX + 2)
        after = ensure_packed(rescanned, self.root / "p")
        self.assertFalse(np.array_equal(np.array(after.taps[0]), first_row))

    def test_a_malformed_trace_is_named_rather_than_clamped(self):
        """Clamping left off < total, which load_packed rejects -- forever, since every rebuild
        would clamp again.  A bad page has to be named instead."""
        bad = self.infos[0]
        broken = TraceInfo(bad.path, bad.n_tokens + 5, bad.prompt_len, bad.state_offset, True,
                           source=bad.source)
        out = self.root / "broken"
        with self.assertRaisesRegex(ValueError, "malformed"):
            pack_traces([broken], out)
        with self.assertRaises(FileNotFoundError):
            load_packed(out)

    def test_a_failed_build_leaves_no_staging_file_behind(self):
        """The staging taps.npy is preallocated full size -- 10.3 GB on the real corpus."""
        no_taps = [TraceInfo(i.path, i.n_tokens, i.prompt_len, i.state_offset, False) for i in self.infos]
        out = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "no layer taps"):
            pack_traces(no_taps, out)
        self.assertFalse((out / ".building").exists())

    def test_an_interrupted_build_does_not_leave_a_loadable_pack(self):
        good = ensure_packed(self.infos, self.root / "p")
        no_taps = [TraceInfo(i.path, i.n_tokens, i.prompt_len, i.state_offset, False) for i in self.infos]
        with self.assertRaises(ValueError):
            pack_traces(no_taps, self.root / "p")
        with self.assertRaises(FileNotFoundError):
            load_packed(self.root / "p")
        # the pack that was already open keeps reading: the rename left its inode alone
        float(good.taps[-1, 0])


class StreamingLoaderTest(unittest.TestCase):
    """``want_states=False`` may change what is built, never what is trained on."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.traces = Path(cls._tmp.name)
        for i, (n, plen) in enumerate([(60, 10), (80, 10)]):
            write_trace(cls.traces / f"page__{i:04d}.safetensors", n, plen, seed=100 + i)
        cls.infos = scan_traces(cls.traces, min_completion=CTX + 2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_want_states_false_drops_only_the_states(self):
        kw = dict(horizon=HORIZON, features="eagle3", seed=3, first_offset=FIRST_OFFSET_1)
        x, states, tok = next(iterate_batches(self.infos, CTX, 2, **kw))
        x2, none, tok2 = next(iterate_batches(self.infos, CTX, 2, want_states=False, **kw))
        self.assertIsNotNone(states)
        self.assertIsNone(none)
        self.assertTrue(torch.equal(x, x2))
        self.assertTrue(torch.equal(tok, tok2))


class PrefetchTest(unittest.TestCase):
    def test_yields_every_batch_in_order(self):
        src = [(torch.full((2, 2), i, dtype=torch.float32),) for i in range(9)]
        got = [b[0][0, 0].item() for b in prefetch_to_device(iter(src), "cpu", depth=2)]
        self.assertEqual(got, list(range(9)))

    def test_producer_exception_reaches_the_consumer(self):
        def boom():
            yield (torch.zeros(2, 2),)
            raise RuntimeError("producer exploded")

        with self.assertRaisesRegex(RuntimeError, "producer exploded"):
            list(prefetch_to_device(boom(), "cpu"))

    def test_early_exit_does_not_leave_the_worker_running(self):
        """Without the drain on close, a worker blocked in ``put`` outlives the loop."""
        import threading

        before = threading.active_count()
        it = prefetch_to_device(iter([(torch.zeros(2, 2),) for _ in range(200)]), "cpu", depth=2)
        next(it)
        it.close()
        for _ in range(100):
            if threading.active_count() == before:
                break
            threading.Event().wait(0.02)
        self.assertEqual(threading.active_count(), before)


if __name__ == "__main__":
    unittest.main()
