from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastdocling.decode import detect
from fastdocling.decode.base import SpecResult, cached_generate
from fastdocling.decode.vllm_decoder import UNSUPPORTED, VLLMDecoder


class _Decoder:
    name = "fake"
    supports_draft = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, image, *, use_draft: bool, **kwargs) -> SpecResult:
        self.calls += 1
        return SpecResult([1], 0.1, 0.2, 1, 0, backend=self.name)


class DecodeContractTest(unittest.TestCase):
    def test_detect_falls_back_when_mlx_is_not_installed(self) -> None:
        with patch("fastdocling.decode.platform.system", return_value="Darwin"), \
             patch("fastdocling.decode.platform.machine", return_value="arm64"), \
             patch("fastdocling.decode.available", return_value=["transformers"]):
            self.assertEqual(detect(), "transformers")

    def test_detect_prefers_installed_mlx_on_apple_silicon(self) -> None:
        with patch("fastdocling.decode.platform.system", return_value="Darwin"), \
             patch("fastdocling.decode.platform.machine", return_value="arm64"), \
             patch("fastdocling.decode.available", return_value=["mlx", "transformers"]):
            self.assertEqual(detect(), "mlx")

    def test_cached_generate_reuses_a_greedy_result(self) -> None:
        decoder = _Decoder()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.png"
            image.write_bytes(b"image")
            with patch("transformers.image_utils.load_image", return_value=object()):
                first, first_hit = cached_generate(
                    decoder, image, use_draft=False, cache_dir=directory, key={"kind": "baseline"}
                )
                second, second_hit = cached_generate(
                    decoder, image, use_draft=False, cache_dir=directory, key={"kind": "baseline"}
                )
        self.assertEqual(decoder.calls, 1)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(first.tokens, second.tokens)

    def test_vllm_rejects_draft_attachment_without_loading_vllm(self) -> None:
        decoder = object.__new__(VLLMDecoder)
        # The reason changed: rollback is vLLM's job for drafters it hosts; what actually
        # blocks the latent draft is that no proposer interface takes a window of states.
        with self.assertRaisesRegex(NotImplementedError, "cannot be attached here"):
            decoder.attach_draft(None)
        # The message must still point somewhere useful: a backend that runs the latent
        # draft, and the way to measure vLLM's own drafter instead.
        self.assertIn("transformers", UNSUPPORTED)
        self.assertIn("speculative_config", UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
