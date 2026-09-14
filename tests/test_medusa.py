"""Medusa draft: export contract, and equivalence with the class vLLM actually runs.

The point of the vLLM half of this file is narrow and load-bearing: training optimises
``MedusaDraft``, but inference runs ``vllm.model_executor.models.medusa.Medusa`` over the exported
checkpoint.  If those two compute different functions, every acceptance number measured here is
about a model that never runs.  So the tests push identical hidden states through both and compare.

They stay on the CPU: vLLM's ``Medusa`` is a plain ``nn.Module``, so it can be built directly from
a stub config with a single-rank gloo process group, without an engine, a target model or a GPU.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fastdocling.medusa import (
    HIDDEN_SIZE,
    MedusaDraft,
    ResidualBlock,
    export_medusa_checkpoint,
)


def _load(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path / "model.safetensors"))


class ExportContractTest(unittest.TestCase):
    """What lands on disk, checked against the names and fields vLLM's loader reads."""

    def test_export_writes_the_key_names_and_config_fields_vllm_reads(self) -> None:
        draft = MedusaDraft(horizon=3, num_layers=2)
        lm_head = torch.randn(512, HIDDEN_SIZE)
        with tempfile.TemporaryDirectory() as d:
            out = export_medusa_checkpoint(draft, lm_head, d, dtype=torch.float32)
            tensors = _load(out)
            config = json.loads((out / "config.json").read_text())

        expected = {f"blocks.{h}.layers.{i}.weight" for h in range(3) for i in range(2)}
        expected.add("lm_heads.0.weight")
        self.assertEqual(set(tensors), expected)
        self.assertEqual(tuple(tensors["lm_heads.0.weight"].shape), (512, HIDDEN_SIZE))
        self.assertEqual(config["architectures"], ["MedusaModel"])
        self.assertEqual(config["model_type"], "medusa")
        self.assertEqual(config["hidden_size"], HIDDEN_SIZE)
        self.assertEqual(config["num_heads"], 3)
        self.assertEqual(config["num_hidden_layers"], 2)
        self.assertEqual(config["vocab_size"], 512)
        self.assertEqual(config["truncated_vocab_size"], 512)
        self.assertTrue(config["original_lm_head"])
        self.assertFalse(config["medusa_fc_bias"])
        self.assertEqual(config["torch_dtype"], "float32")

    def test_medusa_fc_bias_follows_the_model_and_the_bias_tensors_are_written(self) -> None:
        draft = MedusaDraft(horizon=2, num_layers=1, bias=True)
        with tempfile.TemporaryDirectory() as d:
            out = export_medusa_checkpoint(draft, torch.randn(64, HIDDEN_SIZE), d)
            tensors = _load(out)
            config = json.loads((out / "config.json").read_text())
        self.assertTrue(config["medusa_fc_bias"])
        self.assertIn("blocks.0.layers.0.bias", tensors)
        self.assertIn("blocks.1.layers.0.bias", tensors)

    def test_pruned_export_slices_the_head_and_writes_the_map(self) -> None:
        draft = MedusaDraft(horizon=2)
        lm_head = torch.randn(512, HIDDEN_SIZE)
        vocab = np.array([9, 4, 300, 17])
        with tempfile.TemporaryDirectory() as d:
            out = export_medusa_checkpoint(
                draft, lm_head, d, vocab=vocab, dtype=torch.float32, allow_token_map=True)
            tensors = _load(out)
            config = json.loads((out / "config.json").read_text())
        self.assertEqual(tensors["token_map"].dtype, torch.int64)
        self.assertEqual(tensors["token_map"].tolist(), [9, 4, 300, 17])
        self.assertEqual(config["vocab_size"], 512)
        self.assertEqual(config["truncated_vocab_size"], 4)
        # Row order follows the map, not the vocabulary, so vLLM's argmax indexes back through it.
        torch.testing.assert_close(tensors["lm_heads.0.weight"], lm_head[[9, 4, 300, 17]])

    def test_dtype_is_honoured(self) -> None:
        draft = MedusaDraft(horizon=1)
        with tempfile.TemporaryDirectory() as d:
            out = export_medusa_checkpoint(draft, torch.randn(64, HIDDEN_SIZE), d)
            self.assertEqual(_load(out)["lm_heads.0.weight"].dtype, torch.float16)


class ExportGuardTest(unittest.TestCase):
    """The rejections.  Each one stands for a way vLLM fails quietly rather than loudly."""

    def setUp(self) -> None:
        self.draft = MedusaDraft(horizon=2)
        self.lm_head = torch.randn(512, HIDDEN_SIZE)

    def _export(self, **kwargs) -> set[str]:
        with tempfile.TemporaryDirectory() as d:
            return set(_load(export_medusa_checkpoint(self.draft, self.lm_head, d, **kwargs)))

    def test_allow_token_map_gate_refuses_a_map_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow_token_map=True"):
            self._export(vocab=[1, 2, 3])

    def test_allow_token_map_gate_is_only_consulted_for_a_map(self) -> None:
        # No vocab, no flag, no complaint -- and nothing that would need the shim on disk.
        self.assertNotIn("token_map", self._export())

    def test_allow_token_map_permits_the_map(self) -> None:
        self.assertIn("token_map", self._export(vocab=[1, 2, 3], allow_token_map=True))

    def test_a_full_length_vocab_is_refused(self) -> None:
        # vLLM installs a token_map only when truncated_vocab_size < vocab_size.  A permutation of
        # the whole vocabulary would load with the map dropped and emit pruned indices as ids.
        with self.assertRaisesRegex(ValueError, "truncated_vocab_size < vocab_size"):
            self._export(vocab=np.arange(512), allow_token_map=True)

    def test_out_of_range_ids_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the lm_head"):
            self._export(vocab=[1, 512], allow_token_map=True)

    def test_duplicate_ids_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._export(vocab=[1, 2, 2], allow_token_map=True)

    def test_a_transposed_lm_head_is_refused(self) -> None:
        self.lm_head = self.lm_head.T.contiguous()
        with self.assertRaisesRegex(ValueError, "not its transpose"):
            self._export()

    def test_in_dim_must_be_the_targets_hidden_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "FEATURES='last'"):
            MedusaDraft(horizon=2, in_dim=2 * HIDDEN_SIZE)


class ResidualBlockMathsTest(unittest.TestCase):
    """``x = x + SiLU(layer(x))`` per layer, no normalisation -- vLLM's block, exactly."""

    def test_block_is_the_residual_silu_recurrence(self) -> None:
        torch.manual_seed(0)
        block = ResidualBlock(HIDDEN_SIZE, num_layers=3).eval()
        x = torch.randn(4, HIDDEN_SIZE)
        expected = x
        for layer in block.layers:
            expected = expected + torch.nn.functional.silu(layer(expected))
        torch.testing.assert_close(block(x), expected)

    def test_block_applies_no_normalisation(self) -> None:
        # A norm would make the block invariant to input scale; the residual recurrence is not.
        block = ResidualBlock(HIDDEN_SIZE, num_layers=1).eval()
        x = torch.randn(4, HIDDEN_SIZE)
        self.assertFalse(torch.allclose(block(10 * x), 10 * block(x), atol=1e-3))
        self.assertEqual([type(m) for m in block.modules() if "Norm" in type(m).__name__], [])

    def test_forward_stacks_one_head_per_horizon_step(self) -> None:
        draft = MedusaDraft(horizon=3).eval()
        x = torch.randn(2, 5, HIDDEN_SIZE)
        out = draft(x)
        self.assertEqual(tuple(out.shape), (2, 5, 3, HIDDEN_SIZE))
        for h, block in enumerate(draft.blocks):
            torch.testing.assert_close(out[..., h, :], block(x))

    def test_positions_are_independent(self) -> None:
        # No attention, no positional term: this is what lets vLLM call it one state at a time.
        draft = MedusaDraft(horizon=2).eval()
        x = torch.randn(1, 4, HIDDEN_SIZE)
        torch.testing.assert_close(draft(x)[0, 2], draft(x[:, 2:3])[0, 0])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _vllm_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(_vllm_available(), "vLLM is not installed")
class VLLMEquivalenceTest(unittest.TestCase):
    """The claim that matters: vLLM's Medusa computes our ``MedusaDraft``, exactly.

    Built on the CPU from a stub ``vllm_config`` -- ``Medusa.__init__`` reads nothing from it but
    ``speculative_config.draft_model_config.hf_config`` -- so no engine and no GPU are involved.
    """

    ctx = None

    @classmethod
    def setUpClass(cls) -> None:
        # vLLM picks its platform at import; keep it off the shared GPU unless told otherwise.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        try:
            import torch.distributed as dist
            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.distributed.parallel_state import (
                init_distributed_environment,
                initialize_model_parallel,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"vLLM is not importable here: {exc}") from exc

        cls.ctx = set_current_vllm_config(VllmConfig())
        cls.ctx.__enter__()
        try:
            if not dist.is_initialized():
                init_distributed_environment(
                    world_size=1, rank=0, local_rank=0, backend="gloo",
                    distributed_init_method=f"tcp://127.0.0.1:{_free_port()}")
                initialize_model_parallel(1, 1)
        except Exception as exc:  # pragma: no cover - environment-dependent
            cls.ctx.__exit__(None, None, None)
            raise unittest.SkipTest(f"no single-rank process group available: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.ctx is not None:
            cls.ctx.__exit__(None, None, None)

    @staticmethod
    def _build(out: Path, shim: bool = False):
        """Instantiate vLLM's Medusa (or our subclass) over an exported directory."""
        from vllm.transformers_utils.configs.medusa import MedusaConfig
        from types import SimpleNamespace

        if shim:
            from fastdocling.backends._vllm_medusa import MedusaTokenMap as cls_
        else:
            from vllm.model_executor.models.medusa import Medusa as cls_

        hf_config = MedusaConfig.from_pretrained(str(out))
        stub = SimpleNamespace(speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)))
        model = cls_(vllm_config=stub, prefix="")
        loaded = model.load_weights(list(_load(out).items()))
        return model, hf_config, loaded

    def _roundtrip(self, draft, lm_head, *, vocab=None, dtype=torch.float32, shim=False):
        draft.eval()
        with tempfile.TemporaryDirectory() as d:
            out = export_medusa_checkpoint(
                draft, lm_head, d, vocab=vocab, dtype=dtype,
                allow_token_map=vocab is not None)
            return self._build(out, shim=shim)

    # -- hidden states -------------------------------------------------------------------

    def test_hidden_states_match_exactly_in_float32(self) -> None:
        torch.manual_seed(0)
        draft = MedusaDraft(horizon=3, num_layers=2)
        model, _, _ = self._roundtrip(draft, torch.randn(512, HIDDEN_SIZE))
        hidden = torch.randn(7, HIDDEN_SIZE)
        ours = draft(hidden.unsqueeze(0))[0]                 # [T, horizon, 576]
        theirs = torch.stack(model(hidden), dim=1)           # vLLM: list of [T, 576] per head
        self.assertEqual(ours.shape, theirs.shape)
        self.assertEqual((ours - theirs).abs().max().item(), 0.0)

    def test_hidden_states_match_with_bias_heads(self) -> None:
        torch.manual_seed(1)
        draft = MedusaDraft(horizon=2, num_layers=2, bias=True)
        model, hf_config, _ = self._roundtrip(draft, torch.randn(512, HIDDEN_SIZE))
        # The flag has to survive the config round trip, or vLLM builds bias-free layers.
        self.assertTrue(getattr(hf_config, "medusa_fc_bias"))
        self.assertTrue(getattr(hf_config, "original_lm_head"))
        hidden = torch.randn(7, HIDDEN_SIZE)
        ours = draft(hidden.unsqueeze(0))[0]
        theirs = torch.stack(model(hidden), dim=1)
        self.assertEqual((ours - theirs).abs().max().item(), 0.0)

    def test_float16_export_stays_within_rounding(self) -> None:
        torch.manual_seed(2)
        draft = MedusaDraft(horizon=3)
        model, _, _ = self._roundtrip(draft, torch.randn(512, HIDDEN_SIZE), dtype=torch.float16)
        hidden = torch.randn(16, HIDDEN_SIZE)
        ours = draft(hidden.unsqueeze(0))[0]
        theirs = torch.stack(model(hidden), dim=1)
        self.assertLess((ours - theirs).abs().max().item(), 5e-3)

    def test_one_shared_lm_head_backs_every_medusa_head(self) -> None:
        model, _, _ = self._roundtrip(MedusaDraft(horizon=3), torch.randn(512, HIDDEN_SIZE))
        self.assertEqual(len(model.lm_heads), 3)
        self.assertEqual({id(h) for h in model.lm_heads}, {id(model.lm_head)})

    # -- logits and the token map --------------------------------------------------------

    def test_full_vocabulary_logits_match(self) -> None:
        torch.manual_seed(3)
        draft = MedusaDraft(horizon=2)
        lm_head = torch.randn(512, HIDDEN_SIZE)
        model, _, _ = self._roundtrip(draft, lm_head)
        hidden = torch.randn(8, HIDDEN_SIZE)
        ours = draft(hidden.unsqueeze(0))[0]
        theirs = model(hidden)
        logits = torch.stack(model.compute_logits(theirs), dim=1)
        reference = torch.stack([ours[:, h] @ lm_head.T for h in range(2)], dim=1)
        self.assertEqual(tuple(logits.shape), (8, 2, 512))
        torch.testing.assert_close(logits, reference, atol=1e-4, rtol=0)

    def test_token_map_proposals_are_real_vocabulary_ids(self) -> None:
        """The trap: a dropped map would make vLLM emit indices into the pruned head."""
        torch.manual_seed(4)
        draft = MedusaDraft(horizon=3)
        lm_head = torch.randn(4096, HIDDEN_SIZE)
        keep = np.sort(np.random.default_rng(0).choice(4096, size=256, replace=False))
        model, _, loaded = self._roundtrip(draft, lm_head, vocab=keep, shim=True)
        ids = torch.as_tensor(keep, dtype=torch.long)

        hidden = torch.randn(32, HIDDEN_SIZE)
        ours = draft(hidden.unsqueeze(0))[0]
        logits = model.compute_logits(model(hidden))
        # vLLM's MedusaProposer takes exactly this argmax and returns it as a draft token id.
        proposed = torch.stack([lg.argmax(-1) for lg in logits], dim=1)
        expected = ids[torch.stack(
            [(ours[:, h] @ lm_head[ids].T).argmax(-1) for h in range(3)], dim=1)]
        pruned_index = torch.stack(
            [(ours[:, h] @ lm_head[ids].T).argmax(-1) for h in range(3)], dim=1)

        self.assertEqual(tuple(logits[0].shape), (32, 4096), "logits span the full vocabulary")
        torch.testing.assert_close(proposed, expected)
        self.assertTrue(bool(torch.isin(proposed, ids).all()), "every id is a kept vocabulary id")
        self.assertGreater(int(proposed.max()), 256, "ids reach past the pruned head's length")
        # The bug this guards against would have produced `pruned_index` instead.
        self.assertFalse(torch.equal(proposed, pruned_index))
        # Tokens outside the map must be unreachable, not merely unlikely.
        outside = torch.ones(4096, dtype=torch.bool)
        outside[ids] = False
        self.assertTrue(bool(torch.isneginf(logits[0][:, outside]).all()))
        self.assertIn("token_map", loaded)

    # -- the loader bug the shim exists for ----------------------------------------------

    def test_stock_medusa_omits_token_map_from_its_loaded_set(self) -> None:
        """Reproduces the vLLM 0.29 bug: the parameter is registered but never reported."""
        model, _, loaded = self._roundtrip(
            MedusaDraft(horizon=2), torch.randn(4096, HIDDEN_SIZE),
            vocab=np.arange(0, 4096, 4), shim=False)
        registered = {name for name, _ in model.named_parameters()}
        self.assertIn("token_map", registered)
        self.assertNotIn("token_map", loaded)
        # This difference is precisely what DefaultModelLoader.track_weights_loading raises on.
        self.assertEqual(registered - loaded, {"token_map"})

    def test_shim_reports_token_map_so_the_loader_accepts_the_checkpoint(self) -> None:
        model, _, loaded = self._roundtrip(
            MedusaDraft(horizon=2), torch.randn(4096, HIDDEN_SIZE),
            vocab=np.arange(0, 4096, 4), shim=True)
        registered = {name for name, _ in model.named_parameters()}
        self.assertEqual(registered - loaded, set())
        self.assertEqual(model.token_map.device, model.lm_heads[0].weight.device)

    def test_shim_is_a_noop_without_a_token_map(self) -> None:
        model, _, loaded = self._roundtrip(
            MedusaDraft(horizon=2), torch.randn(512, HIDDEN_SIZE), shim=True)
        self.assertIsNone(model.token_map)
        self.assertEqual({name for name, _ in model.named_parameters()} - loaded, set())

    def test_registered_shim_name_resolves(self) -> None:
        from vllm.model_executor.models.medusa import Medusa
        from fastdocling.backends import _vllm_medusa

        module, _, attr = _vllm_medusa.SHIM.partition(":")
        self.assertEqual(module, _vllm_medusa.__name__)
        self.assertTrue(issubclass(getattr(_vllm_medusa, attr), Medusa))


if __name__ == "__main__":
    unittest.main()
