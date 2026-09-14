"""Medusa with a loadable ``token_map``, registered with vLLM as a plugin.

vLLM's ``Medusa.load_weights`` handles a ``token_map`` -- it assigns
``self.token_map = nn.Parameter(loaded_weight, requires_grad=False)`` -- but never adds
``"token_map"`` to the set of loaded names it returns.  Assigning the parameter *registers* it on
the module, so ``default_loader`` then diffs the module's parameters against that set, finds
``token_map`` missing, and rejects the checkpoint:

    ValueError: Following weights were not initialized from checkpoint: {'token_map'}

Any Medusa checkpoint carrying a token map therefore fails to load on vLLM 0.29, and one cannot
simply drop the map either: the class asserts ``truncated_vocab_size == vocab_size or token_map
is not None``.  So without this shim the draft head must span the *entire* vocabulary.

The same method has a second, quieter defect this shim also repairs.  It ends with

    self.token_map.to(device=self.lm_heads[0].weight.device)

whose result is discarded -- ``Tensor.to`` is not in-place.  Weights arrive from the safetensors
iterator on the *host*, so the map stays a CPU int64 tensor while every other parameter was built
inside the loader's ``with target_device`` block.  ``compute_logits`` then runs
``logits[..., self.token_map] = _logits`` with a CPU index into an accelerator tensor: torch
tolerates it, so nothing breaks, but it copies the whole map host-to-device once per Medusa head
per drafted round (~287 KB x num_heads at 35.9k kept tokens) and synchronises to do it.  That is
paid on exactly the rounds pruning was meant to make cheaper, which is a fair suspect for why
pruning 100,352 -> 35,878 tokens measured only 1.7% faster.

That is not a cosmetic cost.  The head runs its shared ``lm_head`` once per Medusa head on every
drafted round, so the projection is ``num_heads x [vocab, hidden]``.  Measured on an RTX 5070 Ti
with granite-docling-258M: 1.47 ms per round over the full 100,352 tokens, against a 1.86 ms
target step -- the head nearly doubles the cost of a round.  Restricting it to the ~35.9k tokens
DocTags actually emits scales that projection down by the same factor.

Registration goes through ``vllm.general_plugins`` for the reason ``_vllm_idefics3`` documents:
the EngineCore runs in a spawned interpreter that re-imports vLLM from scratch, so a patch
applied in the parent process never reaches the process that loads the weights.
"""

from __future__ import annotations

SHIM = "fastdocling.backends._vllm_medusa:MedusaTokenMap"


def register() -> None:
    """vllm.general_plugins entry point; runs once per vLLM process."""
    from vllm.model_executor.models.registry import ModelRegistry

    # By import path, so nothing heavy is imported into processes that only wanted the plugin list.
    ModelRegistry.register_model("MedusaModel", SHIM)


def __getattr__(name: str):
    """Build the subclass only when vLLM resolves ``SHIM``, i.e. inside the worker."""
    if name == "MedusaTokenMap":
        from torch import nn
        from vllm.model_executor.models.medusa import Medusa

        class MedusaTokenMap(Medusa):
            """Medusa that reports the ``token_map`` it loaded, and keeps it on the right device."""

            def load_weights(self, weights):
                loaded = super().load_weights(weights)
                token_map = getattr(self, "token_map", None)
                if token_map is None:
                    return loaded
                # super()'s own `.to(device=...)` throws its result away, so do the move here.
                device = self.lm_heads[0].weight.device
                if token_map.device != device:
                    self.token_map = nn.Parameter(
                        token_map.data.to(device), requires_grad=False)
                # super() sets self.token_map (registering the parameter) but omits the name.
                loaded.add("token_map")
                return loaded

        globals()["MedusaTokenMap"] = MedusaTokenMap
        return MedusaTokenMap
    raise AttributeError(name)
