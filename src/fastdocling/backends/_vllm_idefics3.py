"""Idefics3 + the EAGLE-3 interface, registered with vLLM as a plugin.

vLLM gates ``extract_hidden_states`` on ``isinstance(model, SupportsEagle3)``, and
``Idefics3ForConditionalGeneration`` declares only ``SupportsMultiModal``/``SupportsLoRA`` --
even though the ``LlamaModel`` it wraps carries ``EagleModelMixin`` and can emit aux states.
Mixing the protocol in is the whole fix: ``SupportsEagle3.set_aux_hidden_state_layers``
already unwraps multimodal wrappers through ``get_language_model()``.

Registration runs via the ``vllm.general_plugins`` entry point (declared in pyproject)
rather than from ``VLLMBackend.__init__``.  The EngineCore runs in a *spawned* interpreter:
it re-imports vLLM from scratch and never executes the parent's registration call, so an
in-process ``register_model`` is invisible to the process that actually loads the weights.
vLLM loads its plugins in every process, which is the only hook that reaches the worker.
"""

from __future__ import annotations

SHIM = "fastdocling.backends._vllm_idefics3:Idefics3Eagle3"


def register() -> None:
    """vllm.general_plugins entry point; runs once per vLLM process."""
    from vllm.model_executor.models.registry import ModelRegistry

    # Registered by import path so this stays lazy -- importing the model class eagerly
    # would pull torch/CUDA into processes that only wanted the plugin list.
    ModelRegistry.register_model("Idefics3ForConditionalGeneration", SHIM)


def __getattr__(name: str):
    """Build the subclass only when vLLM resolves ``SHIM``, i.e. inside the worker."""
    if name == "Idefics3Eagle3":
        from vllm.model_executor.models.idefics3 import Idefics3ForConditionalGeneration
        from vllm.model_executor.models.interfaces import SupportsEagle3

        class Idefics3Eagle3(Idefics3ForConditionalGeneration, SupportsEagle3):
            """Idefics3 that advertises EAGLE-3 aux hidden states."""

        globals()["Idefics3Eagle3"] = Idefics3Eagle3
        return Idefics3Eagle3
    raise AttributeError(name)
