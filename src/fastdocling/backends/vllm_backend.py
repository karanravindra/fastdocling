"""CUDA backend: hidden-state capture via vLLM's ``extract_hidden_states``.

Install with ``uv sync --extra cuda``.  vLLM pins ``torch==2.13.0`` exactly, which is why
the backends are mutually-exclusive extras rather than plain dependencies.

How vLLM does this (v0.18+, PR #33736): the ``extract_hidden_states`` speculative method
installs a dummy draft model whose "attention" layer writes the target's residual stream
straight into a KV cache, and a KV connector flushes that to safetensors.  Four facts about
it drive this module:

1. Layer ids follow HF's ``hidden_states`` tuple convention -- entry 0 is the embeddings,
   entry ``i+1`` is decoder layer ``i``'s output -- so our taps shift by one
   (``aux_layer_ids``).  vLLM stores ``hidden_states + residual``, the true residual
   stream, which is exactly what the MLX backend records.
2. The state at layer id ``num_hidden_layers`` is taken *before* the final norm (vLLM
   applies ``self.norm`` after collecting aux states).  We re-apply the target's RMSNorm
   in ``_apply_final_norm`` -- skip it and every trace is silently wrong, because
   ``data.py`` has no way to tell.
3. Only prompt tokens are saved unless a request passes ``include_output_tokens``, so the
   teacher-forced path runs with ``max_tokens=1`` and the whole page in the prompt.
4. granite-docling is an Idefics3 model, which vLLM does not declare as ``SupportsEagle3``
   even though its inner ``text_model`` is a ``LlamaModel`` that implements the mixin.
   ``_register_eagle3_shim`` re-registers it with the interface mixed in.

Chunked prefill is incompatible with the feature and is disabled.  On sm120 the flashinfer
sampler's JIT arch check misfires, so ``VLLM_USE_FLASHINFER_SAMPLER`` is forced off; we
decode greedily and never need it.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch

from .base import (
    END_TAG,
    MAX_NEW_TOKENS,
    PROMPT,
    Trace,
    clip_to_end_tag,
    eagle3_taps,
    finalize_trace,
)

# The MLX conversion of the target; on CUDA we want the original weights.
CUDA_MODEL_ID = "ibm-granite/granite-docling-258M"
FINAL_NORM_KEY = "model.text_model.norm.weight"


def _scalar(v: int) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.int32)


def aux_layer_ids(taps: Sequence[int], num_layers: int) -> list[int]:
    """Our decoder-layer indices -> vLLM's HF-tuple ids, plus the final layer.

    ``taps`` are outputs of decoder layers, so each maps to tuple entry ``tap + 1``.  The
    final entry ``num_layers`` is the last layer's output, which we normalize ourselves.
    """
    return [t + 1 for t in taps] + [num_layers]


def _register_eagle3_shim() -> None:
    """Register the shim in *this* process too.

    The ``vllm.general_plugins`` entry point already covers every vLLM process, including
    the spawned EngineCore.  This call is belt-and-braces for a run with plugins disabled
    (``VLLM_PLUGINS=``) and is idempotent.
    """
    from ._vllm_idefics3 import register

    register()


# How many pages stay resident in the engine at once.  Under ``_stream_requests`` this is a
# steady-state concurrency level, not a chunk size: a finished page is replaced immediately,
# so there is no drain and no boundary stall, and it does not bound crash-loss either
# (traces are saved as each page lands).
#
# It still sets throughput, because it sets how many sequences decode together.  Measured on
# 200 random corpus pages / one 5070 Ti, wall seconds including ~45 s of engine start:
#
#     concurrency     32     64    128
#     seconds        139    125    121
#
# Still rising at 128 rather than plateauing, so there may be more here -- but raising
# max_num_batched_tokens (which sets the encoder budget) costs activation memory, and 32768
# OOMs on a 16 GB card.  Re-sweep both together if you move to a bigger GPU.  Run-to-run
# variance is ~25%, so treat these as a gradient, not exact figures.
VLLM_CHUNK = 128


class VLLMBackend:
    """Trace extraction on CUDA.  Mirrors ``mlx_backend.TraceExtractor``'s interface."""

    needs_length_sorted_input = False   # vLLM schedules each request independently

    def __init__(self, model_id: str = CUDA_MODEL_ID, prompt: str = PROMPT,
                 taps: Sequence[int] | None = None, *, keep_taps: bool = False,
                 storage_path: str | None = None, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = MAX_NEW_TOKENS, enforce_eager: bool = False,
                 read_workers: int = 8, max_num_batched_tokens: int = 16384):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # Anything that touched torch.cuda in this process (backend detection does)
        # poisons a forked EngineCore: "Cannot re-initialize CUDA in forked subprocess".
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        _register_eagle3_shim()

        from transformers import AutoProcessor
        from vllm import LLM

        self.keep_taps = keep_taps
        self.read_workers = read_workers
        self._boundary_checked = False
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        cfg = self._text_config(model_id)
        self.num_layers = cfg["num_hidden_layers"]
        self.taps = tuple(taps) if taps is not None else eagle3_taps(self.num_layers)
        self.aux_ids = aux_layer_ids(self.taps, self.num_layers)

        self.max_model_len = max_model_len
        self.raw_prompt = prompt
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )

        self._storage = storage_path or os.path.join(os.getcwd(), ".vllm-hidden-states")
        os.makedirs(self._storage, exist_ok=True)
        self.norm_weight, self.norm_eps = self._load_final_norm(model_id, cfg)

        self.llm = LLM(
            model=model_id,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": self.aux_ids}},
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {"shared_storage_path": self._storage},
            },
            enable_chunked_prefill=False,     # incompatible with hidden-state extraction
            # This is also the encoder cache budget: vLLM sets both encoder_cache_size and
            # max_num_encoder_input_tokens from it (config/scheduler.py).  Each page's image
            # expands to ~800+ vision tokens, so the 8192 default holds only ~9 images and
            # throttles how fast pages can be admitted -- the batch starves before it fills.
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt={"image": 1},
        )

    # -- model metadata ------------------------------------------------------------------
    @staticmethod
    def _text_config(model_id: str) -> dict:
        import json
        from huggingface_hub import hf_hub_download

        return json.load(open(hf_hub_download(model_id, "config.json")))["text_config"]

    @staticmethod
    def _load_final_norm(model_id: str, cfg: dict) -> tuple[torch.Tensor, float]:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        with safe_open(hf_hub_download(model_id, "model.safetensors"), framework="pt") as f:
            w = f.get_tensor(FINAL_NORM_KEY).float()
        return w, float(cfg.get("rms_norm_eps", 1e-5))

    def _apply_final_norm(self, h: torch.Tensor) -> torch.Tensor:
        """The target's final RMSNorm, which vLLM applies *after* collecting aux states.

        Follows HF's LlamaRMSNorm exactly: normalize in float32, cast back, then scale --
        the cast-before-scale order is load-bearing for bit-level agreement.
        """
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        return self.norm_weight * x.to(h.dtype)

    # -- prompt encoding -----------------------------------------------------------------
    def _prompt_token_len(self, image) -> int:
        """Length of the image+prompt prefix, with vision placeholders already expanded."""
        enc = self.processor(images=[image], text=self.prompt, return_tensors="pt")
        return int(enc["input_ids"].shape[1])

    # -- reading what the connector wrote ------------------------------------------------
    def _read_states(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            cleanup_hidden_states,
            load_hidden_states,
        )

        d = load_hidden_states(path)          # blocks on a shared flock until the write lands
        hs, ids = d["hidden_states"], d["token_ids"]
        cleanup_hidden_states(path)
        return hs, ids

    def _read_one(self, idx: int, path: str, prompt_len: int, extra: dict) -> tuple[int, Trace]:
        """Turn one finished request's handoff file into a trace.  Runs on a worker thread."""
        hs, ids = self._read_states(path)
        trace = self._to_trace(hs, ids, prompt_len)
        trace.update(extra)
        return idx, trace

    def _stream_requests(self, items: Iterable[tuple], params, concurrency: int, finish
                         ) -> Iterator[tuple[int, Trace]]:
        """Keep ``concurrency`` requests resident in the engine, yielding pages as they finish.

        The obvious loop -- ``llm.generate(chunk)`` per chunk -- wastes the GPU twice over: a
        call returns only when its slowest page does, so the batch drains to a handful of
        stragglers, and then the GPU sits at 0% while the chunk's hidden-state files are read
        back.  Measured on this corpus that was a sawtooth between 99% and 61% with a full
        stall at every boundary.

        Driving ``add_request``/``step`` directly instead means a finished page is replaced
        immediately, so the batch never drains and file I/O overlaps the next step.  Images
        are pulled by a feeder thread because loading a PNG inside the step loop would stall
        it, and reads run on a pool whose results are handed back only when already done.

        ``items`` yields ``(index, prompt, aux)``; ``finish(out, aux)`` returns
        ``(prompt_len, extra_trace_keys)``.
        """
        from concurrent.futures import ThreadPoolExecutor

        engine = self.llm.llm_engine
        buf: queue.Queue = queue.Queue(maxsize=max(4, concurrency))
        DONE = object()

        def feed():
            try:
                for item in items:
                    buf.put(item)
            finally:
                buf.put(DONE)

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()

        inflight: dict[str, tuple[int, object]] = {}
        futures: list = []
        exhausted = False
        with ThreadPoolExecutor(max_workers=self.read_workers) as pool:
            while True:
                # Top up.  Block for the next page only when the engine would otherwise have
                # nothing to run; otherwise take what the feeder has ready and go step.
                while not exhausted and len(inflight) < concurrency:
                    try:
                        item = buf.get(block=not inflight)
                    except queue.Empty:
                        break
                    if item is DONE:
                        exhausted = True
                        break
                    idx, prompt, aux = item
                    rid = str(idx)
                    engine.add_request(rid, prompt, params)
                    inflight[rid] = (idx, aux)

                if not inflight:
                    break

                for out in engine.step():
                    if not out.finished:
                        continue
                    idx, aux = inflight.pop(out.request_id)
                    prompt_len, extra = finish(out, aux)
                    futures.append(pool.submit(
                        self._read_one, idx, out.kv_transfer_params["hidden_states_path"],
                        prompt_len, extra))

                # Hand back completed I/O without blocking the step loop.
                for f in [f for f in futures if f.done()]:
                    futures.remove(f)
                    yield f.result()

            for f in futures:
                yield f.result()

    def _to_trace(self, hs: torch.Tensor, ids: torch.Tensor, prompt_len: int) -> Trace:
        """``hs[N, len(aux_ids), 576]`` -> the trace keys ``data.py`` expects."""
        trace: Trace = {}
        for name_idx, tap in enumerate(self.taps):
            trace[f"layer_{tap}"] = hs[:, name_idx].to(torch.float16)
        trace["last_hidden_state"] = self._apply_final_norm(hs[:, len(self.taps)]).to(torch.float16)
        trace["token_ids"] = ids.to(torch.int32)
        trace["prompt_len"] = _scalar(prompt_len)
        trace["completion_start"] = _scalar(prompt_len)
        return finalize_trace(trace, prompt_len, self.keep_taps, _scalar)

    # -- teacher-forced ------------------------------------------------------------------
    def extract(self, items: Iterable[tuple[object, str]], batch_size: int = VLLM_CHUNK
                ) -> Iterator[tuple[int, Trace]]:
        """One prefill per ``(image, gold doctags)`` page, ``batch_size`` resident at a time."""
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=1, skip_special_tokens=False,
                                extra_args={"kv_transfer_params": {}})

        def requests():
            for i, (img, dt) in enumerate(items):
                yield i, {"prompt": self.prompt + dt, "multi_modal_data": {"image": img}}, (img, dt)

        def finish(out, aux):
            img, dt = aux
            # The completion is appended last, so its token count locates the boundary
            # without re-running the (expensive) image processor on every page.
            prompt_len = len(out.prompt_token_ids) - len(
                self.tokenizer.encode(dt, add_special_tokens=False))
            self._check_boundary_once(img, prompt_len)
            return prompt_len, {}

        yield from self._stream_requests(requests(), params, batch_size, finish)

    def _check_boundary_once(self, image, prompt_len: int) -> None:
        """Verify the prompt/completion split against the processor, for the first page only.

        ``prompt_len`` is inferred by subtracting the completion's token count, but vLLM
        tokenizes ``prompt + doctags`` as one string and BPE can merge across the seam.  A
        silent off-by-one shifts every training window, so check it -- but only once: the
        prompt *length* varies per page (Idefics3 tiles by image size), while the seam
        itself does not, since the prompt always ends with the same assistant-turn marker
        and every label starts with ``<doctag>``.  One check settles the merge behaviour
        for the whole corpus, and the processor call it needs is not cheap.
        """
        if self._boundary_checked:
            return
        self._boundary_checked = True
        expected = self._prompt_token_len(image)
        if expected != prompt_len:
            raise ValueError(
                f"prompt/completion boundary moved: processor says prompt_len={expected}, "
                f"token arithmetic says {prompt_len}.  BPE merged across the seam."
            )

    # -- decode + record in one pass -----------------------------------------------------
    def generate_traces(self, images: Iterable[object], batch_size: int = VLLM_CHUNK,
                        max_tokens: int = MAX_NEW_TOKENS) -> Iterator[tuple[int, Trace]]:
        """Greedy-decode DocTags and keep the states that produced them.

        ``include_output_tokens`` lifts vLLM's prompt-only default; the connector then saves
        ``all_token_ids[:-1]`` -- every position whose state predicted the next token.
        """
        from vllm import SamplingParams

        # skip_special_tokens=False is load-bearing: DocTags markup (</text>, </doctag>) is
        # *special* in this tokenizer, so the default detokenizer silently strips the entire
        # structure and leaves only running text.  Generation ends on EOS, which the model
        # emits right after </doctag>.
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, skip_special_tokens=False,
                                extra_args={"kv_transfer_params": {"include_output_tokens": True}})

        def requests():
            for i, img in enumerate(images):
                yield i, {"prompt": self.prompt, "multi_modal_data": {"image": img}}, None

        def finish(out, _aux):
            text = out.outputs[0].text
            return len(out.prompt_token_ids), {
                "doctags": clip_to_end_tag(text), "truncated": END_TAG not in text}

        yield from self._stream_requests(requests(), params, batch_size, finish)

    # -- persistence ---------------------------------------------------------------------
    def save(self, path: Path, trace: Trace) -> None:
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in trace.items()}, str(path))
