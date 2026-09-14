"""CUDA backend: the target under a production runtime, with and without vLLM's own drafter.

This backend answers two different questions, and it is worth keeping them apart.

**The baseline.**  Without a ``speculative_config`` this is plain greedy decoding, and the number
it produces is ``TARGET_DECODE_TPS``: the rate at which a real serving runtime decodes one page.
HF eager reaches ~55 tok/s on an RTX 5070 Ti because 30 layers of small kernels are launch-bound
(measured: 2.2 ms of GPU work per step behind 18 ms of CPU dispatch); vLLM, with CUDA graphs and
a paged cache, is several times that on the same card.  A speedup quoted over the slow baseline
flatters the draft, so acceptance is measured on ``transformers``/``mlx`` and then converted with
*this* rate.

**The bar to beat.**  With a ``speculative_config`` vLLM runs *its own* speculative decoding, and
that is the honest competitor for the latent draft: whatever the trained draft buys has to beat
what a production runtime already gives away for free.  The methods that need no trained model
are ``ngram`` (propose the continuation of the longest matching n-gram seen so far) and its GPU
variant; DocTags output is highly repetitive -- tables, repeated ``<loc_N>`` runs, form rows --
which is exactly the regime ngram drafting was built for.  ``suffix`` needs ``arctic_inference``;
``eagle``/``eagle3``/``medusa``/``draft_model`` need a draft checkpoint in vLLM's own format.

**The latent draft still cannot run here, but not for the reason it is tempting to give.**
Rollback is *not* the obstacle: vLLM rejects and rolls back internally for every drafter it
hosts (``num_rejected_tokens`` in the GPU model runner).  That only blocks driving vLLM's loop
from outside Python, which is what ``TorchDecoder`` does.  The real obstacle is the proposer
interface.  ``custom_class`` receives token ids and never hidden states.  ``medusa`` does receive
hidden states and does parallel per-head argmax -- the same shape of model as ours -- but it is
handed ``[num_reqs, hidden_size]``, the *current* position only, with nowhere to put the window
of past states the block attends over.  ``eagle``/``eagle3`` take hidden states but want an
autoregressive drafter in their own model format.  And every one of them is loaded from disk in
vLLM's format, not from a live torch module.  So ``attach_draft`` raises, and ``use_draft=True``
here means *vLLM's* drafter, never ours.  ``accepts_latent_draft`` is the flag to branch on.

**If that integration is ever attempted, mind which method.**  Configuring ``medusa`` forfeits
async scheduling *and* the V2 model runner, exactly as ``ngram`` does.  The sweep's
``baseline async=off runner=v1`` control measures that combination at 522 tok/s against an
833 tok/s plain baseline -- a 1.92 ms target step instead of 1.20 ms -- which raises break-even
to roughly 1.68 tokens/step, while the single-state interface removes the very window that earns
the acceptance in the first place.  ``eagle3`` keeps both (it is in ``EagleModelTypes`` and the V2 allowlist), so the
target step stays ~1.2 ms and break-even stays near 1.1.

One engine is one speculative configuration -- vLLM bakes it in at construction -- so comparing
settings means building and tearing down an engine per setting.  ``sweep_speculative`` does that,
reusing cached page decodes so a repeated sweep never rebuilds an engine it does not need.

Install with ``uv sync --extra cuda``.
"""

from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Iterable, Sequence

from .base import FEATURE_KEYS_LAST, HF_MODEL_ID, PROMPT, SpecResult, cached_generate

UNSUPPORTED = (
    "vLLM loads drafters from disk in its own formats, and none of their proposer interfaces "
    "accepts a window of target hidden states (medusa gets the current position only), so the "
    "*latent* draft cannot be attached here.  Use backend 'transformers' (CUDA/MPS/CPU) or 'mlx' "
    "to measure the trained draft; pass speculative_config= to measure vLLM's own drafter."
)

# Speculative settings worth trying on DocTags without training anything.  ngram proposes the
# continuation of the longest recent n-gram match, so the useful knobs are how many tokens it
# proposes and how long a match it insists on before proposing at all.
# Only ``num_speculative_tokens`` is swept.  Raising ``prompt_lookup_min`` above 2 was measured
# strictly worse at every k (a longer required match proposes less often without proposing
# better), and it is also the one knob that would *not* need its own engine -- the ngram matcher
# reads ``prompt_lookup_min``/``max`` as plain ints, whereas k sizes pre-allocated buffers and the
# scheduler's draft budget.  So the reusable dimension is the one worth dropping: one engine per
# k, four engines in total.
# Configuring any ngram drafter makes vLLM disable async scheduling ("Async scheduling not
# supported with ngram-based speculative decoding and will be disabled"), which is a cost the
# drafter does not choose and the baseline does not pay.  This control -- no drafter, async
# scheduling off -- separates that from what the drafting itself costs.
ASYNC_CONTROL: dict = {"__engine__": {"async_scheduling": False}}

# Configuring ngram *also* drops vLLM to the V1 model runner ("Model Runner V2 does not yet
# support ngram/ngram_gpu speculative decoding"), a second cost the drafter does not choose.
# This control matches both conditions with no drafter, so what is left over really is drafting.
DRAFTER_CONTROL: dict = {"__engine__": {"async_scheduling": False},
                         "__env__": {"VLLM_USE_V2_MODEL_RUNNER": "0"}}

NGRAM_SWEEP: tuple[dict, ...] = (
    {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_min": 2, "prompt_lookup_max": 8},
    {"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_min": 2, "prompt_lookup_max": 8},
    {"method": "ngram", "num_speculative_tokens": 8, "prompt_lookup_min": 2, "prompt_lookup_max": 8},
)

_SPEC_COUNTERS = {
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted",
}
_SPEC_VECTOR = "vllm:spec_decode_num_accepted_tokens_per_pos"


class GpuSampler:
    """Mean GPU utilisation over a timed region, sampled in a side thread.

    Batch-1 decoding of a 258M model leaves the card almost idle -- that idleness is the headroom
    speculative decoding spends -- so the benchmarks below report it rather than leaving it to be
    discovered with ``nvidia-smi``.  Utilisation is device-wide: anything else running on the GPU
    is included, so read it as an upper bound on this process's share.
    """

    def __init__(self, interval: float = 0.05):
        self.interval, self.samples = interval, []
        self._stop = threading.Event()
        self._thread = None

    def _run(self) -> None:
        import torch

        while not self._stop.wait(self.interval):
            try:
                self.samples.append(torch.cuda.utilization())
            except Exception:       # no NVML, or the device went away -- report nothing
                return

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float("nan")


class VLLMDecoder:
    """Greedy decoding of granite-docling through vLLM, timed per page.

    ``speculative_config`` is passed straight to vLLM (see ``NGRAM_SWEEP`` for the shape).  With
    it, ``generate(use_draft=True)`` runs vLLM's drafter and reports acceptance; without it, only
    ``use_draft=False`` is available and the result is the plain baseline.
    """

    name = "vllm"
    accepts_latent_draft = False    # the trained latent draft never runs here; see the docstring

    def __init__(self, model_id: str = HF_MODEL_ID, prompt: str = PROMPT, *,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 8192,
                 enforce_eager: bool = False, speculative_config: dict | None = None,
                 env: dict | None = None, **llm_kwargs):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # sm120 JIT arch check misfires
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from transformers import AutoProcessor

        self.model_id = model_id
        self.speculative_config = dict(speculative_config) if speculative_config else None
        self.supports_draft = self.speculative_config is not None
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
        )
        self._llm = None
        self._warm = False
        self._env = dict(env or {})   # applied while the engine is built; see the ``llm`` property
        self._llm_kwargs = dict(
            model=model_id,
            # Both timed runs below send the same page; with prefix caching on, the second
            # would skip prefill entirely and the split would be nonsense.
            enable_prefix_caching=False,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt={"image": 1},
            # Acceptance is only readable through the metrics counters, which stat logging feeds.
            disable_log_stats=not self.supports_draft,
        )
        if self.speculative_config:
            self._llm_kwargs["speculative_config"] = self.speculative_config
        self._llm_kwargs.update(llm_kwargs)   # e.g. async_scheduling=False to isolate its effect

    # -- engine lifecycle ----------------------------------------------------------------
    @property
    def llm(self):
        """The vLLM engine, built on first use.

        Loading is deferred so that a notebook run whose page decodes are all cached never pays
        the ~50 s engine start (nor the GPU memory) for a model it is not going to run.
        """
        if self._llm is None:
            from vllm import LLM

            # Some knobs (the model-runner version among them) are read from the environment at
            # construction, not from the config object, and the engine child process inherits it.
            saved = {k: os.environ.get(k) for k in self._env}
            os.environ.update(self._env)
            try:
                self._llm = LLM(**self._llm_kwargs)
            finally:
                for k, v in saved.items():
                    os.environ[k] = v if v is not None else os.environ.pop(k, "")
        return self._llm

    def close(self) -> None:
        """Shut the engine down and release its GPU memory, so another config can be built."""
        llm, self._llm, self._warm = self._llm, None, False
        if llm is None:
            return
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()

    def __enter__(self) -> "VLLMDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def attach_draft(self, draft: Any, lm_head: Any = None, **kwargs) -> None:
        raise NotImplementedError(UNSUPPORTED)

    # -- acceptance ----------------------------------------------------------------------
    def _spec_stats(self) -> dict:
        """Cumulative drafting counters for this engine (empty when no drafter is configured).

        vLLM reports these engine-wide rather than per request, so ``generate`` diffs a snapshot
        taken either side of the timed run.
        """
        if not self.supports_draft:
            return {}
        out = {v: 0 for v in _SPEC_COUNTERS.values()}
        out["accepted_per_pos"] = []
        for metric in self.llm.get_metrics():
            name = getattr(metric, "name", "")
            if name in _SPEC_COUNTERS:
                out[_SPEC_COUNTERS[name]] = int(getattr(metric, "value", 0) or 0)
            elif name == _SPEC_VECTOR:
                out["accepted_per_pos"] = [int(v) for v in (getattr(metric, "values", None) or [])]
        return out

    # -- the loop ------------------------------------------------------------------------
    def generate(self, image, *, use_draft: bool = False, horizon: int = 2,
                 feature_keys: Sequence[str] = FEATURE_KEYS_LAST,
                 history_limit: int | None = 256, max_tokens: int = 8192) -> SpecResult:
        """Decode one page and split prefill from decode with two timed runs.

        ``LLM.generate`` reports no per-phase timing, so the page is decoded twice: once capped
        at one token (prefill plus a single step) and once in full.  The difference is ``n - 1``
        decode steps, which gives a per-token cost, and subtracting one step from the capped run
        leaves prefill.

        ``use_draft`` selects vLLM's *own* drafter, which must have been configured at
        construction; the trained latent draft never runs here, so ``feature_keys``,
        ``history_limit`` and ``horizon`` are accepted for interface parity and ignored.
        """
        from vllm import SamplingParams

        if use_draft and not self.supports_draft:
            raise NotImplementedError(
                "no speculative_config on this decoder: build it with "
                "get_decoder('vllm', speculative_config={'method': 'ngram', ...}).  " + UNSUPPORTED)

        # Build the engine *before* the clock starts.  ``self.llm`` is lazy, and on the first
        # call it pays ~40 s of startup and graph capture; inside the timed region that lands in
        # ``t_one`` and drives the prefill/decode split negative.
        llm = self.llm

        req = {"prompt": self.prompt, "multi_modal_data": {"image": image}}
        # skip_special_tokens=False is load-bearing: DocTags markup is *special* in this
        # tokenizer, and the default detokenizer strips the entire document structure.
        def run(limit: int):
            params = SamplingParams(temperature=0.0, max_tokens=limit, skip_special_tokens=False)
            t0 = perf_counter()
            out = llm.generate([req], params, use_tqdm=False)[0]
            return out, perf_counter() - t0

        if not self._warm:
            run(8)                 # first-call lazy work, off the clock; see the note above
            self._warm = True

        _, t_one = run(1)
        before = self._spec_stats()
        with GpuSampler() as gpu:
            out, t_full = run(max_tokens)
        after = self._spec_stats()

        tokens = list(out.outputs[0].token_ids)
        n = len(tokens)
        per_token = (t_full - t_one) / max(1, n - 1)

        # A drafter that is configured but idle (no n-gram match all page) still decodes one
        # token per round, so rounds falls back to n and tokens_per_round to 1.0.
        accepted = after.get("accepted", 0) - before.get("accepted", 0)
        per_pos = [a - b for a, b in zip(after.get("accepted_per_pos", []),
                                         before.get("accepted_per_pos", []))]
        return SpecResult(
            tokens, prefill_seconds=max(0.0, t_one - per_token), decode_seconds=per_token * n,
            rounds=max(1, n - accepted), accepted=accepted, backend=self.name, gpu_util=gpu.mean,
            drafts=after.get("drafts", 0) - before.get("drafts", 0),
            draft_tokens=after.get("draft_tokens", 0) - before.get("draft_tokens", 0),
            accepted_per_pos=per_pos,
        )

    def generate_batch(self, images: Sequence, *, max_tokens: int = 8192) -> dict:
        """Decode every page *concurrently* in one scheduler pass: throughput, not latency.

        ``generate`` measures one page at a time, which is what speculative decoding is for.
        This is the other half of "what can vLLM do on this card": with several pages in flight
        the scheduler batches their decode steps and the weights are read once per step for the
        whole batch, so the same weight traffic carries several sequences.

        Prefill and decode interleave once requests are batched, so there is no honest per-phase
        split here -- the number reported is aggregate generation throughput, the metric a serving
        deployment is quoted in.
        """
        from vllm import SamplingParams

        llm = self.llm
        reqs = [{"prompt": self.prompt, "multi_modal_data": {"image": im}} for im in images]
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, skip_special_tokens=False)
        if not self._warm:
            llm.generate(reqs[:1], SamplingParams(temperature=0.0, max_tokens=8,
                                                  skip_special_tokens=False), use_tqdm=False)
            self._warm = True

        before = self._spec_stats()
        with GpuSampler() as gpu:
            t0 = perf_counter()
            outs = llm.generate(reqs, params, use_tqdm=False)
            seconds = perf_counter() - t0
        after = self._spec_stats()

        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        accepted = after.get("accepted", 0) - before.get("accepted", 0)
        return {"pages": len(reqs), "tokens": tokens, "seconds": seconds,
                "tps": tokens / seconds, "gpu_util": gpu.mean,
                "tokens_per_round": tokens / max(1, tokens - accepted), "accepted": accepted,
                "draft_tokens": after.get("draft_tokens", 0) - before.get("draft_tokens", 0)}


def sweep_speculative(pages: Iterable[str | Path], configs: Iterable[dict | None], *,
                      cache_dir: str | Path, model_id: str = HF_MODEL_ID,
                      refresh: bool = False, max_tokens: int = 8192, batched: bool = True,
                      progress=None, **decoder_kwargs) -> list[dict]:
    """Decode ``pages`` once per speculative configuration and return one row per (config, page).

    ``configs`` entries are vLLM ``speculative_config`` dicts, or ``None`` for the no-drafter
    baseline.  Each one needs its own engine, so they are built and torn down in turn -- but a
    config whose pages are *all* already in ``cache_dir`` never builds an engine at all, which is
    what makes re-running a sweep nearly free.  Timings come from whichever session measured
    them; pass ``refresh=True`` to re-run and re-time.

    Each row carries ``mode``.  ``"latency"`` rows are one page at a time -- the regime
    speculative decoding exists for, and the one the latent draft is measured in.  With
    ``batched`` (the default) each config also contributes one ``"throughput"`` row that decodes
    every page concurrently; that is what the card can actually do, and it is where a drafter
    normally loses, having nothing spare left to spend.  Engine build dominates a cold sweep
    (~50 s each, mostly torch.compile), so keep the config list short -- see ``NGRAM_SWEEP``.
    """
    pages = [Path(p) for p in pages]
    rows: list[dict] = []
    for config in configs:
        # A config may carry engine-level overrides under "__engine__" (stripped before it
        # reaches vLLM).  That is how the async-scheduling control gets into the sweep: ngram
        # forces async scheduling off, so the only way to read the drafter's own cost is to
        # measure a no-drafter engine with it off too.
        config = dict(config) if config else config
        engine = config.pop("__engine__", {}) if config else {}
        env = config.pop("__env__", {}) if config else {}
        spec = config or None
        label = _label(spec, engine, env)
        decoder = VLLMDecoder(model_id=model_id, speculative_config=spec, env=env,
                              **{**decoder_kwargs, **engine})
        key = {"kind": "vllm-sweep", "backend": decoder.name, "model": model_id,
               "spec": spec, "engine": engine, "env": env, "max_tokens": max_tokens}
        try:
            for page in pages:
                res, hit = cached_generate(decoder, page, use_draft=spec is not None,
                                           cache_dir=cache_dir, key=key, refresh=refresh,
                                           max_tokens=max_tokens)
                rows.append({"config": label, "spec": spec, "page": page,
                             "tokens": len(res.tokens), "decode_tps": res.decode_tps,
                             "tokens_per_round": res.tokens_per_round,
                             "draft_acceptance": res.draft_acceptance, "gpu_util": res.gpu_util,
                             "drafts": res.drafts, "draft_tokens": res.draft_tokens,
                             "accepted_per_pos": res.accepted_per_pos, "cached": hit,
                             "mode": "latency"})
                if progress is not None:
                    progress.update()
            if batched:
                rows.append({"config": label, "spec": spec, "mode": "throughput",
                             **_batched(decoder, pages, key, cache_dir, refresh, max_tokens)})
                if progress is not None:
                    progress.update()
        finally:
            decoder.close()
    return rows


def _batched(decoder: VLLMDecoder, pages: list[Path], key: dict, cache_dir: str | Path,
             refresh: bool, max_tokens: int) -> dict:
    """Batched-throughput measurement for one config, cached like a page decode."""
    import hashlib

    from .base import _file_sha1

    digest = hashlib.sha1(json.dumps(
        {**key, "mode": "throughput", "pages": [_file_sha1(p) for p in pages]},
        sort_keys=True, default=str).encode()).hexdigest()
    path = Path(cache_dir) / f"{digest}.json"
    if path.exists() and not refresh:
        return {**json.loads(path.read_text())["result"], "cached": True}

    from transformers.image_utils import load_image

    out = decoder.generate_batch([load_image(str(p)) for p in pages], max_tokens=max_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": {**key, "mode": "throughput"},
                               "pages": [str(p) for p in pages], "result": out}))
    tmp.replace(path)
    return {**out, "cached": False}


def _label(config: dict | None, engine: dict | None = None, env: dict | None = None) -> str:
    """Compact name for a swept configuration, e.g. ``ngram k=5`` or ``baseline async=off``."""
    if config is None:
        parts = ["baseline"]
    else:
        parts = [str(config.get("method", "?")), f"k={config.get('num_speculative_tokens', '?')}"]
        if config.get("prompt_lookup_min", 2) != 2:
            parts.append(f"min={config['prompt_lookup_min']}")
    if engine and engine.get("async_scheduling") is False:
        parts.append("async=off")
    if env and env.get("VLLM_USE_V2_MODEL_RUNNER") == "0":
        parts.append("runner=v1")
    return " ".join(parts)
