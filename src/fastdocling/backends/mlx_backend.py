"""Apple Silicon backend: greedy decode + hidden-state capture via mlx-vlm.

Install with ``uv sync --extra mlx``.  Importing this module on a machine without a
native MLX build raises ImportError; ``fastdocling.backends.get_backend`` reports that
as a readable message.

The two hot paths both avoid a second pass over the model:

    generate_traces  greedy-decodes with continuous batching and records the residual
                     stream at every step, so labels and states come from one decode
    extract          teacher-forced: labels are supplied, one prefill per page

``mx.async_eval`` keeps the GPU a step ahead of Python in both; traces are yielded one
batch behind so the conversion in ``save`` never stalls the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import mlx.core as mx
from mlx_vlm import batch_generate, load, stream_generate
from mlx_vlm.models.cache import BatchKVCache
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.generate import prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from ..target import MODEL_ID
from .base import (
    END_OF_UTTERANCE_ID,
    END_TAG,
    END_TOKEN_ID,
    GEN_BATCH_SIZE,
    MAX_NEW_TOKENS,
    PROMPT,
    Trace,
    clip_to_end_tag,
    eagle3_taps,
    finalize_trace,
)


def _scalar(v: int) -> mx.array:
    return mx.array([v])


class TraceExtractor:
    needs_length_sorted_input = True   # refill groups must share a prompt length

    def __init__(self, model_id: str = MODEL_ID, prompt: str = PROMPT, taps: Sequence[int] | None = None,
                 *, model=None, processor=None, config=None, keep_taps: bool = False):
        """Pass ``model``/``processor``/``config`` to reuse an already-loaded model.

        ``keep_taps``: store the EAGLE-3 layer taps in traces (4x larger); off by default."""
        self.keep_taps = keep_taps
        if model is None:
            model, processor = load(model_id)
            config = load_config(model_id)
        self.model, self.processor, self.config = model, processor, config
        self.raw_prompt = prompt
        self.prompt = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        self.lm = self.model.language_model
        self.taps = tuple(taps) if taps is not None else eagle3_taps(len(self.lm.layers))

    # -- label generation --------------------------------------------------------------
    def generate(self, image, max_tokens: int = MAX_NEW_TOKENS) -> str:
        """Greedy-decode DocTags for one page, stopping at the DocTags terminator."""
        text = ""
        for tok in stream_generate(self.model, self.processor, self.prompt, [image],
                                   max_tokens=max_tokens, temp=0.0, verbose=False):
            text += tok.text
            if END_TAG in text:
                return clip_to_end_tag(text)
        return text  # truncated page; caller decides whether to keep it

    def generate_batch(self, images: Sequence[object], max_tokens: int = MAX_NEW_TOKENS) -> list[str]:
        """Greedy-decode several pages at once (left-padded batch, shared decode steps).

        Roughly 2x the per-page throughput of ``generate`` at batch 8.  Batched bf16 kernels can
        flip near-tied argmax choices (typically +/-1 in <loc_N> coordinates), so labels differ
        slightly from sequential decoding; they remain the target's own greedy output.
        """
        if len(images) == 1:
            return [self.generate(images[0], max_tokens)]
        out = batch_generate(
            self.model, self.processor, images=list(images), prompts=[self.raw_prompt] * len(images),
            max_tokens=max_tokens, sampler=make_sampler(0.0), verbose=False,
        )
        return [t[: t.index(END_TAG) + len(END_TAG)] if END_TAG in t else t for t in out.texts]

    # -- generation with traces (no second pass) --------------------------------------------
    def encode_prompt(self, image) -> tuple[mx.array, mx.array]:
        """Return (prompt_ids[1,P], prompt_embeds[1,P,D]) with vision features merged in."""
        inputs = prepare_inputs(
            self.processor, images=[image], prompts=self.prompt,
            image_token_index=self.model.config.image_token_index,
        )
        embeds = self.model.get_input_embeddings(inputs["input_ids"], inputs["pixel_values"]).inputs_embeds
        return inputs["input_ids"], embeds

    def _step(self, h: mx.array, caches, mask) -> tuple[dict[str, mx.array], mx.array]:
        """One language-model pass; returns tap outputs (incl. final norm) and next-token ids."""
        taps: dict[str, mx.array] = {}
        for i, (layer, c) in enumerate(zip(self.lm.layers, caches)):
            h = layer(h, mask, c)
            if i in self.taps:
                taps[f"layer_{i}"] = h
        h = self.lm.norm(h)
        taps["last_hidden_state"] = h
        next_ids = self.lm.lm_head(h[:, -1]).argmax(-1)
        return taps, next_ids

    def _prefill(self, prompts: Sequence[tuple[mx.array, mx.array]]):
        """Prefill a group of equal-length prompts into a fresh BatchKVCache set."""
        caches = [BatchKVCache([0] * len(prompts)) for _ in self.lm.layers]
        h = mx.concatenate([e for _, e in prompts], axis=0).astype(self.lm.norm.weight.dtype)
        taps, next_ids = self._step(h, caches, caches[0].make_mask(h.shape[1]))
        mx.async_eval(next_ids, *taps.values())
        return caches, taps, next_ids

    def generate_traces(self, images: Iterable[object], batch_size: int = GEN_BATCH_SIZE,
                        max_tokens: int = MAX_NEW_TOKENS) -> Iterator[tuple[int, dict]]:
        """Greedy-decode pages with continuous batching, recording hidden states at every position.

        Yields ``(index, trace)`` as pages finish (not in input order).  Each trace has the keys
        used by ``train.ipynb`` plus ``doctags`` and ``truncated``.

        Speed comes from: (1) rows share every decode step, so weights are read once per step
        for the whole batch; (2) finished rows are dropped and replaced by freshly prefilled
        pages, so the batch stays full instead of draining to one row; (3) tokens are read back
        one step behind the launch, so the GPU never waits on Python.  Pages are encoded lazily
        as they are admitted, so ``images`` can be the whole corpus.  A refill group must share
        one prompt length (no padding at prefill); sort inputs by image size to keep groups big.
        """
        it = iter(enumerate(images))
        pending: list[tuple[int, mx.array, mx.array]] = []  # encoded but not yet admitted
        stop = {END_TOKEN_ID, self.processor.tokenizer.eos_token_id, END_OF_UTTERANCE_ID}
        rows: list[dict] = []  # active rows: {"i", "ids", "taps": {k: [..]}, "comp": [..]}
        caches = None
        next_ids = None

        def admit():
            nonlocal caches, next_ids
            free = batch_size - len(rows)
            while len(pending) < free:
                nxt = next(it, None)
                if nxt is None:
                    break
                i, img = nxt
                ids, emb = self.encode_prompt(img)
                pending.append((i, ids, emb))
            if not pending or free <= 0:
                return
            P = pending[0][1].shape[1]
            group = [e for e in pending[:free] if e[1].shape[1] == P]
            for e in group:
                pending.remove(e)
            new_caches, taps, ids = self._prefill([(ids, emb) for _, ids, emb in group])
            for local, (i, pid, _) in enumerate(group):
                rows.append({"i": i, "ids": pid, "taps": {k: [v[local]] for k, v in taps.items()}, "comp": []})
            if caches is None:
                caches, next_ids = new_caches, ids
            else:
                for c, n in zip(caches, new_caches):
                    c.extend(n)
                next_ids = mx.concatenate([next_ids, ids])

        def retire(row: dict) -> dict:
            ids, comp = row["ids"], row["comp"]
            trace = {k: mx.concatenate(v, axis=0).astype(mx.float16) for k, v in row["taps"].items()}
            trace["token_ids"] = mx.concatenate([ids[0], mx.array(comp, dtype=ids.dtype)])
            trace["prompt_len"] = mx.array([ids.shape[1]])
            trace["completion_start"] = trace["prompt_len"]
            trace = finalize_trace(trace, ids.shape[1], self.keep_taps, _scalar)
            mx.eval(*trace.values())
            text = self.processor.tokenizer.decode(comp, skip_special_tokens=False)
            trace["doctags"] = text[: text.index(END_TAG) + len(END_TAG)] if END_TAG in text else text
            trace["truncated"] = comp[-1] not in stop
            return trace

        admit()
        while rows:
            # Launch the step that consumes the tokens chosen last step (stop tokens included, so
            # their hidden states are saved), then read those tokens back while the GPU works.
            taps, new_ids = self._step(self.lm.embed_tokens(next_ids[:, None]), caches, caches[0].make_mask(1))
            mx.async_eval(new_ids, *taps.values())
            toks = next_ids.tolist()
            keep = []
            for local, (row, tok) in enumerate(zip(rows, toks)):
                row["comp"].append(tok)
                for k, v in taps.items():
                    row["taps"][k].append(v[local])
                if tok in stop or len(row["comp"]) >= max_tokens:
                    yield row["i"], retire(row)
                else:
                    keep.append(local)
            if len(keep) < len(rows):
                rows = [rows[l] for l in keep]
                if rows:
                    for c in caches:
                        c.filter(keep)
                    new_ids = new_ids[mx.array(keep)]
                else:
                    caches = None
            next_ids = new_ids
            admit()

    # -- per-page encoding ---------------------------------------------------------------
    def encode(self, image, doctags: str) -> tuple[mx.array, mx.array, int]:
        """Return (token_ids[1,N], inputs_embeds[1,N,D], prompt_len)."""
        inputs = prepare_inputs(
            self.processor, images=[image], prompts=self.prompt,
            image_token_index=self.model.config.image_token_index,
        )
        completion = mx.array([self.processor.tokenizer.encode(doctags, add_special_tokens=False)])
        prompt_len = inputs["input_ids"].shape[1]
        ids = mx.concatenate((inputs["input_ids"], completion), axis=1)
        # Embed the prompt (with image features) and the completion separately: generated
        # DocTags can contain text that tokenizes to the <image> placeholder id, which would
        # otherwise break the image-feature scatter ("tokens do not match features").
        prompt_embeds = self.model.get_input_embeddings(
            inputs["input_ids"], inputs["pixel_values"]
        ).inputs_embeds
        completion_embeds = self.lm.embed_tokens(completion)
        embeds = mx.concatenate((prompt_embeds, completion_embeds.astype(prompt_embeds.dtype)), axis=1)
        return ids, embeds, prompt_len

    # -- batched teacher-forced forward ----------------------------------------------------
    def forward(self, inputs_embeds: mx.array) -> dict[str, mx.array]:
        """inputs_embeds: [B, N, D] right-padded.  Returns tap outputs + final normed state.

        Right padding + causal mask means valid positions never see pad, so batching is exact.
        """
        h = inputs_embeds.astype(self.lm.norm.weight.dtype)
        out: dict[str, mx.array] = {}
        for i, layer in enumerate(self.lm.layers):
            h = layer(h, mask="causal", cache=None)
            if i in self.taps:
                out[f"layer_{i}"] = h
        out["last_hidden_state"] = self.lm.norm(h)
        return out

    def extract(self, items: Iterable[tuple[object, str]], batch_size: int = 4) -> Iterable[dict[str, mx.array]]:
        """Yield one trace dict per (image, doctags) item; batches by length for speed."""
        encoded = [self.encode(img, dt) for img, dt in items]
        order = sorted(range(len(encoded)), key=lambda i: encoded[i][0].shape[1])
        pending: list[tuple[int, dict]] = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            lens = [encoded[i][0].shape[1] for i in idx]
            n = max(lens)
            batch = mx.concatenate(
                [mx.pad(encoded[i][1], ((0, 0), (0, n - l), (0, 0))) for i, l in zip(idx, lens)], axis=0
            )
            outs = self.forward(batch)
            traces = []
            for b, (i, l) in enumerate(zip(idx, lens)):
                ids, _, prompt_len = encoded[i]
                t = {k: v[b, :l].astype(mx.float16) for k, v in outs.items()}
                t["token_ids"] = ids[0]
                t["prompt_len"] = mx.array([prompt_len])
                t["completion_start"] = t["prompt_len"]  # alias kept for train.ipynb
                traces.append((i, finalize_trace(t, prompt_len, self.keep_taps, _scalar)))
            # async_eval lets the GPU run this batch while Python pads the next one.
            mx.async_eval(*[v for _, t in traces for v in t.values()])
            for done in pending:
                yield done
            pending = traces
        for done in pending:
            mx.eval(*done[1].values())
            yield done


    # -- persistence ---------------------------------------------------------------------
    def save(self, path: Path, trace: Trace) -> None:
        mx.save_safetensors(str(path), trace)
