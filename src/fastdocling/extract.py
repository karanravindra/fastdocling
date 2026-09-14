"""Extract EAGLE-3 training traces from granite-docling in one teacher-forced pass.

For every page we run a *single* prefill over ``[image + prompt + gold DocTags]`` with a
causal mask and record the residual stream after three layers (low / mid / high, the
EAGLE-3 recipe) plus the final normalized hidden state.  No decoding happens here; the
DocTags labels are an input, so extraction costs one forward pass per page.

Labels are the model's *own* greedy output (so the draft learns the target's
distribution, as in EAGLE).  In the default image-directory mode the hidden states are
recorded *during* batched greedy decoding, so each saved state is exactly the one that
produced the next token and no second pass is needed.  Labels are cached as ``.dt``.

Usage:
    fastdocling-extract data/images data/traces          # generate DocTags + traces for every PNG
    fastdocling-extract manifest.jsonl data/traces       # use provided labels instead
    # manifest lines: {"image": "data/images/x/0001.png", "doctags": "<doctag>...</doctag>"}
    # or               {"image": ..., "doctags_path": "labels/x/0001.dt"}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx
from mlx_vlm import batch_generate, load, stream_generate
from mlx_vlm.models.cache import BatchKVCache
from mlx_vlm.sample_utils import make_sampler
from mlx_vlm.generate import prepare_inputs
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from tqdm import tqdm
from transformers.image_utils import load_image

MODEL_ID = "ibm-granite/granite-docling-258M-mlx"
PROMPT = "Convert this page to docling."
MAX_NEW_TOKENS = 8192
GEN_BATCH_SIZE = 8
END_TAG = "</doctag>"
END_TOKEN_ID = 100328  # "</doctag>"
END_OF_UTTERANCE_ID = 100352


def eagle3_taps(num_layers: int) -> tuple[int, int, int]:
    """Layer indices whose *outputs* EAGLE-3 fuses (low, mid, high).

    Mirrors the reference data script, which takes ``hidden_states[3]``,
    ``hidden_states[len // 2]`` and ``hidden_states[-3]`` from HF's tuple whose entry 0 is
    the embeddings.  For 30 layers this is (2, 14, 27).
    """
    n = num_layers + 1
    return 2, n // 2 - 1, num_layers - 3


class TraceExtractor:
    def __init__(self, model_id: str = MODEL_ID, prompt: str = PROMPT, taps: Sequence[int] | None = None,
                 *, model=None, processor=None, config=None):
        """Pass ``model``/``processor``/``config`` to reuse an already-loaded model."""
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
                return text[: text.index(END_TAG) + len(END_TAG)]
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

    def generate_traces(self, images: Sequence[object], batch_size: int = GEN_BATCH_SIZE,
                        max_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
        """Greedy-decode pages with continuous batching, recording hidden states at every position.

        Speed comes from three things: (1) rows share every decode step, so the model's weights
        are read once per step for the whole batch; (2) finished rows are dropped and replaced
        by freshly prefilled pages, so the batch stays full instead of draining to one row;
        (3) tokens are read back one step behind the launch, so the GPU never waits on Python.
        Prompts are grouped by length so a group prefills with no padding; BatchKVCache keeps
        per-row offsets (and RoPE positions) once groups of different lengths share a batch.

        Each result has the trace keys used by ``train.ipynb`` plus ``doctags`` and ``truncated``.
        """
        encoded = [self.encode_prompt(img) for img in images]
        # queue grouped by prompt length, so every refill group is padding-free
        order = sorted(range(len(encoded)), key=lambda i: encoded[i][0].shape[1])
        queue = list(order)
        stop = {END_TOKEN_ID, self.processor.tokenizer.eos_token_id, END_OF_UTTERANCE_ID}

        results: list[dict | None] = [None] * len(images)
        rows: list[dict] = []  # active rows: {"i", "taps": {k: [..]}, "comp": [..]}
        caches = None
        next_ids = None

        def admit():
            nonlocal caches, next_ids
            free = batch_size - len(rows)
            if free <= 0 or not queue:
                return
            P = encoded[queue[0]][0].shape[1]
            group = []
            while queue and len(group) < free and encoded[queue[0]][0].shape[1] == P:
                group.append(queue.pop(0))
            new_caches, taps, ids = self._prefill([encoded[i] for i in group])
            for local, i in enumerate(group):
                rows.append({"i": i, "taps": {k: [v[local]] for k, v in taps.items()}, "comp": []})
            if caches is None:
                caches, next_ids = new_caches, ids
            else:
                for c, n in zip(caches, new_caches):
                    c.extend(n)
                next_ids = mx.concatenate([next_ids, ids])

        def retire(row: dict):
            i, comp = row["i"], row["comp"]
            ids = encoded[i][0]
            trace = {k: mx.concatenate(v, axis=0).astype(mx.float16) for k, v in row["taps"].items()}
            trace["token_ids"] = mx.concatenate([ids[0], mx.array(comp, dtype=ids.dtype)])
            trace["prompt_len"] = mx.array([ids.shape[1]])
            trace["completion_start"] = trace["prompt_len"]
            mx.eval(*trace.values())
            text = self.processor.tokenizer.decode(comp, skip_special_tokens=False)
            trace["doctags"] = text[: text.index(END_TAG) + len(END_TAG)] if END_TAG in text else text
            trace["truncated"] = comp[-1] not in stop
            results[i] = trace
            encoded[i] = None  # free the prompt embeddings

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
                    retire(row)
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
        return results  # type: ignore[return-value]

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
                traces.append((i, t))
            # async_eval lets the GPU run this batch while Python pads the next one.
            mx.async_eval(*[v for _, t in traces for v in t.values()])
            for done in pending:
                yield done
            pending = traces
        for done in pending:
            mx.eval(*done[1].values())
            yield done


def _read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if "doctags" not in r:
            r["doctags"] = Path(r["doctags_path"]).read_text()
    return rows


def _trace_name(image: Path, root: Path) -> str:
    rel = image.relative_to(root) if image.is_relative_to(root) else Path(image.name)
    return "__".join(rel.with_suffix("").parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="directory of page PNGs, or a .jsonl manifest with labels")
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=GEN_BATCH_SIZE, help="pages decoded together (1 = sequential)")
    ap.add_argument("--chunk", type=int, default=16, help="pages per prefill sweep (manifest mode)")
    ap.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--keep-truncated", action="store_true", help="keep pages that never emitted </doctag>")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    ex = TraceExtractor(args.model)

    if args.source.suffix == ".jsonl":
        rows = _read_manifest(args.source)
        for r in rows:
            r["name"] = Path(r["image"]).stem
    else:
        rows = [{"image": str(p), "name": _trace_name(p, args.source)}
                for p in sorted(args.source.rglob("*.png"))]
    total = len(rows)
    rows = [r for r in rows if not (args.out / f"{r['name']}.safetensors").exists()]
    skipped = total - len(rows)
    print(f"taps={ex.taps} pages={len(rows)} (skipping {skipped} already extracted)", file=sys.stderr)

    # Generation dominates runtime, so the bar advances per page once its label is ready;
    # the batched prefill at the end of each chunk is a fraction of a second per page.
    bar = tqdm(total=len(rows), unit="page", dynamic_ncols=True, smoothing=0.05)
    truncated = 0
    if args.source.suffix == ".jsonl":
        # labels supplied: one teacher-forced prefill per page
        for c0 in range(0, len(rows), args.chunk):
            chunk = rows[c0 : c0 + args.chunk]
            items = [(load_image(r["image"]), r["doctags"]) for r in chunk]
            for i, trace in ex.extract(items, args.batch_size):
                mx.save_safetensors(str(args.out / f"{chunk[i]['name']}.safetensors"), trace)
                bar.update()
    else:
        # no labels: greedy-decode in batches and record the taps during decoding itself
        for c0 in range(0, len(rows), args.gen_batch):
            chunk = rows[c0 : c0 + args.gen_batch]
            bar.set_postfix_str(f"gen x{len(chunk)} {chunk[0]['name'][-30:]}", refresh=True)
            traces = ex.generate_traces([load_image(r["image"]) for r in chunk], args.gen_batch, args.max_tokens)
            for r, tr in zip(chunk, traces):
                (args.out / f"{r['name']}.dt").write_text(tr.pop("doctags"))
                if tr.pop("truncated") and not args.keep_truncated:
                    truncated += 1
                    continue
                mx.save_safetensors(str(args.out / f"{r['name']}.safetensors"), tr)
            bar.update(len(chunk))
    bar.close()
    if truncated:
        print(f"{truncated} pages skipped: no {END_TAG} within --max-tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
