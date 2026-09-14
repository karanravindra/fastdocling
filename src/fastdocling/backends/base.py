"""Shared contract for trace-extraction backends.

A backend turns page images into *traces*: one file per page holding the target model's
residual stream at every generated position, which ``fastdocling.data`` cuts into training
windows.  Two backends exist because the target runs on different silicon:

    mlx    Apple Silicon, via mlx-vlm              (uv sync --extra mlx)
    vllm   CUDA, via vLLM's extract_hidden_states  (uv sync --extra cuda)

Both write the same on-disk format, so traces from either are interchangeable and the
Hub dataset can hold a mix.  The trace keys are:

    token_ids[N]              int   every token, prompt included
    prompt_len[1]             int   where the DocTags completion starts
    completion_start[1]       int   alias of prompt_len, kept for train.ipynb
    state_offset[1]           int   position of the first *stored* hidden state
    last_hidden_state[M,576]  f16   post-final-norm state, M = N - state_offset
    layer_2/14/27[M,576]      f16   EAGLE-3 taps, only with keep_taps

Arrays stay in each backend's native framework (mx.array / torch.Tensor) until the
backend's own ``save`` writes them; the shared helpers here only slice and index, which
behaves identically across frameworks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, runtime_checkable

PROMPT = "Convert this page to docling."
MAX_NEW_TOKENS = 8192
GEN_BATCH_SIZE = 16
END_TAG = "</doctag>"
END_TOKEN_ID = 100328        # "</doctag>"
END_OF_UTTERANCE_ID = 100352
HIDDEN_SIZE = 576

Trace = dict[str, Any]


def eagle3_taps(num_layers: int) -> tuple[int, int, int]:
    """Layer indices whose *outputs* EAGLE-3 fuses (low, mid, high).

    Mirrors the reference data script, which takes ``hidden_states[3]``,
    ``hidden_states[len // 2]`` and ``hidden_states[-3]`` from HF's tuple whose entry 0 is
    the embeddings.  For 30 layers this is (2, 14, 27).

    These are indices into the decoder's own layer list.  Backends that speak the HF
    tuple convention (vLLM does) must add 1; see ``vllm_backend.aux_layer_ids``.
    """
    n = num_layers + 1
    return 2, n // 2 - 1, num_layers - 3


def finalize_trace(trace: Trace, prompt_len: int, keep_taps: bool, scalar: Callable[[int], Any]) -> Trace:
    """Shrink a trace losslessly for training.

    - States before ``prompt_len - 1`` belong to image/prompt tokens the loader never reads;
      they are dropped and ``state_offset`` records where the stored states start.
    - The EAGLE-3 taps (``layer_*``) are dropped unless ``keep_taps``; the final normed state is
      what the current draft trains on.  ``token_ids`` stays complete (int64, tiny).

    ``scalar`` builds a 1-element array in the caller's framework, the only operation here
    that is not plain slicing.
    """
    off = prompt_len - 1
    out: Trace = {}
    for k, v in trace.items():
        if k.startswith("layer_"):
            if keep_taps:
                out[k] = v[off:]
        elif k == "last_hidden_state":
            out[k] = v[off:]
        else:
            out[k] = v
    out["state_offset"] = scalar(off)
    return out


def trace_name(image: Path, root: Path) -> str:
    """Flat, collision-free filename for a page: ``ml_papers__<doc>__0001``."""
    rel = image.relative_to(root) if image.is_relative_to(root) else Path(image.name)
    return "__".join(rel.with_suffix("").parts)


def clip_to_end_tag(text: str) -> str:
    """Trim decoded text at the DocTags terminator; return it whole if never emitted."""
    return text[: text.index(END_TAG) + len(END_TAG)] if END_TAG in text else text


@runtime_checkable
class Backend(Protocol):
    """What ``fastdocling.extract`` needs from a target-model backend."""

    taps: tuple[int, ...]
    keep_taps: bool

    needs_length_sorted_input: bool
    """Whether the caller should sort pages by image size before handing them over.

    MLX prefills a refill group in one shot with no padding, so a group must share a prompt
    length and sorting keeps groups full.  vLLM schedules each request independently and
    gains nothing, so sorting there is a wasted pass over the corpus.
    """

    def generate_traces(
        self, images: Iterable[object], batch_size: int = GEN_BATCH_SIZE, max_tokens: int = MAX_NEW_TOKENS
    ) -> Iterator[tuple[int, Trace]]:
        """Greedy-decode pages and record hidden states in the same pass.

        Yields ``(index, trace)`` as pages finish, *not* in input order.  Each trace carries
        the extra keys ``doctags`` (str) and ``truncated`` (bool) for the caller to pop.
        """
        ...

    def extract(self, items: Iterable[tuple[object, str]], batch_size: int = 4) -> Iterator[tuple[int, Trace]]:
        """Teacher-forced pass over ``(image, gold doctags)`` pairs; yields ``(index, trace)``."""
        ...

    def save(self, path: Path, trace: Trace) -> None:
        """Write one finalized trace as safetensors."""
        ...
