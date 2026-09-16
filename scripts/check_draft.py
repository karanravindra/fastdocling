"""Teacher-forced acceptance of the EAGLE-3 draft, on whatever machine holds the traces.

    uv run python scripts/check_draft.py data/traces_mac

This is the metric ``train.ipynb``'s ``acceptance()`` prints, extracted from the notebook so it
can run somewhere the notebook cannot.  It needs no decode loop and no vLLM -- just torch and a
directory of traces with EAGLE-3 taps -- which makes it the one end-to-end check that works on a
Mac today, against taps the Mac's own MLX target produced:

    uv run --extra mlx fastdocling-extract data/ood/images/finance data/traces_mac --keep-taps
    uv run --extra mlx python scripts/check_draft.py data/traces_mac

What a match proves is narrow but worth having: the checkpoint loads, the MLX backend taps the
same layers the pack was built from, and MLX's bf16 numerics do not move the number.  A *drop*
there is the drafter's problem; a drop only once vLLM serves it is a serving problem.

**This is an upper bound on what a served drafter accepts**, for two reasons that both push the
same way.  Every position is scored against the target's own state and the target's own previous
token, so it measures slot 1 of a speculative round and nothing else: at serve time slot 2 feeds
the drafter its *own* hidden state and its own proposed token, which training never simulated.
And labels outside the pruned vocabulary are masked out rather than counted as misses -- the
served head cannot propose them at all, so they are guaranteed rejections (the run prints how
many there were).

Slim checkpoints are handled: ``embed_tokens`` and ``lm_head`` are frozen copies of the target's
own tensors, so a checkpoint that omits them is rebuilt from the target by ``init_from_target``,
bit for bit.  A pruned checkpoint needs its vocabulary either way -- the head's rows are draft
ids, and mapping a label onto one takes the id array that ``output_vocab`` wrote.
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fastdocling.data import load_trace, scan_traces, source_counts
from fastdocling.eagle3 import AUX_LAYERS, VOCAB_SIZE, Eagle3Draft, init_from_target

# Windows match train.ipynb's CONTEXT_LENGTH, so the number here is comparable to the holdout
# accuracy the training cell prints.  It also bounds attention memory: a 7,000-token page in one
# causal forward materialises ~1.8 GB of scores on a backend that does not fuse them.
DEFAULT_WINDOW = 128
FROZEN = {"embed_tokens.weight", "lm_head.weight"}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_draft(checkpoint: Path, vocab_path: Path | None) -> tuple[Eagle3Draft, torch.Tensor | None, dict]:
    """``(draft, vocab or None, checkpoint metadata)``, rebuilding frozen tensors when omitted."""
    try:
        ckpt = torch.load(checkpoint, map_location="cpu")
    except pickle.UnpicklingError as e:
        # torch.load defaults to weights_only=True, which accepts tensors and plain Python
        # scalars and nothing else -- a numpy array saved alongside the state dict lands here.
        raise SystemExit(f"{checkpoint} holds something torch.load will not unpickle safely: {e}\n"
                         f"Save `vocab` as a torch tensor rather than a numpy array, or leave it "
                         f"out of the checkpoint and pass --vocab.") from e
    sd = {k: v.float() for k, v in ckpt["state_dict"].items()}

    raw = np.load(vocab_path) if vocab_path is not None else ckpt.get("vocab")
    vocab = None if raw is None else torch.as_tensor(np.asarray(raw), dtype=torch.long)
    if "lm_head.weight" in sd:
        draft_vocab = sd["lm_head.weight"].shape[0]
    elif vocab is not None:
        draft_vocab = len(vocab)
    else:
        raise SystemExit(
            f"{checkpoint} ships no lm_head, so its vocabulary size is unknown; pass --vocab "
            f"with the array output_vocab wrote (data/cache/output_vocab_<corpus key>.npy)")
    if draft_vocab != VOCAB_SIZE and vocab is None:
        raise SystemExit(
            f"{checkpoint} has a pruned head ({draft_vocab:,} of {VOCAB_SIZE:,} rows) but carries "
            f"no vocabulary, so a label cannot be mapped onto a draft row.  Pass --vocab with the "
            f"matching data/cache/output_vocab_<corpus key>.npy, or re-save the checkpoint with "
            f"the ids in it.")
    if vocab is not None and len(vocab) != draft_vocab:
        raise SystemExit(f"vocabulary has {len(vocab):,} ids but the head has {draft_vocab:,} rows "
                         f"-- they are from different runs")

    draft = Eagle3Draft(in_dim=576 * len(AUX_LAYERS), draft_vocab_size=draft_vocab)
    if FROZEN - set(sd):                       # slim checkpoint: take them from the target itself
        init_from_target(draft, vocab=vocab if draft_vocab != VOCAB_SIZE else None)
    missing, unexpected = draft.load_state_dict(sd, strict=False)
    if unexpected or (set(missing) - FROZEN):
        raise SystemExit(f"{checkpoint} does not match Eagle3Draft: missing "
                         f"{sorted(set(missing) - FROZEN)}, unexpected {sorted(unexpected)}")
    return draft.eval(), vocab, {k: v for k, v in ckpt.items() if k != "state_dict"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path, nargs="*", default=[Path("data/traces_taps")],
                    help="trace directories, extracted with --keep-taps (default: data/traces_taps)")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/eagle3_draft.pt"))
    ap.add_argument("--vocab", type=Path, default=None,
                    help="id array for a pruned head, if the checkpoint does not carry one")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"positions scored per forward, as in training (default {DEFAULT_WINDOW}; "
                         f"0 scores a whole page at once, which costs quadratic attention memory)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pages")
    ap.add_argument("--device", default="auto", help="auto (default) | cpu | mps | cuda")
    a = ap.parse_args()

    device = pick_device(a.device)
    draft, vocab, meta = load_draft(a.checkpoint, a.vocab)
    draft.to(device)

    # Labels are target token ids; the head's rows are draft ids.  -100 marks a label the pruned
    # head cannot propose, which is masked out rather than counted as a miss -- see the note above.
    to_draft = None
    if vocab is not None:
        to_draft = torch.full((VOCAB_SIZE,), -100, dtype=torch.long, device=device)
        to_draft[vocab.to(device)] = torch.arange(len(vocab), device=device)

    infos = scan_traces(a.traces, min_completion=3)   # 3 positions is the shortest scorable page
    no_taps = [i for i in infos if not i.has_taps]
    if no_taps:
        roots = sorted({i.source for i in no_taps})
        raise SystemExit(f"{len(no_taps)}/{len(infos)} traces have no EAGLE-3 taps (in "
                         f"{', '.join(roots)}); re-extract those roots with --keep-taps")
    infos = infos[: a.limit] if a.limit else infos

    hits = total = out_of_vocab = 0
    per_page: list[float] = []
    with torch.inference_mode():
        for info in tqdm(infos, unit="page", dynamic_ncols=True, leave=False):
            x, _, tok = load_trace(info, "eagle3")
            # At cursor i the drafter sees the target's state h_i and token i+1 (the bonus token
            # the target already emitted) and must predict token i+2 -- train.ipynb's alignment,
            # which is packed_batches(first_offset=1).
            x, inp, label = x[:-2].to(device), tok[1:-1].to(device), tok[2:].to(device)
            gold = to_draft[label] if to_draft is not None else label
            mask = gold >= 0
            page_hits = 0
            step = a.window or x.shape[0]
            for s in range(0, x.shape[0], step):
                pred = draft(x[s : s + step][None], inp[s : s + step][None]).argmax(-1)[0]
                m = mask[s : s + step]
                page_hits += int((pred[m] == gold[s : s + step][m]).sum())
            n = int(mask.sum())
            hits += page_hits
            total += n
            out_of_vocab += int(x.shape[0] - n)
            if n:
                per_page.append(page_hits / n)

    if not total:
        raise SystemExit("no scorable positions")
    p = hits / total
    print(f"{len(infos)} pages, {total:,} positions scored on {device.type}"
          + (f" ({source_counts(infos)})" if len({i.source for i in infos}) > 1 else ""))
    if out_of_vocab:
        print(f"{out_of_vocab:,} positions ({out_of_vocab / (total + out_of_vocab):.1%}) masked: "
              f"the pruned head cannot propose that token, so the figure below reads slightly high")
    print(f"\ntop-1 next-token accuracy p = {p:.3f}   (pooled over positions)")
    if len(per_page) > 1:
        # Spread only.  The pooled figure above is the metric: an unweighted mean over pages lets
        # a short page outvote a long one, which is how 1.20 tokens/step once read as 2.07.
        print(f"per-page spread: min {min(per_page):.3f}  median {statistics.median(per_page):.3f}  "
              f"max {max(per_page):.3f}")
    if "accuracy" in meta:
        print(f"checkpoint recorded {meta['accuracy']:.3f} on its own holdout")
    print(f"\nk=1 -> {1 + p:.3f} tokens/step (vLLM break-even 1.588)")
    print(f"k=2 -> at best {1 + p + p ** 2:.3f} tokens/step (break-even 1.726); the second slot "
          f"runs on the drafter's own state, which this does not measure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
