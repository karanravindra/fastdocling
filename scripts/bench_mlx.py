"""End-to-end speculative decode of granite-docling on Apple Silicon, drafted against baseline.

    uv run --extra mlx python scripts/bench_mlx.py data/ood/images/finance --horizon 2

One page at a time, latency only -- the opposite job from extraction, which batches everything
it can.  Each page is decoded twice through the same loop, once with the drafter and once with
an empty proposal block, so both sides pay the same Python, cache and hidden-state overhead and
the ratio between them is honest.  Greedy verification means the drafted output is token-
identical to the baseline's, which the run checks rather than assumes.

**Read the acceptance, not the speedup.**  Break-even is set by what a round costs, and an
mlx-vlm round on a 258M model is an order of magnitude slower than vLLM's 1.35 ms -- which puts
break-even near ~1.05 tokens/step here against vLLM's 1.588 at k=1.  Nearly any acceptance wins
on this machine.  What transfers between backends is tokens/step and the per-slot rates; the
wall-clock number does not.

And what runs here is the drafter *as trained*, without the rotary embedding vLLM applies when
it serves the same weights (see ``draft_mlx``).  So a gap between this and the vLLM sweep in
train.ipynb is a measurement of that mismatch.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from tqdm import tqdm


def pooled(results):
    """(tokens, tok/s, tokens per target round) -- totals first, ratio second.

    An unweighted mean of per-page rates lets a short page outvote a long one: on this corpus a
    729-token page at 3.14 tokens/step and a 7,050-token page at 1.09 averaged to 2.07 against a
    pooled 1.20.
    """
    tokens = sum(len(r.tokens) for r in results)
    seconds = sum(r.decode_seconds for r in results)
    rounds = sum(r.rounds for r in results)
    return tokens, tokens / seconds, tokens / rounds


def slot_table(results, horizon: int) -> list[tuple[int, int, int]]:
    """(slot, rounds that reached it, rounds that accepted it), 1-based.

    Slot i is only reached when slot i-1 was accepted, so its denominator is the acceptances at
    i-1 -- not the total number of proposals.  A flat accepted/proposed hides that the later
    slots are judged on a much smaller population, and it is the later slots that decide whether
    raising the horizon pays, since break-even rises with it.
    """
    # Every slot up to the horizon, including ones nothing ever reached: a slot 1 that is
    # never accepted is the single most useful row in this table, and sizing the table by the
    # best round seen would omit it entirely.
    rows = []
    reached = sum(r.drafts for r in results)
    for i in range(1, horizon + 1):
        accepted = sum(sum(1 for n in r.accepted_hist if n >= i) for r in results)
        rows.append((i, reached, accepted))
        reached = accepted
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, help="directory of page PNGs (searched recursively)")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/eagle3_draft.pt"))
    ap.add_argument("--horizon", type=int, default=2,
                    help="num_speculative_tokens (default 2; k=1..2 is where break-even is lowest)")
    ap.add_argument("--window", type=int, default=None,
                    help="positions of history the drafter re-reads each round (default 128)")
    ap.add_argument("--limit", type=int, default=8, help="pages to decode (default 8)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()

    from transformers.image_utils import load_image

    from fastdocling.decode import FEATURE_KEYS_EAGLE3, get_decoder

    pages = sorted(a.images.rglob("*.png"))[: a.limit]
    if not pages:
        raise SystemExit(f"no PNGs under {a.images}")

    decoder = get_decoder("mlx")
    if decoder.name != "mlx":
        raise SystemExit(f"this benchmark is the MLX loop; got the {decoder.name!r} backend "
                         f"(run it on Apple Silicon with `uv sync --extra mlx`)")
    decoder.attach_eagle3_draft(a.checkpoint, horizon=a.horizon, window=a.window)

    base, spec, mismatched = [], [], []
    for page in tqdm(pages, unit="page", dynamic_ncols=True, leave=False):
        image = load_image(str(page))
        # The baseline is asked for the same taps as the drafted arm, though it has no drafter to
        # feed: the loop accumulates that history either way, and a 576-d concatenation per round
        # against a 1728-d one would quietly hand the baseline a discount the comparison is
        # supposed to exclude.
        b = decoder.generate(image, use_draft=False, feature_keys=FEATURE_KEYS_EAGLE3,
                             max_tokens=a.max_tokens)
        s = decoder.generate(image, use_draft=True, horizon=a.horizon,
                             feature_keys=FEATURE_KEYS_EAGLE3, max_tokens=a.max_tokens)
        # Greedy verification makes this an identity, not an approximation.  If it ever fails the
        # speedup is meaningless, because the two arms are no longer decoding the same thing.
        if b.tokens != s.tokens:
            mismatched.append(page.name)
        base.append(b)
        spec.append(s)

    tokens, base_tps, _ = pooled(base)
    _, spec_tps, per_step = pooled(spec)
    print(f"\n{len(pages)} pages, {tokens:,} tokens, horizon {a.horizon}")
    print(f"{'configuration':22s}{'tok/s (pooled)':>16s}{'tokens/step':>13s}{'vs plain':>10s}")
    print(f"{'plain mlx':22s}{base_tps:16.1f}{1.0:13.3f}{1.0:9.2f}x")
    print(f"{'eagle3 k=' + str(a.horizon):22s}{spec_tps:16.1f}{per_step:13.3f}"
          f"{spec_tps / base_tps:9.2f}x")

    proposed = sum(r.draft_tokens for r in spec)
    accepted = sum(r.accepted for r in spec)
    draft_s = sum(r.draft_seconds for r in spec)
    print(f"\ndrafted {sum(r.drafts for r in spec):,} rounds, proposed {proposed:,} tokens, "
          f"accepted {accepted:,} ({accepted / max(1, proposed):.1%})")
    print(f"{'slot':>6s}{'reached':>10s}{'accepted':>10s}{'rate':>8s}")
    for i, reached, acc in slot_table(spec, a.horizon):
        print(f"{i:>6d}{reached:>10,}{acc:>10,}{acc / max(1, reached):>8.1%}")

    # Graph construction only -- the proposal is lazy and its compute is fused into the round's
    # single eval, so this is the Python cost of drafting, not the drafter's cost.
    print(f"\ndraft call overhead {draft_s / max(1, sum(r.rounds for r in spec)) * 1e3:.3f} ms/round")
    per_page = [len(s.tokens) / s.rounds for s in spec]
    if len(per_page) > 1:
        print(f"per-page tokens/step spread: min {min(per_page):.3f}  "
              f"median {statistics.median(per_page):.3f}  max {max(per_page):.3f}")
    if mismatched:
        print(f"\nWARNING: drafted output differs from baseline on {len(mismatched)} page(s): "
              f"{', '.join(mismatched[:3])}{' ...' if len(mismatched) > 3 else ''}\n"
              f"Greedy speculative decoding must be token-identical; the numbers above are not "
              f"comparing like with like.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
