"""The EAGLE-3 drafter's autoregressive proposal loop.

``forward`` is teacher-forced and ``propose`` is not, so they agree on exactly one thing: the
first proposal.  These tests pin that agreement, the KV cache the later proposals run on, and
the vocabulary mapping that stands between a draft row and a token the target understands --
the three places where a silent error would show up only as a lower acceptance rate, which is
indistinguishable from a drafter that simply learned less.

Pure torch, so this runs on the training box and on a Mac alike; the MLX port in ``draft_mlx``
mirrors it structurally and is covered by ``scripts/bench_mlx.py`` on hardware that has MLX.
"""

from __future__ import annotations

import unittest

import torch

from fastdocling.eagle3 import AUX_LAYERS, Eagle3Draft

IN_DIM = 576 * len(AUX_LAYERS)
VOCAB, DRAFT_VOCAB, WINDOW = 512, 64, 8


def a_draft(seed: int = 0) -> Eagle3Draft:
    torch.manual_seed(seed)
    return Eagle3Draft(in_dim=IN_DIM, vocab_size=VOCAB, draft_vocab_size=DRAFT_VOCAB).eval()


class TestPropose(unittest.TestCase):
    def setUp(self):
        self.draft = a_draft()
        torch.manual_seed(1)
        self.aux = torch.randn(WINDOW, IN_DIM)
        self.tokens = torch.randint(0, VOCAB, (WINDOW,))
        self.vocab = torch.randperm(VOCAB)[:DRAFT_VOCAB].sort().values

    def test_first_proposal_is_the_teacher_forced_argmax(self):
        """Slot 1 is drafted from the target's own state, so it is what ``forward`` predicts.

        This is the link between the number ``check_draft.py`` reports and what the drafter does
        when it runs for real: a teacher-forced accuracy is exactly the slot-1 acceptance rate,
        and nothing beyond that.
        """
        proposals = self.draft.propose(self.aux, self.tokens, horizon=4, vocab=self.vocab)
        want = self.vocab[self.draft(self.aux[None], self.tokens[None])[0, -1].argmax(-1)]
        self.assertEqual(int(proposals[0]), int(want))

    def test_cache_matches_reattending_over_the_whole_sequence(self):
        """The incremental step must equal one pass over window + the position just appended.

        A cache that drifts from the non-incremental computation is the classic speculative
        decoding bug: nothing raises, proposals just get worse the further into a round they are.
        """
        with torch.inference_mode():
            hidden = self.draft.fc(self.aux[None])
            embeds = self.draft.embed_tokens(self.tokens[None])
            x1, past = self.draft.layer.attend(embeds, hidden)
            nxt = self.vocab[self.draft.lm_head(self.draft.norm(x1[:, -1:])).argmax(-1)]

            stepped, _ = self.draft.layer.attend(self.draft.embed_tokens(nxt), x1[:, -1:], past)
            whole, _ = self.draft.layer.attend(
                torch.cat([embeds, self.draft.embed_tokens(nxt)], dim=1),
                torch.cat([hidden, x1[:, -1:]], dim=1))
        self.assertLess((whole[:, -1] - stepped[:, 0]).abs().max().item(), 1e-4)

    def test_proposals_are_target_ids_the_head_can_reach(self):
        """A pruned head spans ``vocab``; every id it proposes has to come back through it."""
        proposals = self.draft.propose(self.aux, self.tokens, horizon=5, vocab=self.vocab)
        self.assertEqual(proposals.shape, (5,))
        self.assertTrue(bool(torch.isin(proposals, self.vocab).all()))

    def test_unpruned_head_proposes_raw_draft_ids(self):
        torch.manual_seed(2)
        draft = Eagle3Draft(in_dim=IN_DIM, vocab_size=VOCAB, draft_vocab_size=VOCAB).eval()
        proposals = draft.propose(self.aux, self.tokens, horizon=3)
        self.assertEqual(proposals.shape, (3,))
        self.assertTrue(bool(((proposals >= 0) & (proposals < VOCAB)).all()))

    def test_forward_is_unchanged_by_the_cache_path(self):
        """``forward`` now routes through ``attend``; the training-time function must not move."""
        out = self.draft(self.aux[None], self.tokens[None])
        direct, _ = self.draft.layer.attend(self.draft.embed_tokens(self.tokens[None]),
                                            self.draft.fc(self.aux[None]))
        expected = self.draft.lm_head(self.draft.norm(direct))
        self.assertLess((out - expected).abs().max().item(), 1e-5)
        self.assertEqual(out.shape, (1, WINDOW, DRAFT_VOCAB))


if __name__ == "__main__":
    unittest.main()
