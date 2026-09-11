"""CPU parity of the ragged engram row map against the uniform verify path.

Compact ragged verify packs each request's verify tokens into per-request
runs of unequal length (planner budget trimming), while the uniform
target-verify path assumes one equal block per request. The ragged-aware
EngramHasher.forward derives the per-token request row from the layout's
qo_indptr (searchsorted, fixed shape) and hashes with MODE_EXTEND semantics.
These tests pin that path to a hand-rolled ground truth on CPU only -- no
triton, no CUDA -- so they run in the base-a CPU suite.
"""

import unittest

import torch

from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_EXTEND,
    MODE_VERIFY,
)
from sglang.srt.layers.engram import _torch_row_map_from_qo_indptr
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 997
N = 4
L = 2
H = 4
COLS = (N - 1) * H


class _FakeHasher:
    """Minimal stand-in exposing _torch_hash_ids with real tables."""

    def __init__(self, seed=0):
        from sglang.srt.layers.engram import EngramHasher

        g = torch.Generator().manual_seed(seed)
        self.max_ngram_size = N
        # A synthetic token map: identity over VOCAB is fine for parity.
        self.token_map = torch.arange(VOCAB, dtype=torch.int64)
        bound = (2**63 - 1) // VOCAB // 2
        self.multipliers = (
            torch.randint(0, bound, (L, N), generator=g, dtype=torch.int64) * 2 + 1
        )
        primes = torch.randint(
            10_000, 20_000, (L, N - 1, H), generator=g, dtype=torch.int64
        )
        self.primes = primes
        flat = primes.view(L, COLS)
        self.offsets = torch.cumsum(
            torch.cat([flat.new_zeros(L, 1), flat[:, :-1]], dim=1), dim=1
        )
        self.pad_id = 5
        self.image_token_id = None
        # Bind the unbound function so instance calls match the real class.
        self._torch_hash_ids = EngramHasher._torch_hash_ids.__get__(self)


def _packed_batch(lens, seed):
    """Tokens, positions, and a history table for a packed ragged batch.

    Row r's tokens occupy [indptr[r], indptr[r+1]); positions restart at zero
    per row so the n-gram windows behave like real verify tokens. The last
    history row is the spare pad row (EngramHasher.pad_row analogue)."""
    g = torch.Generator().manual_seed(seed)
    indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)))
    total = int(indptr[-1])
    ids = torch.randint(0, VOCAB, (total,), generator=g)
    pos = torch.zeros(total, dtype=torch.int64)
    for r in range(len(lens)):
        width = int(indptr[r + 1]) - int(indptr[r])
        if width:
            pos[int(indptr[r]) : int(indptr[r + 1])] = torch.arange(width)
    history = torch.randint(
        0, VOCAB, (len(lens) + 1, N - 1), generator=g, dtype=torch.int32
    )
    return indptr, ids, pos, history


class TestRaggedEngramRowMap(CustomTestCase):
    def test_row_map_handles_zero_and_ghost_rows(self):
        """Leading/middle zero-length rows are skipped by the bisect-right
        mapping; tokens past the last cumsum clamp into the last row (the
        capped layout's tail pad -- no consumer reads their hash)."""
        indptr = torch.tensor([0, 6, 6, 12, 15])
        row = _torch_row_map_from_qo_indptr(indptr, num_tokens=17, bs=4)
        self.assertEqual(row[:6].tolist(), [0] * 6)
        self.assertEqual(row[6:12].tolist(), [2] * 6)
        self.assertEqual(row[12:].tolist(), [3] * 5)

    def test_ragged_parity_vs_uniform_blocks(self):
        """Same packed tokens hashed two ways must agree token for token:
        ragged (row map + MODE_EXTEND) vs uniform equal blocks (MODE_VERIFY)."""
        hasher = _FakeHasher(seed=0)
        for lens in ([6, 6, 6, 6], [6, 5, 6, 5], [5, 0, 6, 7]):
            indptr, ids, pos, history = _packed_batch(
                lens, seed=abs(hash(tuple(lens))) % 2**31
            )

            row = _torch_row_map_from_qo_indptr(
                indptr, num_tokens=ids.numel(), bs=len(lens)
            )
            ragged_ids, ragged_toks = hasher._torch_hash_ids(
                ids,
                pos,
                kmode=MODE_EXTEND,
                history=history,
                num_real=ids.numel(),
                block=1,
                row=row,
                starts=indptr[:-1],
            )

            if len(set(lens)) == 1:
                # Uniform case: the pre-fix MODE_VERIFY interpretation is the
                # ground truth; both paths must produce identical tokens and
                # hash ids.
                uniform_ids, uniform_toks = hasher._torch_hash_ids(
                    ids,
                    pos,
                    kmode=MODE_VERIFY,
                    history=history,
                    num_real=ids.numel(),
                    block=lens[0],
                    row=None,
                    starts=None,
                )
                self.assertTrue(torch.equal(ragged_toks, uniform_toks))
                self.assertTrue(torch.equal(ragged_ids, uniform_ids))
            else:
                # Heterogeneous case (the shape compact mode really produces):
                # manual ground truth, one row at a time, each row hashing as a
                # standalone run seeded from its own history slot.
                for r in range(len(lens)):
                    lo, hi = int(indptr[r]), int(indptr[r + 1])
                    if hi == lo:
                        continue
                    seg_ids = ids[lo:hi]
                    want_ids, want_toks = hasher._torch_hash_ids(
                        seg_ids,
                        pos[lo:hi],
                        kmode=MODE_EXTEND,
                        history=history[r : r + 1],
                        num_real=seg_ids.numel(),
                        block=1,
                        row=torch.zeros(seg_ids.numel(), dtype=torch.int64),
                        starts=torch.zeros(seg_ids.numel(), dtype=torch.int64),
                    )
                    got_ids, got_toks = ragged_ids[lo:hi], ragged_toks[lo:hi]
                    self.assertTrue(
                        torch.equal(got_toks, want_toks),
                        f"lens={lens} row={r}: {got_toks} != {want_toks}",
                    )
                    self.assertTrue(
                        torch.equal(got_ids, want_ids),
                        f"lens={lens} row={r}: hash mismatch",
                    )


if __name__ == "__main__":
    unittest.main()
