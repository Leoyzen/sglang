"""DCP localization of the MLA pool's retraction CPU backup/restore gathers.

Under ``dcp_size > 1`` the retraction path feeds ``req_to_token`` entries into
``MLATokenToKVPool.get_cpu_copy`` / ``load_cpu_copy``. Those entries live in the
DCP-widened virtual space, while this SHARDED pool addresses per-rank rows by
the physical id — the same owner rule the write kernels apply
(``loc % dcp == rank`` -> ``loc // dcp``). These tests pin the translation and
its round-trip on CPU tensors.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _pool(rows: int = 8) -> MLATokenToKVPool:
    pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
    pool.cpu_offloading_chunk_size = 64
    pool.layer_num = 1
    pool.size = rows
    pool.kv_buffer = [torch.arange(rows).reshape(rows, 1).float()]
    return pool


class TestLocalizeDcpIndices(CustomTestCase):
    def test_no_dcp_passes_indices_through(self):
        pool = _pool()
        indices = torch.tensor([0, 1, 2])
        with get_parallel().override(dcp_enabled=False):
            self.assertIs(pool._localize_dcp_indices(indices), indices)

    def test_owned_widened_ids_collapse_to_physical_rows(self):
        # dcp2 rank1 owns the odd residues; widened [1, 3, 5, 7] -> rows [0..3].
        pool = _pool(rows=4)
        with get_parallel().override(
            dcp_enabled=True, attn_dcp_size=2, attn_dcp_rank=1
        ):
            loc = pool._localize_dcp_indices(torch.tensor([1, 3, 5, 7]))
        self.assertEqual(loc.tolist(), [0, 1, 2, 3])

    def test_non_owned_ids_are_dropped(self):
        pool = _pool(rows=4)
        with get_parallel().override(
            dcp_enabled=True, attn_dcp_size=2, attn_dcp_rank=1
        ):
            loc = pool._localize_dcp_indices(torch.tensor([1, 0, 5, 2, 3]))
        # Only the rank-owned odd ids survive.
        self.assertEqual(loc.tolist(), [0, 2, 1])

    def test_round_trip_matches_row_content(self):
        """Backup then restore must reproduce the owned rows exactly."""
        pool = _pool(rows=4)
        widened = torch.tensor([1, 3, 5, 7])
        with get_parallel().override(
            dcp_enabled=True, attn_dcp_size=2, attn_dcp_rank=1
        ):
            backup = pool.get_cpu_copy(widened)
            pool.load_cpu_copy(backup, widened)
        # Each restored local row still holds its original arange value.
        self.assertTrue(
            torch.equal(
                pool.kv_buffer[0],
                torch.arange(4).reshape(4, 1).float(),
            )
        )


if __name__ == "__main__":
    unittest.main()
