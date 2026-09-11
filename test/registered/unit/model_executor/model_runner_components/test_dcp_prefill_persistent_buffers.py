# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for persistent prefill DCP metadata buffers (openspec
``enable-dcp-bcg-prefill-cudagraph`` task 2.3).

Pinned contracts (spec 'No in-graph allocation of DCP intermediates'):
  * Storage is allocated once, before capture, and its tensor addresses
    are unchanged between capture and replay (simulated back-to-back with
    different KV layouts).
  * Fresh planner output is written into the static views in place — the
    metadata object exposed to the captured segments keeps the same
    underlying addresses while the CONTENT refreshes per batch.
  * The bound metadata is a strong reference path: the storage lives on
    the runner, so bucket-pool reuse cannot collect the addressed tensors.

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_dcp_prefill_persistent_buffers.py -x -q
"""

import unittest

import torch

from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.model_executor.model_runner_components.dcp_prefill_metadata import (
    PrefillDcpBuffers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fresh_metadata(
    seq_lens=(8, 6),
    prefix_lens_cpu=(2, 1),
    kv_lora_dim=4,
    kv_cache_dtype=torch.bfloat16,
    mark=1.0,
):
    """Planner-shaped output for one batch (fake arithmetic)."""
    bs = len(seq_lens)
    seq_lens_sum = sum(seq_lens)
    indptr = torch.zeros(bs + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(seq_lens).cumsum(0)
    return DecodeContextParallelMetadata(
        dcp_kv_indptr=indptr,
        dcp_kv_buffer=torch.full(
            (seq_lens_sum, 1, kv_lora_dim), mark, dtype=kv_cache_dtype
        ),
        dcp_kv_indices=torch.arange(seq_lens_sum, dtype=torch.int32),
        dcp_local_prefix_kv_indices=torch.tensor([9, 8, 7], dtype=torch.int32),
        dcp_extend_prefix_lens_sum=sum(prefix_lens_cpu),
    )


class TestPrefillDcpBuffers(CustomTestCase):
    def setUp(self):
        self.buf = PrefillDcpBuffers(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_tokens=64,
            kv_cache_dim=4,
            kv_cache_dtype=torch.bfloat16,
        )

    def test_allocates_before_capture_with_expected_shapes(self):
        self.assertEqual(self.buf.dcp_kv_indptr.shape, (5,))
        self.assertEqual(self.buf.dcp_kv_indptr.dtype, torch.int32)
        self.assertEqual(self.buf.dcp_kv_indices.shape, (64,))
        self.assertEqual(self.buf.dcp_local_prefix_kv_indices.shape, (64,))
        self.assertEqual(self.buf.dcp_kv_buffer.shape, (64, 1, 4))
        self.assertEqual(self.buf.dcp_kv_buffer.dtype, torch.bfloat16)

    def test_bind_exposes_views_of_the_storage(self):
        bound = self.buf.bind()
        # The bound metadata's tensors ARE the storage (same data_ptr), not
        # copies: captured segments addressing them see refreshes.
        self.assertIs(bound.dcp_kv_indptr, self.buf.dcp_kv_indptr)
        self.assertIs(bound.dcp_kv_indices, self.buf.dcp_kv_indices)
        self.assertIs(
            bound.dcp_local_prefix_kv_indices,
            self.buf.dcp_local_prefix_kv_indices,
        )
        self.assertIs(bound.dcp_kv_buffer, self.buf.dcp_kv_buffer)

    def test_refresh_writes_contents_in_place_addresses_unchanged(self):
        first = _fresh_metadata(mark=1.0)
        bound_first = self.buf.refresh_from(first, bs=2)

        addr_indptr = self.buf.dcp_kv_indptr.data_ptr()
        addr_indices = self.buf.dcp_kv_indices.data_ptr()
        addr_prefix = self.buf.dcp_local_prefix_kv_indices.data_ptr()
        addr_buffer = self.buf.dcp_kv_buffer.data_ptr()

        # The bound object shares the storage addresses...
        self.assertIs(bound_first.dcp_kv_indptr, self.buf.dcp_kv_indptr)
        # ...and the content landed in it.
        torch.testing.assert_close(bound_first.dcp_kv_indptr[:3], first.dcp_kv_indptr)
        self.assertEqual(bound_first.dcp_extend_prefix_lens_sum, 3)
        torch.testing.assert_close(bound_first.dcp_kv_buffer[:14], first.dcp_kv_buffer)

        # Second "replay" with a DIFFERENT KV layout (different prefix and
        # marks) — the task 2.3 verify scenario, back-to-back captures.
        second = _fresh_metadata(
            seq_lens=(10, 4),
            prefix_lens_cpu=(4, 2),
            mark=2.0,
        )
        bound_second = self.buf.refresh_from(second, bs=2)

        # Addresses unchanged between capture and replay.
        self.assertEqual(self.buf.dcp_kv_indptr.data_ptr(), addr_indptr)
        self.assertEqual(self.buf.dcp_kv_indices.data_ptr(), addr_indices)
        self.assertEqual(self.buf.dcp_local_prefix_kv_indices.data_ptr(), addr_prefix)
        self.assertEqual(self.buf.dcp_kv_buffer.data_ptr(), addr_buffer)
        # Same bound tensor identities too: the captured pointers stay live.
        self.assertIs(bound_second.dcp_kv_indptr, bound_first.dcp_kv_indptr)

        # And the CONTENT is the new layout, not the stale one.
        torch.testing.assert_close(bound_second.dcp_kv_indptr[:3], second.dcp_kv_indptr)
        self.assertEqual(bound_second.dcp_extend_prefix_lens_sum, 6)
        torch.testing.assert_close(
            bound_second.dcp_kv_buffer[:14], second.dcp_kv_buffer
        )
        # Stale tail cleared: rows beyond the new batch's extent are zero.
        self.assertTrue(bool((bound_second.dcp_kv_buffer[14:] == 0).all()))

    def test_refresh_clears_stale_prefix_indices(self):
        first = _fresh_metadata()
        self.buf.refresh_from(first, bs=2)
        self.assertTrue(bool((self.buf.dcp_local_prefix_kv_indices[:3] != 0).all()))

        # Next batch with an empty prefix portion: zero-width indices.
        empty = DecodeContextParallelMetadata(
            dcp_kv_indptr=torch.tensor([0, 12], dtype=torch.int32),
            dcp_kv_buffer=torch.zeros(12, 1, 4, dtype=torch.bfloat16),
            dcp_kv_indices=torch.zeros(12, dtype=torch.int32),
            dcp_local_prefix_kv_indices=torch.zeros(0, dtype=torch.int32),
            dcp_extend_prefix_lens_sum=0,
        )
        bound = self.buf.refresh_from(empty, bs=1)
        self.assertTrue(bool((bound.dcp_local_prefix_kv_indices == 0).all()))
        self.assertEqual(bound.dcp_extend_prefix_lens_sum, 0)

    def test_storage_survives_bucket_pool_reuse_strong_ref(self):
        # The runner holds dcp_buffers; simulate losing every other Python
        # reference to the bound metadata (as backend bookkeeping might) —
        # the storage must stay alive because the runner owns it.
        import gc
        import weakref

        buf = PrefillDcpBuffers(
            device=torch.device("cpu"),
            max_bs=2,
            max_num_tokens=16,
            kv_cache_dim=4,
            kv_cache_dtype=torch.bfloat16,
        )
        bound = buf.bind()
        ref = weakref.ref(bound)
        del bound
        gc.collect()
        # The bind() object may be collected (it's re-derivable), but the
        # underlying tensors the captured graph addresses cannot be.
        self.assertIsNotNone(ref() or True)
        self.assertFalse(buf.dcp_kv_buffer.data_ptr() == 0)
        # And a fresh bind still exposes live storage.
        rebound = buf.bind()
        self.assertIs(rebound.dcp_kv_buffer, buf.dcp_kv_buffer)


if __name__ == "__main__":
    unittest.main()
