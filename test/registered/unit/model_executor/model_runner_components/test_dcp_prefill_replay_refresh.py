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
"""Unit tests for in-place DCP metadata refresh on prefill graph replay
(openspec ``enable-dcp-bcg-prefill-cudagraph`` task 2.4).

Pinned contract (spec 'Consecutive replays with different KV layouts'):
two consecutive replays whose DCP KV ownership differs (different prefix
lengths) must leave the PERSISTENT buffers holding each replay's fresh
values — computed through the shared builder and written in place
(copy_/index writes, no reallocation) — so the captured segments read live
state every replay.

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_dcp_prefill_replay_refresh.py -x -q
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner_components.dcp_prefill_metadata import (
    PrefillDcpBuffers,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _planned(seq_lens, prefix_lens_cpu, mark, kv_dim=4):
    bs = len(seq_lens)
    seq_lens_sum = sum(seq_lens)
    indptr = torch.zeros(bs + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(seq_lens).cumsum(0)
    prefix_sum = sum(prefix_lens_cpu)
    return DecodeContextParallelMetadata(
        dcp_kv_indptr=indptr,
        dcp_kv_buffer=torch.full((seq_lens_sum, 1, kv_dim), mark, dtype=torch.bfloat16),
        dcp_kv_indices=torch.arange(seq_lens_sum, dtype=torch.int32),
        dcp_local_prefix_kv_indices=(
            torch.tensor([prefix_sum], dtype=torch.int32)
            if prefix_sum
            else torch.zeros(0, dtype=torch.int32)
        ),
        dcp_extend_prefix_lens_sum=prefix_sum,
    )


class _RefreshRecordingModel:
    """Records each prep call so consecutive replays are observable."""

    def __init__(self):
        self.calls = []

    def prepare_context_parallel_metadata_for_dcp(
        self,
        seq_lens,
        extend_prefix_lens,
        extend_prefix_lens_cpu,
        extend_seq_lens,
        req_pool_indices,
        req_to_token,
        seq_lens_sum,
        kv_buffer_shape,
        kv_cache_dtype,
        kv_cache_device,
        create_chunked_prefix_cache_kv_indices_fn,
    ):
        seq = [int(x) for x in seq_lens.tolist()]
        pre = list(extend_prefix_lens_cpu)
        self.calls.append((seq, pre))
        return _planned(seq, pre, mark=float(len(self.calls)))


def _make_runner():
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    model = _RefreshRecordingModel()
    dcp_buffers = PrefillDcpBuffers(
        device=torch.device("cpu"),
        max_bs=4,
        max_num_tokens=32,
        kv_cache_dim=4,
        kv_cache_dtype=torch.bfloat16,
    )
    req_to_token = torch.arange(24, dtype=torch.int64).reshape(4, 6)
    fake_backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            get_kv_buffer_shape=lambda: (
                torch.Size([100, 1, 4]),
                torch.Size([100, 1, 2]),
            )
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )
    runner.model_runner = SimpleNamespace(
        ps=SimpleNamespace(attn_dcp_size=2),
        model=model,
        kv_cache_dtype=torch.bfloat16,
        device="cpu",
        attn_backend=fake_backend,
    )
    runner.dcp_buffers = dcp_buffers
    return runner, model, dcp_buffers


def _serving_batch(seq_lens, prefix_lens):
    """A real serving-time extend batch as it reaches load_batch."""
    seq_lens = list(seq_lens)
    prefix_lens = list(prefix_lens)
    extend_lens = [s - p for s, p in zip(seq_lens, prefix_lens)]
    start_locs = [0]
    for length in extend_lens[:-1]:
        start_locs.append(start_locs[-1] + length)
    num_tokens = sum(extend_lens)
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int64),
        extend_seq_lens_cpu=extend_lens,
        req_pool_indices=torch.arange(len(seq_lens), dtype=torch.int64),
        seq_lens_sum=sum(seq_lens),
        extend_start_loc=torch.tensor(start_locs, dtype=torch.int64),
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        out_cache_loc=torch.zeros(num_tokens, dtype=torch.int64),
    )


class TestDcpReplayRefresh(CustomTestCase):
    def test_two_consecutive_replays_refresh_in_place(self):
        runner, model, dcp_buffers = _make_runner()
        ctx = SimpleNamespace()
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        fc = ForwardContext(attn_backend=runner.model_runner.attn_backend)
        # Track the persistent tensors' addresses across replays.
        addr = {
            name: getattr(dcp_buffers, name).data_ptr()
            for name in (
                "dcp_kv_indptr",
                "dcp_kv_indices",
                "dcp_local_prefix_kv_indices",
                "dcp_kv_buffer",
            )
        }

        # Replay 1: prefix layout (2, 1), seqs (8, 6).
        with forward_context(fc):
            refreshed1 = runner._prepare_capture_dcp_metadata(
                _serving_batch((8, 6), (2, 1))
            )
        self.assertIsNotNone(refreshed1)
        self.assertEqual(refreshed1.dcp_extend_prefix_lens_sum, 3)
        self.assertEqual(model.calls, [([8, 6], [2, 1])])
        torch.testing.assert_close(
            dcp_buffers.dcp_kv_indptr[:3],
            torch.tensor([0, 8, 14], dtype=torch.int32),
        )
        self.assertAlmostEqual(float(dcp_buffers.dcp_kv_buffer[0, 0, 0]), 1.0)

        # Replay 2: DIFFERENT prefix layout (4, 2), seqs (10, 8) — the KV
        # ownership indices change (different prefix lengths).
        with forward_context(fc):
            refreshed2 = runner._prepare_capture_dcp_metadata(
                _serving_batch((10, 8), (4, 2))
            )
        self.assertIsNotNone(refreshed2)
        self.assertEqual(refreshed2.dcp_extend_prefix_lens_sum, 6)
        self.assertEqual(model.calls, [([8, 6], [2, 1]), ([10, 8], [4, 2])])
        torch.testing.assert_close(
            dcp_buffers.dcp_kv_indptr[:3],
            torch.tensor([0, 10, 18], dtype=torch.int32),
        )
        self.assertAlmostEqual(float(dcp_buffers.dcp_kv_buffer[0, 0, 0]), 2.0)

        # In place: no reallocation between the two replays.
        for name, expected_addr in addr.items():
            self.assertEqual(
                getattr(dcp_buffers, name).data_ptr(),
                expected_addr,
                f"{name} was reallocated",
            )
        # And the exposed metadata objects address that same storage.
        self.assertIs(refreshed2.dcp_kv_buffer, dcp_buffers.dcp_kv_buffer)
        self.assertIs(refreshed2.dcp_kv_indptr, dcp_buffers.dcp_kv_indptr)

    def test_refresh_stale_tail_cleared_between_replays(self):
        runner, _, dcp_buffers = _make_runner()
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        fc = ForwardContext(attn_backend=runner.model_runner.attn_backend)
        with forward_context(fc):
            runner._prepare_capture_dcp_metadata(_serving_batch((12,), (4,)))
        self.assertTrue(bool((dcp_buffers.dcp_kv_buffer[:12] != 0).all()))

        # Smaller next batch: rows past its extent must not carry replay-1
        # values.
        with forward_context(fc):
            runner._prepare_capture_dcp_metadata(_serving_batch((6,), (2,)))
        self.assertTrue(bool((dcp_buffers.dcp_kv_buffer[:6] != 0).all()))
        self.assertTrue(bool((dcp_buffers.dcp_kv_buffer[6:] == 0).all()))

    def test_dcp_off_skips_refresh_entirely(self):
        runner, model, _ = _make_runner()
        runner.model_runner.ps.attn_dcp_size = 1
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        fc = ForwardContext(attn_backend=runner.model_runner.attn_backend)
        with forward_context(fc):
            result = runner._prepare_capture_dcp_metadata(
                _serving_batch((8, 6), (2, 1))
            )
        self.assertIsNone(result)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
