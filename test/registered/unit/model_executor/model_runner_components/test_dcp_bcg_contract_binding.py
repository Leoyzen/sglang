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
"""Unit tests for binding persistent DCP buffers into the BCG captured-
metadata contract (openspec ``enable-dcp-bcg-prefill-cudagraph`` task 3.1).

Pinned contracts:
  * During BCG capture under DCP, the batch the backend metadata planner
    reads carries the persistent ``PrefillDcpBuffers`` views — identity/
    data_ptr checks against ephemeral planner tensors.
  * An ephemeral ``attn_dcp_metadata`` that slips onto a batch is rebound:
    its contents are copied into the static storage and the batch exposes
    the static views instead.
  * The BCG replay contract call receives both batches with the persistent
    views.
  * DCP-off path is untouched: no rebinding happens and no metadata is
    attached.

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_dcp_bcg_contract_binding.py -x -q
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


def _planned(seq_lens=(8, 6), prefix_lens_cpu=(2, 1), mark=1.0, kv_dim=4):
    bs = len(seq_lens)
    seq_lens_sum = sum(seq_lens)
    indptr = torch.zeros(bs + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(seq_lens).cumsum(0)
    return DecodeContextParallelMetadata(
        dcp_kv_indptr=indptr,
        dcp_kv_buffer=torch.full((seq_lens_sum, 1, kv_dim), mark, dtype=torch.bfloat16),
        dcp_kv_indices=torch.arange(seq_lens_sum, dtype=torch.int32),
        dcp_local_prefix_kv_indices=torch.tensor([9], dtype=torch.int32),
        dcp_extend_prefix_lens_sum=sum(prefix_lens_cpu),
    )


class _RecordingContractBackend:
    """Attention-backend stub recording what the contract hands it."""

    def __init__(self):
        self.capture_batches = []
        self.replay_calls = []
        self.metadata_planned_from = []

    def init_forward_metadata(self, forward_batch):
        self.metadata_planned_from.append(forward_batch)

    def init_forward_metadata_for_breakable_cuda_graph_capture(self, forward_batch):
        self.capture_batches.append(forward_batch)
        return object()  # opaque per-bucket capture metadata

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self, metadata, forward_batch, *, static_forward_batch=None
    ):
        self.replay_calls.append((metadata, forward_batch, static_forward_batch))

    def prepare_prefill_shared_read_snapshot(self, forward_batch, *, num_qo_tokens):
        pass


def _make_runner(*, dcp_size=2, captured_contract=True, dcp_buffers=True):
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    backend = _RecordingContractBackend()
    runner.model_runner = SimpleNamespace(
        ps=SimpleNamespace(attn_dcp_size=dcp_size),
        attn_backend=backend,
    )
    runner.use_captured_attn_metadata = captured_contract
    runner.attn_metadata_buffers = {} if captured_contract else None
    runner._is_full_backend = False
    if dcp_buffers:
        runner.dcp_buffers = PrefillDcpBuffers(
            device=torch.device("cpu"),
            max_bs=4,
            max_num_tokens=32,
            kv_cache_dim=4,
            kv_cache_dtype=torch.bfloat16,
        )
    else:
        runner.dcp_buffers = None
    return runner, backend


def _batch(attn_dcp_metadata=None):
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        input_ids=torch.zeros(11, dtype=torch.int64),
        positions=torch.arange(11, dtype=torch.int64),
        out_cache_loc=torch.zeros(11, dtype=torch.int64),
        req_pool_indices=torch.zeros(2, dtype=torch.int64),
        seq_lens=torch.tensor([8, 6], dtype=torch.int64),
        seq_lens_sum=14,
        extend_seq_lens=torch.tensor([6, 5], dtype=torch.int64),
        extend_prefix_lens=torch.tensor([2, 1], dtype=torch.int64),
        extend_prefix_lens_cpu=[2, 1],
        extend_seq_lens_cpu=[6, 5],
        extend_start_loc=torch.tensor([0, 6], dtype=torch.int64),
    )
    fb.attn_dcp_metadata = attn_dcp_metadata
    return fb


def _fc(runner):
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    return forward_context(
        ForwardContext(attn_backend=runner.model_runner.attn_backend)
    )


class TestBcgContractBindsPersistentDcpViews(CustomTestCase):
    def test_capture_contract_receives_persistent_views(self):
        runner, backend = _make_runner()
        # Capture built the metadata through the shared builder earlier; it
        # carries EPHEMERAL planner tensors here. The contract hook must
        # rebind to the static views before backend planning.
        fb = _batch(_planned())
        ephemeral_ptrs = {
            name: getattr(fb.attn_dcp_metadata, name).data_ptr()
            for name in ("dcp_kv_indptr", "dcp_kv_buffer", "dcp_kv_indices")
        }
        buf = runner.dcp_buffers

        with _fc(runner):
            runner._init_forward_metadata_for_capture(fb, num_tokens=14)

        # Contract backends plan from the batch handed to the capture entry.
        planned_from = backend.capture_batches[-1]
        self.assertIs(planned_from, fb)
        bound = planned_from.attn_dcp_metadata
        # Views ARE the persistent storage (identity + pointer).
        self.assertIs(bound.dcp_kv_indptr, buf.dcp_kv_indptr)
        self.assertIs(bound.dcp_kv_buffer, buf.dcp_kv_buffer)
        self.assertIs(bound.dcp_kv_indices, buf.dcp_kv_indices)
        self.assertEqual(bound.dcp_kv_indptr.data_ptr(), buf.dcp_kv_indptr.data_ptr())
        for name, ptr in ephemeral_ptrs.items():
            self.assertNotEqual(
                getattr(bound, name).data_ptr(),
                ptr,
                f"{name} still addresses ephemeral storage",
            )
        # Contents were carried over into the static storage.
        torch.testing.assert_close(buf.dcp_kv_indptr[:3], _planned().dcp_kv_indptr)
        self.assertEqual(bound.dcp_extend_prefix_lens_sum, 3)

    def test_contract_replay_call_receives_persistent_views(self):
        runner, backend = _make_runner()
        buf = runner.dcp_buffers
        capture_meta = object()
        runner.attn_metadata_buffers[14] = capture_meta

        live = _batch(_planned(mark=1.0))
        static = _batch(_planned(mark=2.0))
        runner._prepare_forward_metadata_for_replay(live, static, num_tokens=14)

        metadata, r_live, r_static = backend.replay_calls[-1]
        self.assertIs(metadata, capture_meta)
        # Both batches pinned to the SAME persistent storage.
        self.assertIs(r_live.attn_dcp_metadata.dcp_kv_buffer, buf.dcp_kv_buffer)
        self.assertIs(r_static.attn_dcp_metadata.dcp_kv_buffer, buf.dcp_kv_buffer)
        self.assertIs(r_live.attn_dcp_metadata.dcp_kv_indptr, buf.dcp_kv_indptr)
        # The static batch's content won (last writer refreshes storage).
        torch.testing.assert_close(
            buf.dcp_kv_buffer[:14], _planned(mark=2.0).dcp_kv_buffer
        )

    def test_generic_eager_init_also_sees_persistent_views_under_bcg(self):
        # use_captured_attn_metadata=False (BCG with generic eager metadata
        # init): the planning batch still carries the persistent views.
        runner, backend = _make_runner(captured_contract=False)
        buf = runner.dcp_buffers
        fb = _batch(_planned())
        with _fc(runner):
            runner._prepare_forward_metadata_for_replay(fb, fb, num_tokens=14)
        planned_from = backend.metadata_planned_from[-1]
        self.assertIs(planned_from.attn_dcp_metadata.dcp_kv_buffer, buf.dcp_kv_buffer)

    def test_dcp_off_path_untouched(self):
        runner, backend = _make_runner(dcp_size=1)
        fb = _batch(None)  # DCP-off batches carry no DCP metadata at all
        with _fc(runner):
            runner._init_forward_metadata_for_capture(fb, num_tokens=14)
        planned_from = backend.capture_batches[-1]
        self.assertIsNone(planned_from.attn_dcp_metadata)
        # No metadata was conjured or rebound.
        self.assertIs(planned_from, fb)

    def test_no_persistent_buffers_passthrough(self):
        runner, backend = _make_runner(dcp_buffers=False)
        fb = _batch(_planned())
        original = fb.attn_dcp_metadata
        with _fc(runner):
            runner._init_forward_metadata_for_capture(fb, num_tokens=14)
        # Without storage there is nothing to rebind; the ephemeral object
        # passes through unchanged (capture would then fail the audit later).
        self.assertIs(backend.capture_batches[-1].attn_dcp_metadata, original)

    def test_identity_when_already_bound(self):
        runner, backend = _make_runner()
        buf = runner.dcp_buffers
        fb = _batch(buf.bind())  # already carries the static views
        with _fc(runner):
            runner._init_forward_metadata_for_capture(fb, num_tokens=14)
        bound = backend.capture_batches[-1].attn_dcp_metadata
        self.assertIs(bound.dcp_kv_indptr, buf.dcp_kv_indptr)
        self.assertIs(bound.dcp_kv_buffer, buf.dcp_kv_buffer)


if __name__ == "__main__":
    unittest.main()
