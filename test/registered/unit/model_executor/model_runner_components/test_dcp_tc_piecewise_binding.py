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
"""Unit tests pinning DCP metadata binding for BOTH prefill CG backends
(openspec ``enable-dcp-bcg-prefill-cudagraph`` task 3.3; spec 'DCP metadata
present at capture').

Finding recorded (opsx 3.3): on this base, BCG and tc_piecewise share ONE
PrefillCudaGraphRunner capture/replay path — ``capture_prepare`` binds real
DCP metadata via the shared builder BEFORE backend dispatch, and
``load_batch`` refreshes the persistent views before any replay. The
backends differ only in (a) the captured-metadata contract flag (BCG-only,
DSV4 opt-in) — the generic-eager branch still routes through the 3.1
rebinding — and (b) the FX/compile capture mechanics (TcPiecewiseCudaGraphBackend
warm+FX-capture inside ``capture_one`` vs BCG's segmented CUDAGraph). So
tc_piecewise needs no separate binding seam; these tests pin that
unification.

Pinned contracts:
  * tc_piecewise's generic-eager metadata-init branch receives the batch
    with real, persistent-view DCP metadata at capture.
  * The tc_piecewise metadata-init and the BCG contract entry address the
    same persistent-storage contract (each runner's own PrefillDcpBuffers).
  * tc_piecewise-style replay consumes a static batch whose
    attn_dcp_metadata is the refreshed persistent view with live values.

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_dcp_tc_piecewise_binding.py -x -q
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


def _planned(seq_lens=(8,), prefix_lens_cpu=(0,), mark=1.0, kv_dim=4):
    seq_lens_sum = sum(seq_lens)
    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(seq_lens).cumsum(0)
    return DecodeContextParallelMetadata(
        dcp_kv_indptr=indptr,
        dcp_kv_buffer=torch.full((seq_lens_sum, 1, kv_dim), mark, dtype=torch.bfloat16),
        dcp_kv_indices=torch.arange(seq_lens_sum, dtype=torch.int32),
        dcp_local_prefix_kv_indices=torch.zeros(0, dtype=torch.int32),
        dcp_extend_prefix_lens_sum=sum(prefix_lens_cpu),
    )


def _batch():
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.zeros(8, dtype=torch.int64),
        positions=torch.arange(8, dtype=torch.int64),
        out_cache_loc=torch.zeros(8, dtype=torch.int64),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int64),
        seq_lens_sum=8,
        extend_seq_lens=torch.tensor([8], dtype=torch.int64),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int64),
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[8],
        extend_start_loc=torch.tensor([0], dtype=torch.int64),
    )


class _RecordingBackend:
    """Stub for both backend shapes: generic eager init + contract entries."""

    def __init__(self):
        self.eager_init_batches = []
        self.contract_capture_batches = []
        self.contract_replays = []

    def init_forward_metadata(self, forward_batch):
        self.eager_init_batches.append(forward_batch)

    def init_forward_metadata_for_breakable_cuda_graph_capture(self, forward_batch):
        self.contract_capture_batches.append(forward_batch)
        return object()

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self, metadata, forward_batch, *, static_forward_batch=None
    ):
        self.contract_replays.append((forward_batch, static_forward_batch))

    def prepare_prefill_shared_read_snapshot(self, forward_batch, *, num_qo_tokens):
        pass


def _make_runner(*, captured_contract: bool):
    """Runner skeleton parameterized on the backend-specific flag.

    captured_contract=True models the BCG-with-contract configuration (the
    backend opted in); False models tc_piecewise / non-contract BCG (generic
    eager metadata init). The capture/replay plumbing under test is shared
    by both backends.
    """
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    backend = _RecordingBackend()
    runner.model_runner = SimpleNamespace(
        ps=SimpleNamespace(attn_dcp_size=2),
        attn_backend=backend,
        eager_runner=None,
    )
    runner.use_captured_attn_metadata = captured_contract
    runner.attn_metadata_buffers = {} if captured_contract else None
    runner._is_full_backend = False
    runner._dcp_replay_failure_logged = False
    runner.dcp_metadata_prep_failed = False
    runner.capture_hidden_mode = 0
    runner.dcp_buffers = PrefillDcpBuffers(
        device=torch.device("cpu"),
        max_bs=4,
        max_num_tokens=32,
        kv_cache_dim=4,
        kv_cache_dtype=torch.bfloat16,
    )
    return runner, backend


def _fc(runner):
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    return forward_context(
        ForwardContext(attn_backend=runner.model_runner.attn_backend)
    )


class TestTcPiecewiseDcpBinding(CustomTestCase):
    def test_generic_eager_branch_binds_persistent_views_at_capture(self):
        # tc_piecewise capture: use_captured_attn_metadata=False ->
        # init_forward_metadata(forward_batch) — the batch must carry the
        # persistent views when the metadata planner runs, so FX-traced
        # segments bind stable DCP addresses.
        runner, backend = _make_runner(captured_contract=False)
        fb = _batch()
        fb.attn_dcp_metadata = _planned()  # ephemeral from the dummy builder
        buf = runner.dcp_buffers

        with _fc(runner):
            runner._init_forward_metadata_for_capture(fb, num_tokens=8)

        self.assertEqual(len(backend.eager_init_batches), 1)
        planned_from = backend.eager_init_batches[0]
        bound = planned_from.attn_dcp_metadata
        # Real (non-null) DCP metadata present at capture for tc_piecewise.
        self.assertIsInstance(bound, DecodeContextParallelMetadata)
        self.assertIsNotNone(bound.dcp_kv_indices)
        # And it is the persistent storage, not the ephemeral allocation.
        self.assertIs(bound.dcp_kv_buffer, buf.dcp_kv_buffer)
        self.assertIs(bound.dcp_kv_indptr, buf.dcp_kv_indptr)

    def test_both_backends_bind_the_persistent_storage_contract(self):
        # BCG (contract) and tc_piecewise (generic) each end up addressing
        # their runner's PrefillDcpBuffers storage — one shared binding site
        # (capture_prepare + _init_forward_metadata_for_capture), two
        # backends. Same field set and dtype on both paths.
        runner_bcg, bcg_backend = _make_runner(captured_contract=True)
        runner_tc, tc_backend = _make_runner(captured_contract=False)

        fb = _batch()
        fb.attn_dcp_metadata = _planned()
        with _fc(runner_bcg):
            runner_bcg._init_forward_metadata_for_capture(fb, num_tokens=8)
        bound_bcg = bcg_backend.contract_capture_batches[0].attn_dcp_metadata

        fb2 = _batch()
        fb2.attn_dcp_metadata = _planned()
        with _fc(runner_tc):
            runner_tc._init_forward_metadata_for_capture(fb2, num_tokens=8)
        bound_tc = tc_backend.eager_init_batches[0].attn_dcp_metadata

        # Each bound to its own runner's storage...
        self.assertIs(bound_bcg.dcp_kv_buffer, runner_bcg.dcp_buffers.dcp_kv_buffer)
        self.assertIs(bound_tc.dcp_kv_buffer, runner_tc.dcp_buffers.dcp_kv_buffer)
        # ...with the identical persistent shape/dtype contract.
        self.assertEqual(bound_bcg.dcp_kv_buffer.dtype, torch.bfloat16)
        self.assertEqual(bound_tc.dcp_kv_buffer.dtype, torch.bfloat16)
        self.assertEqual(bound_bcg.dcp_kv_indptr.dtype, torch.int32)
        self.assertEqual(bound_tc.dcp_kv_indptr.dtype, torch.int32)

    def test_replay_path_hands_refreshed_static_batch_to_compiled_callable(self):
        # tc_piecewise replay: _execute_tc_piecewise passes
        # static_forward_batch into the compiled callable; after load_batch's
        # refresh its attn_dcp_metadata is the persistent view holding THIS
        # batch's values (live, not capture-time values).
        runner, _ = _make_runner(captured_contract=False)
        buf = runner.dcp_buffers

        static = _batch()
        static.attn_dcp_metadata = _planned(mark=7.0)
        with _fc(runner):
            runner._prepare_forward_metadata_for_replay(static, static, num_tokens=8)

        bound = static.attn_dcp_metadata
        self.assertIs(bound.dcp_kv_buffer, buf.dcp_kv_buffer)
        torch.testing.assert_close(
            buf.dcp_kv_buffer[:8], _planned(mark=7.0).dcp_kv_buffer
        )
        self.assertIs(buf.dcp_kv_buffer, runner.dcp_buffers.bind().dcp_kv_buffer)


if __name__ == "__main__":
    unittest.main()
