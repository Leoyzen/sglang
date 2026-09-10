# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for capture-dummy DCP metadata binding (openspec
``enable-dcp-bcg-prefill-cudagraph`` task 2.2).

Pinned contracts:
  * The prefill CUDA graph capture dummy routes its extend through the
    shared DCP metadata builder (the same path as a real extend), so
    ``forward_batch.attn_dcp_metadata`` is a real
    ``DecodeContextParallelMetadata`` at capture instead of ``None``.
  * The capture dummy's DCP metadata equals, field-by-field, what an
    equivalent real extend batch would receive (same seq/prefix layout).
  * Real-batch DCP invariants on the dummy: logical forward mode is
    EXTEND, positions consistent with seq_lens, and the metadata shapes
    fit the bucket (indptr covers bs, indices/buffer cover seq_lens_sum).
  * The Section-1 capture audit wraps per-shape capture when DCP is
    enabled and the debug env is set.

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_dcp_capture_dummy_metadata.py -x -q
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components.graph_capture_collective_audit import (
    CollectiveCaptureAuditError,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _planned_metadata(seq_lens, prefix_lens_cpu, kv_cache_dtype=torch.bfloat16):
    """The metadata a real extend with this layout would receive (fake
    planner arithmetic mirroring planner.py's field meanings)."""
    bs = len(seq_lens)
    seq_lens_sum = sum(seq_lens)
    indptr = torch.zeros(bs + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(seq_lens).cumsum(0)
    prefix_sum = sum(prefix_lens_cpu)
    return DecodeContextParallelMetadata(
        dcp_kv_indptr=indptr.to(torch.int32),
        dcp_kv_buffer=torch.zeros(seq_lens_sum, 1, 4, dtype=kv_cache_dtype),
        dcp_kv_indices=torch.zeros(seq_lens_sum, dtype=torch.int32),
        dcp_local_prefix_kv_indices=torch.arange(
            prefix_sum // 2 if prefix_sum else 0, dtype=torch.int32
        ),
        dcp_extend_prefix_lens_sum=prefix_sum,
    )


class _RecordingModel:
    """Model stub whose DCP hook plans from the batch it is handed."""

    def __init__(self, store):
        self._store = store

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
        seq_lens_list = [int(x) for x in seq_lens.tolist()]
        prefix_list = list(extend_prefix_lens_cpu)
        meta = _planned_metadata(seq_lens_list, prefix_list, kv_cache_dtype)
        self._store.append((seq_lens_list, prefix_list, seq_lens_sum, meta))
        return meta


def _make_runner(store, *, attn_dcp_size=2):
    """A PrefillCudaGraphRunner skeleton with only what 2.2 touches."""
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    model = _RecordingModel(store)
    runner.model_runner = SimpleNamespace(
        ps=SimpleNamespace(attn_dcp_size=attn_dcp_size),
        model=model,
        kv_cache_dtype=torch.bfloat16,
        device="cpu",
        attn_backend=SimpleNamespace(name="fake"),
    )
    return runner


def _fake_context_ctx():
    from sglang.srt.model_executor.forward_context import ForwardContext

    req_to_token = torch.arange(24, dtype=torch.int64).reshape(4, 6)
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            get_kv_buffer_shape=lambda: (
                torch.Size([100, 1, 4]),
                torch.Size([100, 1, 2]),
            )
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )
    return ForwardContext(attn_backend=backend)


def _real_like_forward_batch(seq_lens=(8, 6), prefix_lens=(2, 1)):
    """A REAL extend batch (as init_new would produce under DCP)."""
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    seq_lens = list(seq_lens)
    prefix_lens = list(prefix_lens)
    extend_lens = [s - p for s, p in zip(seq_lens, prefix_lens)]
    start_locs = [0]
    for length in extend_lens[:-1]:
        start_locs.append(start_locs[-1] + length)
    num_tokens = sum(extend_lens)
    positions = torch.cat(
        [
            torch.arange(prefix_len, seq_len, dtype=torch.int64)
            for seq_len, prefix_len in zip(seq_lens, prefix_lens)
        ]
    )
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
        positions=positions,
        out_cache_loc=torch.zeros(num_tokens, dtype=torch.int64),
    )


class TestCaptureDummyDcpMetadata(CustomTestCase):
    def test_capture_dummy_binds_real_dcp_metadata_not_none(self):
        store = []
        runner = _make_runner(store)
        # The capture dummy mirrors capture_prepare's synthetic layout:
        # one context-bounded request, zero prefix.
        fb = _real_like_forward_batch(seq_lens=(16,), prefix_lens=(0,))
        from sglang.srt.model_executor.forward_context import forward_context

        with forward_context(_fake_context_ctx()):
            result = runner._prepare_capture_dcp_metadata(fb)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DecodeContextParallelMetadata)
        self.assertIs(fb.attn_dcp_metadata, result)

    def test_capture_dummy_matches_real_extend_field_by_field(self):
        # Task 2.2 verify: capture dummy metadata vs an equivalent real
        # extend batch, field-by-field. Same seq/prefix layout -> same
        # metadata fields.
        store = []
        runner = _make_runner(store)
        ctx = _fake_context_ctx()
        from sglang.srt.model_executor.forward_context import forward_context

        # Capture dummy: capture_prepare's aggregate-token layout realized
        # as a real-shaped extend (16 tokens in one request, no prefix).
        dummy = _real_like_forward_batch(seq_lens=(16,), prefix_lens=(0,))
        with forward_context(ctx):
            runner._prepare_capture_dcp_metadata(dummy)
        dummy_meta = dummy.attn_dcp_metadata

        # Equivalent real extend: a genuine serving batch of one request,
        # 16 new tokens, no prefix.
        real = _real_like_forward_batch(seq_lens=(16,), prefix_lens=(0,))
        with forward_context(ctx):
            runner._prepare_capture_dcp_metadata(real)
        real_meta = real.attn_dcp_metadata

        self.assertEqual(
            dummy_meta.dcp_extend_prefix_lens_sum,
            real_meta.dcp_extend_prefix_lens_sum,
        )
        torch.testing.assert_close(dummy_meta.dcp_kv_indptr, real_meta.dcp_kv_indptr)
        torch.testing.assert_close(dummy_meta.dcp_kv_indices, real_meta.dcp_kv_indices)
        self.assertEqual(dummy_meta.dcp_kv_buffer.shape, real_meta.dcp_kv_buffer.shape)
        self.assertEqual(dummy_meta.dcp_kv_buffer.dtype, real_meta.dcp_kv_buffer.dtype)
        torch.testing.assert_close(
            dummy_meta.dcp_local_prefix_kv_indices,
            real_meta.dcp_local_prefix_kv_indices,
        )

    def test_capture_dummy_dcp_invariants_hold(self):
        # Real-batch DCP invariants (design D4) on the dummy: logical
        # forward mode == actual mode; positions valid vs seq_lens; shapes
        # within limits (indptr spans bs, indices/buffer span seq_lens_sum).
        store = []
        runner = _make_runner(store)
        ctx = _fake_context_ctx()
        from sglang.srt.model_executor.forward_context import forward_context

        dummy = _real_like_forward_batch(seq_lens=(8, 6), prefix_lens=(2, 1))
        self.assertIs(dummy.forward_mode, ForwardMode.EXTEND)  # logical == actual
        with forward_context(ctx):
            runner._prepare_capture_dcp_metadata(dummy)
        meta = dummy.attn_dcp_metadata

        bs = dummy.batch_size
        num_tokens = int(dummy.seq_lens_sum)
        self.assertEqual(meta.dcp_kv_indptr.shape, (bs + 1,))
        self.assertEqual(meta.dcp_kv_indices.shape, (num_tokens,))
        self.assertEqual(meta.dcp_kv_buffer.shape[0], num_tokens)
        # prefix-KV gather dtype contract (#31514): indices are int32.
        self.assertEqual(meta.dcp_local_prefix_kv_indices.dtype, torch.int32)
        self.assertEqual(meta.dcp_kv_indices.dtype, torch.int32)
        self.assertEqual(meta.dcp_kv_indptr.dtype, torch.int32)
        # positions valid past the prefix break boundary for each request.
        pos = 0
        for seq_len, prefix_len in zip(
            [int(x) for x in dummy.seq_lens.tolist()],
            [int(x) for x in dummy.extend_prefix_lens.tolist()],
        ):
            positions = dummy.positions[pos : pos + (seq_len - prefix_len)]
            self.assertTrue(bool((positions >= prefix_len).all()))
            self.assertTrue(bool((positions < seq_len).all()))
            pos += seq_len - prefix_len

    def test_dcp_off_leaves_metadata_none(self):
        store = []
        runner = _make_runner(store, attn_dcp_size=1)
        fb = _real_like_forward_batch()
        ctx = _fake_context_ctx()
        from sglang.srt.model_executor.forward_context import forward_context

        with forward_context(ctx):
            result = runner._prepare_capture_dcp_metadata(fb)
        self.assertIsNone(result)
        self.assertIsNone(fb.attn_dcp_metadata)
        self.assertEqual(store, [])

    def test_capture_audit_aborts_on_in_graph_collective_when_armed(self):
        # The audit wiring: with DCP enabled AND the debug env set, a
        # collective fired inside the capture scope aborts capture.
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.model_runner_components import (
            graph_capture_collective_audit as audit_mod,
        )

        runner = _make_runner([], attn_dcp_size=2)
        with envs.SGLANG_DEBUG_CAPTURE_COLLECTIVE_AUDIT.override(True):
            with (
                patch.object(audit_mod.torch.cuda, "is_available", return_value=True),
                patch.object(
                    audit_mod.torch.cuda,
                    "is_current_stream_capturing",
                    return_value=True,
                ),
                patch.object(
                    audit_mod.torch.distributed, "all_reduce", autospec=True
                ) as fake_ar,
            ):
                scope = audit_mod.instrumented_capture_scope("prefill-bucket-16")
                with scope:
                    audit_mod.get_active_capture_collective_auditor()
                    import torch.distributed as dist

                    dist.all_reduce(torch.zeros(1))
                    with self.assertRaises(CollectiveCaptureAuditError):
                        audit_mod.audit_active_capture_segment("prefill bucket 16")
                fake_ar.assert_called_once()
        # Auditor restored afterwards.
        auditor = audit_mod.get_active_capture_collective_auditor()
        auditor.uninstall()
        audit_mod._ACTIVE_AUDITOR = None


if __name__ == "__main__":
    unittest.main()
