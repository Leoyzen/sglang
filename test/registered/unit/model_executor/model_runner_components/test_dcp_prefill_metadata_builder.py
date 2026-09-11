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
"""Unit tests for the shared prefill DCP metadata builder (openspec
``enable-dcp-bcg-prefill-cudagraph`` task 2.1).

The builder must reproduce the eager call site exactly: same guards
(DCP active + model implements the hook), same argument wiring, and
byte-identical metadata for the same batch. A mock model records the
arguments it receives so the test pins the wiring, and a fake planner
returns distinguishable tensors so equality is checked field-by-field.

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_dcp_prefill_metadata_builder.py -x -q
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.model_executor.model_runner_components.dcp_prefill_metadata import (
    prepare_dcp_extend_metadata,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakePlanner:
    """Stands in for layers.dcp.planner.prepare_decode_context_parallel_metadata."""

    last_kwargs = None

    @classmethod
    def plan(cls, **kwargs):
        cls.last_kwargs = kwargs
        bs = len(kwargs["seq_lens"])
        return DecodeContextParallelMetadata(
            dcp_kv_indptr=torch.arange(bs + 1, dtype=torch.int32) * 7,
            dcp_kv_buffer=torch.zeros(
                int(kwargs["seq_lens_sum"]), 1, 4, dtype=kwargs["kv_cache_dtype"]
            ),
            dcp_kv_indices=torch.full(
                (int(kwargs["seq_lens_sum"]),), 3, dtype=torch.int32
            ),
            dcp_local_prefix_kv_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            dcp_extend_prefix_lens_sum=11,
        )


def _fake_forward_context(model_runner):
    """The builder reads pools through the forward context; provide fakes."""
    from sglang.srt.model_executor.forward_context import ForwardContext

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
    ctx = ForwardContext(attn_backend=fake_backend)
    return req_to_token, ctx


def _make_model_runner(*, attn_dcp_size=2, has_hook=True):
    captured = {}

    def prepare_fn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakePlanner.plan(
            seq_lens=args[0],
            extend_prefix_lens=args[1],
            extend_prefix_lens_cpu=args[2],
            extend_seq_lens=args[3],
            req_pool_indices=args[4],
            req_to_token=args[5],
            seq_lens_sum=args[6],
            kv_buffer_shape=args[7],
            kv_cache_dtype=args[8],
            kv_cache_device=args[9],
            create_chunked_prefix_cache_kv_indices_fn=args[10],
        )

    hook = prepare_fn if has_hook else None
    model = SimpleNamespace()
    if has_hook:
        object.__setattr__(model, "prepare_context_parallel_metadata_for_dcp", hook)

    model_runner = SimpleNamespace(
        ps=SimpleNamespace(attn_dcp_size=attn_dcp_size),
        model=model,
        kv_cache_dtype=torch.bfloat16,
        device="cpu",
    )
    return model_runner, captured


def _make_forward_batch(*, seq_lens=(8, 6), prefix_lens=(2, 1)):

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    seq_lens = list(seq_lens)
    prefix_lens = list(prefix_lens)
    extend_lens = [s - p for s, p in zip(seq_lens, prefix_lens)]
    start_locs = [0]
    for length in extend_lens[:-1]:
        start_locs.append(start_locs[-1] + length)
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
        input_ids=torch.zeros(sum(extend_lens), dtype=torch.int64),
        positions=torch.zeros(sum(extend_lens), dtype=torch.int64),
        out_cache_loc=torch.zeros(sum(extend_lens), dtype=torch.int64),
    )


class TestPrepareDcpExtendMetadata(CustomTestCase):
    def test_dcp_disabled_is_a_noop(self):
        model_runner, captured = _make_model_runner(attn_dcp_size=1)
        fb = _make_forward_batch()
        _, ctx = _fake_forward_context(model_runner)
        from sglang.srt.model_executor.forward_context import forward_context

        with forward_context(ctx):
            result = prepare_dcp_extend_metadata(model_runner, fb)
        self.assertIsNone(result)
        self.assertIsNone(fb.attn_dcp_metadata)
        self.assertNotIn("args", captured)

    def test_model_without_hook_is_a_noop(self):
        model_runner, captured = _make_model_runner(has_hook=False)
        fb = _make_forward_batch()
        _, ctx = _fake_forward_context(model_runner)
        from sglang.srt.model_executor.forward_context import forward_context

        with forward_context(ctx):
            result = prepare_dcp_extend_metadata(model_runner, fb)
        self.assertIsNone(result)
        self.assertIsNone(fb.attn_dcp_metadata)

    def test_builder_wires_eager_call_site_arguments_exactly(self):
        model_runner, captured = _make_model_runner(attn_dcp_size=2)
        fb = _make_forward_batch(seq_lens=(8, 6), prefix_lens=(2, 1))
        req_to_token, ctx = _fake_forward_context(model_runner)
        from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
            create_chunked_prefix_cache_kv_indices,
        )
        from sglang.srt.model_executor.forward_context import forward_context

        with forward_context(ctx):
            result = prepare_dcp_extend_metadata(model_runner, fb)

        args = captured["args"]
        # Positional wiring pinned against the original eager call site
        # (seq_lens, extend_prefix_lens, extend_prefix_lens_cpu,
        # extend_seq_lens, req_pool_indices, req_to_token, seq_lens_sum,
        # kv_buffer_shape[0], kv_cache_dtype, device, indices-kernel fn).
        self.assertEqual(len(args), 11)
        torch.testing.assert_close(args[0], fb.seq_lens)
        torch.testing.assert_close(args[1], fb.extend_prefix_lens)
        self.assertEqual(args[2], fb.extend_prefix_lens_cpu)
        torch.testing.assert_close(args[3], fb.extend_seq_lens)
        torch.testing.assert_close(args[4], fb.req_pool_indices)
        torch.testing.assert_close(args[5], req_to_token)
        self.assertEqual(args[6], fb.seq_lens_sum)
        self.assertEqual(args[7], torch.Size([100, 1, 4]))
        self.assertEqual(args[8], torch.bfloat16)
        self.assertEqual(args[9], "cpu")
        self.assertIs(args[10], create_chunked_prefix_cache_kv_indices)

    def test_metadata_assigned_to_batch_and_returned_field_by_field(self):
        model_runner, _ = _make_model_runner(attn_dcp_size=2)
        fb = _make_forward_batch()
        _, ctx = _fake_forward_context(model_runner)
        from sglang.srt.model_executor.forward_context import forward_context

        with forward_context(ctx):
            result = prepare_dcp_extend_metadata(model_runner, fb)

        self.assertIsNotNone(result)
        self.assertIs(fb.attn_dcp_metadata, result)
        expected = _FakePlanner.last_kwargs
        # Field-by-field: the metadata the builder fills is the planner's
        # output, undistorted.
        self.assertEqual(result.dcp_extend_prefix_lens_sum, expected and 11)
        torch.testing.assert_close(
            result.dcp_kv_indptr, torch.tensor([0, 7, 14], dtype=torch.int32)
        )
        self.assertEqual(result.dcp_kv_buffer.shape, (14, 1, 4))
        self.assertEqual(result.dcp_kv_buffer.dtype, torch.bfloat16)
        torch.testing.assert_close(
            result.dcp_kv_indices, torch.full((14,), 3, dtype=torch.int32)
        )
        torch.testing.assert_close(
            result.dcp_local_prefix_kv_indices,
            torch.tensor([1, 2, 3], dtype=torch.int32),
        )

    def test_same_batch_produces_identical_metadata(self):
        # Task 2.1 verify: builder produces identical metadata for the same
        # batch.
        _, ctx = _fake_forward_context(None)

        from sglang.srt.model_executor.forward_context import forward_context

        with forward_context(ctx):
            model_runner_a, _ = _make_model_runner(attn_dcp_size=2)
            fb_a = _make_forward_batch()
            out_a = prepare_dcp_extend_metadata(model_runner_a, fb_a)

            model_runner_b, _ = _make_model_runner(attn_dcp_size=2)
            fb_b = _make_forward_batch()
            out_b = prepare_dcp_extend_metadata(model_runner_b, fb_b)

        self.assertEqual(
            out_a.dcp_extend_prefix_lens_sum, out_b.dcp_extend_prefix_lens_sum
        )
        torch.testing.assert_close(out_a.dcp_kv_indptr, out_b.dcp_kv_indptr)
        torch.testing.assert_close(out_a.dcp_kv_indices, out_b.dcp_kv_indices)
        torch.testing.assert_close(
            out_a.dcp_local_prefix_kv_indices, out_b.dcp_local_prefix_kv_indices
        )
        self.assertEqual(out_a.dcp_kv_buffer.shape, out_b.dcp_kv_buffer.shape)


if __name__ == "__main__":
    unittest.main()
