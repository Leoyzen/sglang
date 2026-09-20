# SPDX-License-Identifier: Apache-2.0
"""Hopper FP4 indexer scores across visible-length and graph replay boundaries.

The paged cases compare ``fp4_index_logits_decode_paged`` against the
materialized-slot-map path it replaces.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    fp4_index_logits_decode as fp4_index_logits_decode_materialized,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    fp4_index_logits_decode_paged,
)
from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
    fp4_index_logits_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64
HEAD_DIM = 128


def _make_inputs(num_heads, capacity, visible_lengths):
    torch.manual_seed(42 + num_heads + capacity)
    batch_size = len(visible_lengths)
    num_pages = max(1, (capacity + PAGE_SIZE - 1) // PAGE_SIZE)
    fp4_values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.bfloat16,
        device="cuda",
    )
    codes = torch.randint(0, 16, (num_pages * PAGE_SIZE, HEAD_DIM), device="cuda")
    scale_exponents = torch.randint(
        125, 129, (num_pages * PAGE_SIZE, HEAD_DIM // 32), device="cuda"
    )
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    table = torch.cat(
        [
            payload.reshape(num_pages, PAGE_SIZE * 64),
            scale_exponents.to(torch.uint8).reshape(num_pages, PAGE_SIZE * 4),
        ],
        dim=1,
    )
    keys = (
        fp4_values[codes].float()
        * torch.exp2(scale_exponents.float() - 127).repeat_interleave(32, dim=1)
    ).to(torch.bfloat16)
    query_values = torch.tensor([-0.5, 0, 0.5], dtype=torch.bfloat16, device="cuda")
    q = query_values[
        torch.randint(0, 3, (batch_size, num_heads, HEAD_DIM), device="cuda")
    ]
    weight_values = torch.tensor(
        [-0.5, 0.25, 0.5, 1], dtype=torch.bfloat16, device="cuda"
    )
    weights = weight_values[
        torch.randint(0, len(weight_values), (batch_size, num_heads), device="cuda")
    ]
    logical_positions = torch.arange(capacity, device="cuda")
    page_orders = torch.stack(
        [
            torch.arange(num_pages - 1, -1, -1, device="cuda").roll(row)
            for row in range(batch_size)
        ]
    )
    slots = (
        page_orders[:, logical_positions // PAGE_SIZE] * PAGE_SIZE
        + logical_positions % PAGE_SIZE
    )
    lens = torch.tensor(visible_lengths, dtype=torch.int64, device="cuda")
    return q, weights, slots, lens, table, keys


def _reference(q, weights, slots, lens, keys):
    # Small dyadic inputs keep both dot decompositions exact in FP32.
    scores = torch.einsum("bhd,bld->bhl", q.float(), keys[slots].float())
    scores = scores.to(torch.bfloat16).relu()
    scores = (scores * weights.unsqueeze(-1)).to(torch.bfloat16)
    logits = scores.sum(dim=1, dtype=torch.bfloat16).float()
    visible = torch.arange(slots.shape[1], device=slots.device) < lens[:, None]
    return logits.masked_fill(~visible, -torch.inf)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90 (Hopper)",
)
class TestSM90FP4Indexer(CustomTestCase):
    def test_visible_length_boundaries(self):
        cases = (
            (0, [0]),
            (63, [0, 1, 62, 63]),
            (64, [0, 1, 63, 64]),
            (65, [0, 1, 63, 64, 65]),
            (129, [0, 1, 63, 64, 65, 129]),
        )
        for num_heads in (16, 32):
            for capacity, lengths in cases:
                with self.subTest(num_heads=num_heads, capacity=capacity):
                    q, weights, slots, lens, table, keys = _make_inputs(
                        num_heads, capacity, lengths
                    )
                    actual = fp4_index_logits_decode(
                        q, weights, slots, lens, table, PAGE_SIZE
                    )
                    expected = _reference(q, weights, slots, lens, keys)
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    tail = torch.arange(capacity, device="cuda") >= lens[:, None]
                    self.assertTrue(torch.isneginf(actual[tail]).all().item())

    def test_device_lengths_change_during_graph_replay(self):
        capacity = 129
        for num_heads in (16, 32):
            with self.subTest(num_heads=num_heads):
                q, weights, slots, lens, table, keys = _make_inputs(
                    num_heads, capacity, [capacity] * 3
                )
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        fp4_index_logits_decode(
                            q, weights, slots, lens, table, PAGE_SIZE
                        )
                torch.cuda.current_stream().wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = fp4_index_logits_decode(
                        q, weights, slots, lens, table, PAGE_SIZE
                    )
                for visible_length in (capacity, 0, 65, 63, capacity):
                    with self.subTest(visible_length=visible_length):
                        lens.fill_(visible_length)
                        graph.replay()
                        expected = _reference(q, weights, slots, lens, keys)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(
                            torch.isneginf(actual[:, visible_length:]).all().item()
                        )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90 (Hopper)",
)
class TestSM90FP4IndexerPaged(CustomTestCase):
    """The paged path resolves slots in-kernel instead of materializing [B, L]."""

    def _inputs(self, batch_size: int, lmax: int, ratio: int):
        device = "cuda"
        num_requests = 11
        page_size = 64
        num_pages = 64
        row_padding = 17
        width = lmax * ratio
        generator = torch.Generator(device=device).manual_seed(
            1000 + batch_size * 100 + lmax * 3 + ratio
        )

        # This view has a padded row stride but unit inner stride. It mirrors the
        # pool contract without relying on a fully contiguous allocation.
        req_to_token_storage = torch.randint(
            0,
            num_pages * page_size * ratio,
            (num_requests, width + row_padding),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        req_to_token = req_to_token_storage[:, :width]
        req = torch.arange(batch_size, device=device, dtype=torch.int64) % 5
        visible_lengths = [
            0,
            min(1, lmax),
            max(lmax - 1, 0),
            lmax,
            min(63, lmax),
            min(64, lmax),
            min(511, lmax),
        ]
        lens = torch.tensor(
            (visible_lengths * ((batch_size + 6) // 7))[:batch_size],
            device=device,
            dtype=torch.int64,
        )
        if batch_size == 1:
            lens.fill_(lmax)
        q = torch.randn(
            batch_size,
            32,
            128,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        weights = torch.randn(
            batch_size,
            32,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        table = torch.randint(
            0,
            256,
            (num_pages, page_size * 68),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        # E8M0 scales with arbitrary bytes can intentionally overflow to inf;
        # use a finite range so exact output comparisons test the indexer math.
        table[:, page_size * 64 :] = torch.randint(
            124,
            129,
            (num_pages, page_size * 4),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        positions = torch.arange(lmax, device=device)
        slots = req_to_token[req[:, None], (positions * ratio)[None, :]].to(torch.int64)
        slots = (slots // ratio).masked_fill(positions[None, :] >= lens[:, None], 0)
        return q, weights, req_to_token, req, lens, table, slots, page_size

    def test_paged_matches_materialized_slot_map(self):
        for ratio in (1, 2):
            for batch_size in (1, 7, 64):
                for lmax in (1, 63, 64, 65, 513):
                    with self.subTest(ratio=ratio, batch_size=batch_size, lmax=lmax):
                        (
                            q,
                            weights,
                            req_to_token,
                            req,
                            lens,
                            table,
                            slots,
                            page_size,
                        ) = self._inputs(batch_size, lmax, ratio)
                        expected = fp4_index_logits_decode_materialized(
                            q, weights, slots, lens, table, page_size
                        )
                        actual = fp4_index_logits_decode_paged(
                            q,
                            weights,
                            req_to_token,
                            req,
                            ratio,
                            lmax,
                            lens,
                            table,
                            page_size,
                        )
                        torch.cuda.synchronize()
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int32), expected.view(torch.int32)
                            )
                        )

    def test_graph_replay_with_changed_lengths_and_request_ids(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                lmax = 513
                q, weights, req_to_token, req, lens, table, _, page_size = self._inputs(
                    7, lmax, ratio
                )
                lens.fill_(64)

                def paged():
                    return fp4_index_logits_decode_paged(
                        q,
                        weights,
                        req_to_token,
                        req,
                        ratio,
                        lmax,
                        lens,
                        table,
                        page_size,
                    )

                # Compile and warm up before entering graph capture.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        paged()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = paged()

                positions = torch.arange(lmax, device=q.device)
                for visible_length in (0, 1, 63, 64, 512, lmax):
                    lens.fill_(visible_length)
                    req.copy_(req.roll(1))
                    req_to_token.copy_(req_to_token.roll(1, dims=0))
                    graph.replay()
                    slots = (
                        req_to_token[req[:, None], (positions * ratio)[None, :]].to(
                            torch.int64
                        )
                        // ratio
                    )
                    slots.masked_fill_(positions[None, :] >= lens[:, None], 0)
                    expected = fp4_index_logits_decode_materialized(
                        q, weights, slots, lens, table, page_size
                    )
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int32), expected.view(torch.int32)
                        )
                    )

    def test_paged_ignores_unallocated_invisible_slots(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                q, weights, req_to_token, req, lens, table, _, page_size = self._inputs(
                    7, 513, ratio
                )
                lens.fill_(63)
                req_to_token[:, 63 * ratio :] = -1
                positions = torch.arange(513, device=q.device)
                slots = (
                    req_to_token[req[:, None], (positions * ratio)[None, :]].to(
                        torch.int64
                    )
                    // ratio
                ).masked_fill(positions[None, :] >= lens[:, None], 0)
                expected = fp4_index_logits_decode_materialized(
                    q, weights, slots, lens, table, page_size
                )
                actual = fp4_index_logits_decode_paged(
                    q, weights, req_to_token, req, ratio, 513, lens, table, page_size
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                )

    def test_empty_output(self):
        for batch_size, lmax in ((0, 65), (7, 0)):
            with self.subTest(batch_size=batch_size, lmax=lmax):
                q, weights, req_to_token, req, lens, table, _, page_size = self._inputs(
                    batch_size, lmax, 1
                )
                actual = fp4_index_logits_decode_paged(
                    q, weights, req_to_token, req, 1, lmax, lens, table, page_size
                )
                self.assertEqual(actual.shape, (batch_size, lmax))
                self.assertEqual(actual.dtype, torch.float32)

    def test_paged_rejects_nonunit_req_to_token_inner_stride(self):
        q, weights, req_to_token, req, lens, table, _, page_size = self._inputs(
            1, 64, 1
        )
        strided = torch.empty(
            req_to_token.shape[0],
            req_to_token.shape[1] * 2,
            dtype=torch.int32,
            device="cuda",
        )[:, ::2]
        strided.copy_(req_to_token)
        self.assertEqual(strided.stride(1), 2)
        with self.assertRaises(AssertionError):
            fp4_index_logits_decode_paged(
                q, weights, strided, req, 1, 64, lens, table, page_size
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
