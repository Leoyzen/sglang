import unittest
from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.dsa.transform_index as transform_index_module
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_fast,
    transform_index_page_table_prefill_fast,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")

TOPK = 2048


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDSATransformIndex(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _make_page_table(self, rows: int, context_length: int) -> torch.Tensor:
        columns = torch.arange(context_length, dtype=torch.int32, device=self.device)
        row_bias = (
            torch.arange(rows, dtype=torch.int32, device=self.device).unsqueeze(1) * 17
        )
        return columns.unsqueeze(0) + row_bias

    def _make_topk(self, rows: int, context_length: int) -> torch.Tensor:
        topk = (
            torch.arange(TOPK, dtype=torch.int64, device=self.device)
            .remainder(context_length)
            .repeat(rows, 1)
        )
        if rows > 0:
            topk[:, 0] = 0
            topk[:, 1] = context_length - 1
            topk[:, 257::257] = -1
        return topk

    def _expected(
        self,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
        extend_lens_cpu: list[int],
        output_num_tokens: int,
        page_table_is_expanded: bool,
    ) -> torch.Tensor:
        real_num_tokens = sum(extend_lens_cpu)
        expected = torch.full(
            (output_num_tokens, TOPK),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        if real_num_tokens == 0:
            return expected

        if page_table_is_expanded:
            source_rows = page_table[:real_num_tokens]
        else:
            request_ids = torch.repeat_interleave(
                torch.arange(
                    len(extend_lens_cpu), dtype=torch.int64, device=self.device
                ),
                torch.tensor(extend_lens_cpu, dtype=torch.int64, device=self.device),
            )
            source_rows = page_table[request_ids]

        real_topk = topk_indices[:real_num_tokens]
        torch.gather(
            source_rows,
            dim=1,
            index=real_topk.clamp(min=0),
            out=expected[:real_num_tokens],
        )
        expected[:real_num_tokens][real_topk < 0] = -1
        return expected

    def _check_decode_case(
        self,
        batch_size: int,
        context_length: int,
        *,
        zero_row_stride: bool = False,
        provide_result: bool = False,
    ) -> None:
        if zero_row_stride:
            page_table = self._make_page_table(1, context_length).expand(batch_size, -1)
        else:
            page_table = self._make_page_table(batch_size, context_length)
        topk_indices = self._make_topk(batch_size, context_length)
        expected = torch.empty(
            (batch_size, TOPK), dtype=torch.int32, device=self.device
        )
        torch.gather(
            page_table,
            dim=1,
            index=topk_indices.clamp(min=0),
            out=expected,
        )
        expected[topk_indices < 0] = -1
        result = torch.empty_like(expected) if provide_result else None

        actual = transform_index_page_table_decode_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            result=result,
        )
        torch.cuda.synchronize()
        if result is not None:
            self.assertIs(actual, result)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _check_case(
        self,
        extend_lens_cpu: list[int],
        context_length: int,
        *,
        page_table_is_expanded: bool,
        topk_padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        real_num_tokens = sum(extend_lens_cpu)
        page_table_rows = (
            real_num_tokens if page_table_is_expanded else len(extend_lens_cpu)
        )
        topk_num_tokens = real_num_tokens + topk_padding
        output_num_tokens = topk_num_tokens + output_padding
        page_table = self._make_page_table(page_table_rows, context_length)
        topk_indices = self._make_topk(topk_num_tokens, context_length)
        expected = self._expected(
            page_table,
            topk_indices,
            extend_lens_cpu,
            output_num_tokens,
            page_table_is_expanded,
        )

        actual = transform_index_page_table_prefill_fast(
            page_table=page_table,
            topk_indices=topk_indices,
            extend_lens_cpu=extend_lens_cpu,
            output_num_tokens=output_num_tokens,
            page_table_is_expanded=page_table_is_expanded,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_uses_dedicated_kernel(self):
        extend_lens_cpu = [2, 1]
        context_length = 4096
        page_table = self._make_page_table(len(extend_lens_cpu), context_length)
        topk_indices = self._make_topk(sum(extend_lens_cpu), context_length)

        with patch.object(
            transform_index_module,
            "transform_index_page_table_decode_fast",
            side_effect=AssertionError("prefill must not launch decode per request"),
        ):
            transform_index_page_table_prefill_fast(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
            )

    def test_prefill_uses_device_cu_seqlens(self):
        extend_lens_cpu = [2, 1]
        context_length = 4096
        page_table = self._make_page_table(len(extend_lens_cpu), context_length)
        topk_indices = self._make_topk(sum(extend_lens_cpu), context_length)
        cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32, device=self.device)

        with patch.object(
            transform_index_module.torch,
            "tensor",
            side_effect=AssertionError("must reuse device-side metadata"),
        ):
            transform_index_page_table_prefill_fast(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
                cu_seqlens_q=cu_seqlens_q,
            )

    def test_prefill_page_table_row_stride_is_not_specialized(self):
        kernel = transform_index_module.transform_index_page_table_prefill_kernel
        stride_param = next(
            param for param in kernel.params if param.name == "page_table_stride_0"
        )

        self.assertFalse(stride_param.is_constexpr)
        self.assertTrue(stride_param.do_not_specialize)

    def test_width_bound_is_not_specialized(self):
        # Expanded EAGLE tables give every context length its own column
        # width; a constexpr PAGE_TABLE_WIDTH recompiled the kernel per
        # width. The bound must be a runtime scalar (do_not_specialize) in
        # BOTH kernels.
        for kernel_name in (
            "transform_index_page_table_prefill_kernel",
            "transform_index_page_table_decode_kernel",
        ):
            with self.subTest(kernel=kernel_name):
                kernel = getattr(transform_index_module, kernel_name)
                width_param = next(
                    param for param in kernel.params if param.name == "page_table_width"
                )

                self.assertFalse(width_param.is_constexpr)
                self.assertTrue(width_param.do_not_specialize)

    def test_prefill_dynamic_page_table_row_strides(self):
        context_lengths = (4096, 4160, 4224)
        kernel = transform_index_module.transform_index_page_table_prefill_kernel

        self._check_case(
            [2, 1],
            context_lengths[0],
            page_table_is_expanded=True,
        )
        kernel_cache = kernel.device_caches[torch.cuda.current_device()][0]
        specialization_count = len(kernel_cache)

        for context_length in context_lengths[1:]:
            with self.subTest(context_length=context_length):
                self._check_case(
                    [2, 1],
                    context_length,
                    page_table_is_expanded=True,
                )
                self.assertEqual(len(kernel_cache), specialization_count)

    def test_mixed_lengths_padding_and_empty_batch(self):
        self._check_case(
            [0, 3, 1, 0, 4],
            8192,
            page_table_is_expanded=False,
            topk_padding=5,
            output_padding=7,
        )
        self._check_case(
            [0, 0],
            16,
            page_table_is_expanded=False,
            output_padding=8,
        )

    def test_large_batch_size(self):
        self._check_case(
            [1] * 8192,
            4096,
            page_table_is_expanded=False,
        )

    def test_large_context_lengths(self):
        for context_length, page_table_is_expanded in (
            (640_000, True),
            (1_000_000, False),
        ):
            with self.subTest(
                context_length=context_length,
                page_table_is_expanded=page_table_is_expanded,
            ):
                self._check_case(
                    [2, 1],
                    context_length,
                    page_table_is_expanded=page_table_is_expanded,
                )

    def test_decode_fast_correctness_and_strides(self):
        self._check_decode_case(17, 8192, provide_result=True)
        self._check_decode_case(17, 8192, zero_row_stride=True)

    def test_decode_fast_extreme_shapes(self):
        self._check_decode_case(8192, 4096)
        self._check_decode_case(2, 1_000_000)

    def test_dcp_owned_rows_survive_width_bound(self):
        # Under DCP the page table lives in the WIDENED page domain: its
        # column width is tree_page * dcp_size and ownership is
        # slot % dcp_size == rank (see dsa_backend DCP owner filter). The OOB
        # guard must use THAT table's own width -- deriving the bound from a
        # folded width (width // dcp_size) would silently drop owned rows
        # near the top of the widened domain.
        context_length = 4096
        dcp_size = 2
        rows = 4
        # Flat table: gathered value == index, so ownership is index parity
        # independent of the row.
        page_table = (
            torch.arange(context_length, dtype=torch.int32, device=self.device)
            .unsqueeze(0)
            .repeat(rows, 1)
        )
        topk_indices = self._make_topk(rows, context_length)

        for dcp_rank in (0, 1):
            with self.subTest(dcp_rank=dcp_rank):
                probe = topk_indices.clone()
                # Park owned and foreign slots at the very top of the widened
                # domain: the owned one must survive (mapped to its folded
                # row), the foreign one must be dropped by the owner filter --
                # and neither may be dropped by the width bound.
                owned_top = context_length - dcp_size + dcp_rank
                foreign_top = context_length - dcp_size + (1 - dcp_rank)
                oob_top = context_length  # first index past the width
                probe[:, 3] = owned_top
                probe[:, 4] = foreign_top
                probe[:, 5] = oob_top

                expected = torch.full_like(probe, -1, dtype=torch.int32)
                in_width = (probe >= 0) & (probe < context_length)
                owned = in_width & (probe % dcp_size == dcp_rank)
                expected[owned] = (probe[owned] // dcp_size).to(torch.int32)

                actual = transform_index_page_table_decode_fast(
                    page_table=page_table,
                    topk_indices=probe,
                    page_size=1,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # The width bound did not eat the top-of-domain owned row.
                self.assertTrue(
                    bool((actual[:, 3] == owned_top // dcp_size).all()),
                    "owned row at the top of the widened page domain was "
                    "dropped by the width bound",
                )
                self.assertTrue(bool((actual[:, 4] == -1).all()))
                self.assertTrue(bool((actual[:, 5] == -1).all()))

    def test_dcp_prefill_owned_rows_survive_width_bound(self):
        context_length = 4096
        dcp_size = 2
        extend_lens_cpu = [2, 1]
        real_num_tokens = sum(extend_lens_cpu)
        for page_table_is_expanded in (False, True):
            with self.subTest(page_table_is_expanded=page_table_is_expanded):
                page_table_rows = (
                    real_num_tokens if page_table_is_expanded else len(extend_lens_cpu)
                )
                page_table = (
                    torch.arange(context_length, dtype=torch.int32, device=self.device)
                    .unsqueeze(0)
                    .repeat(page_table_rows, 1)
                )
                topk_indices = self._make_topk(real_num_tokens, context_length)

                for dcp_rank in (0, 1):
                    with self.subTest(
                        page_table_is_expanded=page_table_is_expanded,
                        dcp_rank=dcp_rank,
                    ):
                        probe = topk_indices.clone()
                        owned_top = context_length - dcp_size + dcp_rank
                        foreign_top = context_length - dcp_size + (1 - dcp_rank)
                        probe[:, 3] = owned_top
                        probe[:, 4] = foreign_top

                        if page_table_is_expanded:
                            source_rows = page_table
                        else:
                            request_ids = torch.repeat_interleave(
                                torch.arange(
                                    len(extend_lens_cpu),
                                    dtype=torch.int64,
                                    device=self.device,
                                ),
                                torch.tensor(
                                    extend_lens_cpu,
                                    dtype=torch.int64,
                                    device=self.device,
                                ),
                            )
                            source_rows = page_table[request_ids]

                        expected = torch.full_like(probe, -1, dtype=torch.int32)
                        gathered = torch.gather(
                            source_rows, dim=1, index=probe.clamp(min=0)
                        )
                        in_width = (probe >= 0) & (probe < context_length)
                        owned = in_width & (gathered % dcp_size == dcp_rank)
                        expected[owned] = (gathered[owned] // dcp_size).to(torch.int32)

                        actual = transform_index_page_table_prefill_fast(
                            page_table=page_table,
                            topk_indices=probe,
                            extend_lens_cpu=extend_lens_cpu,
                            page_size=1,
                            page_table_is_expanded=page_table_is_expanded,
                            dcp_size=dcp_size,
                            dcp_rank=dcp_rank,
                        )
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(
                            bool((actual[:, 3] == (owned_top // dcp_size)).all()),
                            "owned row at the top of the widened page domain "
                            "was dropped by the width bound (prefill)",
                        )


if __name__ == "__main__":
    unittest.main()
