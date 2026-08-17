"""Unit tests for ``C4IndexerBackendMixin._apply_dcp_owner_filter``.

The filter is a pure tensor operation that runs on every DCP rank during
decode.  These tests guard the ownership rule (``slot % dcp_size == dcp_rank``),
the local-index remap (``slot // dcp_size``), the descending sort invariant,
and the no-op fast path when DCP is disabled.

The indexer module imports CUDA kernels at module scope, so we defer the
import into each test function to keep this file CPU-safe.
"""

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _import_filter():
    """Late import to avoid pulling CUDA kernels at module scope."""
    from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin

    return C4IndexerBackendMixin._apply_dcp_owner_filter


def test_filter_keeps_owned_slots_and_remaps() -> None:
    """Rank 0 with dcp_size=2 keeps even slots, remaps to local indices,
    and sorts descending with -1s at the end."""
    apply_filter = _import_filter()

    # Global slots: [0, 1, 2, 3, 4, 5, -1, -1]
    page_indices = torch.tensor([0, 1, 2, 3, 4, 5, -1, -1], dtype=torch.int32)
    with get_parallel().override(dcp_enabled=True, attn_dcp_size=2, dcp_rank=0):
        apply_filter(page_indices)

    # Rank 0 owns even slots: 0, 2, 4 → local: 0, 1, 2
    # Odd slots 1, 3, 5 → -1; pre-existing -1s stay -1
    # Sorted descending: [2, 1, 0, -1, -1, -1, -1, -1]
    expected = torch.tensor([2, 1, 0, -1, -1, -1, -1, -1], dtype=torch.int32)
    assert torch.equal(
        page_indices, expected
    ), f"got {page_indices}, expected {expected}"


def test_filter_dcp_size_4() -> None:
    """With dcp_size=4, rank 1 owns slots where ``slot % 4 == 1``."""
    apply_filter = _import_filter()

    # Global slots: 0..11
    page_indices = torch.arange(12, dtype=torch.int32)
    with get_parallel().override(dcp_enabled=True, attn_dcp_size=4, dcp_rank=1):
        apply_filter(page_indices)

    # Rank 1 owns: 1, 5, 9 → local: 0, 1, 2
    # Sorted descending: [2, 1, 0, -1, ..., -1]
    expected = torch.tensor([2, 1, 0] + [-1] * 9, dtype=torch.int32)
    assert torch.equal(
        page_indices, expected
    ), f"got {page_indices}, expected {expected}"


def test_filter_does_not_touch_invalid_entries() -> None:
    """Pre-existing -1 entries (slots beyond seq_len) stay -1 after filter."""
    apply_filter = _import_filter()

    page_indices = torch.tensor([-1, 3, -1, 7, -1], dtype=torch.int32)
    original_invalid = page_indices == -1
    with get_parallel().override(dcp_enabled=True, attn_dcp_size=2, dcp_rank=1):
        apply_filter(page_indices)

    # After filter, there should be at least as many -1s as before.
    # Rank 1 with dcp_size=2 owns odd slots: 3, 7 → local 1, 3
    # Sorted desc: [3, 1, -1, -1, -1]
    expected = torch.tensor([3, 1, -1, -1, -1], dtype=torch.int32)
    assert torch.equal(
        page_indices, expected
    ), f"got {page_indices}, expected {expected}"
    # Every original -1 is still -1 somewhere in the result (reordered to tail).
    assert (page_indices == -1).sum() >= original_invalid.sum()


def test_filter_skips_when_dcp_disabled() -> None:
    """When ``dcp_enabled=False`` the tensor is returned unchanged."""
    apply_filter = _import_filter()

    page_indices = torch.tensor([5, 3, 1, 7, -1], dtype=torch.int32)
    original = page_indices.clone()
    with get_parallel().override(dcp_enabled=False, attn_dcp_size=1, dcp_rank=0):
        apply_filter(page_indices)

    assert torch.equal(
        page_indices, original
    ), f"tensor mutated when DCP disabled: {page_indices} vs {original}"


def test_filter_matches_c4_and_c128_payloads() -> None:
    """The filter must work on 2-D tensors (bs, seq_len) like the c128 layout."""
    apply_filter = _import_filter()

    # Shape (2, 6) — simulates a small batch of c128 page indices
    page_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int32,
    )
    with get_parallel().override(dcp_enabled=True, attn_dcp_size=2, dcp_rank=0):
        apply_filter(page_indices)

    # Rank 0 owns even slots: row 0 → {0,2,4} → local {0,1,2}, row 1 → {6,8,10} → local {3,4,5}
    # Sorted desc per row:
    expected = torch.tensor(
        [
            [2, 1, 0, -1, -1, -1],
            [5, 4, 3, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(
        page_indices, expected
    ), f"got {page_indices}, expected {expected}"


def test_filter_is_capture_safe() -> None:
    """The owner filter must be CUDA-graph capture safe: it must operate
    in-place (no new tensor allocation for the output) so that the data_ptr
    of the input tensor is preserved after the call.

    This patches ``get_is_capture_mode`` → True to simulate graph-capture
    mode, then verifies:
    1. ``page_indices.data_ptr()`` is unchanged (in-place via ``copy_``).
    2. The filtered values are correct (rank 0, dcp_size=2).
    """

    apply_filter = _import_filter()

    page_indices = torch.tensor([0, 1, 2, 3, 4, 5, -1, -1], dtype=torch.int32)
    original_ptr = page_indices.data_ptr()

    import unittest.mock as mock

    with mock.patch(
        "sglang.srt.models.deepseek_v4.get_is_capture_mode", return_value=True
    ):
        with get_parallel().override(dcp_enabled=True, attn_dcp_size=2, dcp_rank=0):
            apply_filter(page_indices)

    # In-place: same underlying storage.
    assert (
        page_indices.data_ptr() == original_ptr
    ), f"data_ptr changed: {page_indices.data_ptr()} != {original_ptr}"

    # Rank 0 owns even slots: 0,2,4 → local 0,1,2; sorted desc: [2,1,0,-1,...]
    expected = torch.tensor([2, 1, 0, -1, -1, -1, -1, -1], dtype=torch.int32)
    assert torch.equal(
        page_indices, expected
    ), f"got {page_indices}, expected {expected}"


def test_lse_slice_is_non_contiguous() -> None:
    """LSE tensors from FlashMLA kernels can have shape [B, 128, 1] (head64-padded).
    Slicing ``[:, :32]`` produces a non-contiguous view; ``.contiguous()`` fixes it.

    This mirrors the slicing in ``deepseek_v4.py`` at ``lse[:, :ls_heads]``.
    Similarly, the ``o`` tensor has shape [B, 128, 512] and the slice
    ``o[:, out_slice, :]`` is non-contiguous.

    Guard: if PyTorch ever changes slicing to return contiguous views for
    these shapes, this test would need updating — that's the point.
    """
    # LSE: [4, 128, 1] → view [4, 128] → slice [:, :32]
    lse = torch.randn(4, 128, 1)
    lse_flat = lse.view(lse.shape[0], -1)  # [4, 128]
    lse_sliced = lse_flat[:, :32]
    assert (
        not lse_sliced.is_contiguous()
    ), "lse[:, :32] should be non-contiguous for [4,128,1] base tensor"
    assert (
        lse_sliced.contiguous().is_contiguous()
    ), ".contiguous() must produce a contiguous tensor"

    # O tensor: [4, 128, 512] → slice [:, :32, :]
    o = torch.randn(4, 128, 512)
    o_sliced = o[:, :32, :]
    assert (
        not o_sliced.is_contiguous()
    ), "o[:, :32, :] should be non-contiguous for [4,128,512] base tensor"
    assert (
        o_sliced.contiguous().is_contiguous()
    ), ".contiguous() must produce a contiguous tensor"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
