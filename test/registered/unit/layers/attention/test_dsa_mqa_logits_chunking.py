"""Contract tests for the MQA-logits chunk decision shared by the DSA and DSV4
indexers.

On ROCm the `[num_q x num_k]` fp32 logits tensor goes to aiter's
`fp8_mqa_logits`, which only compiles below 2 GiB, so the budget that decides
chunking is a correctness bound there and not only an out-of-memory guard.

The measured memory budget is stubbed: it is the only input the limit has to
beat, and stubbing it keeps these tests on CPU.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.attention.mqa_logits_utils import (  # noqa: E402
    MQA_LOGITS_MAX_BYTES_ROCM,
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
    mqa_logits_should_chunk,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

CEILING = MQA_LOGITS_MAX_BYTES_ROCM
# More than any single logits tensor here needs, so it never decides a case.
HUGE_MEM_BUDGET = 64 * 2**30


def _decide(num_q, num_k, mem_budget=HUGE_MEM_BUDGET, is_hip=True):
    return mqa_logits_should_chunk(
        num_rows=num_q,
        num_cols=num_k,
        get_budget_bytes=lambda: mem_budget,
        rocm=is_hip,
    )


def test_the_ceiling_is_the_largest_logits_aiter_still_takes():
    # 16384 x 32768 x 4 bytes is exactly 2 GiB, and aiter compares `bytes <
    # 2 GiB`, so that shape has to chunk and one KV token less must not.
    assert _decide(16_384, 32_768) == (True, CEILING)
    assert _decide(16_384, 32_767) == (False, CEILING)


def test_a_smaller_memory_budget_still_wins():
    one_gib = 2**30
    assert _decide(16_384, 32_767, mem_budget=one_gib) == (True, one_gib)


def test_off_rocm_the_budget_is_untouched():
    # Elsewhere the logits go to DeepGEMM, which has no such limit.
    assert _decide(16_384, 32_768, is_hip=False) == (False, HUGE_MEM_BUDGET)


def test_rows_per_chunk_splits_the_budget_across_the_k_axis():
    # The DSA kpool path chunks per request under this helper.
    one_gib = 2**30
    # 2 GiB of columns per row: only a single row fits, never zero rows.
    assert (
        mqa_logits_rows_per_chunk(
            num_rows=16_384, row_bytes=2**31, budget_bytes=one_gib
        )
        == 1
    )
    # 4 KiB per row -> 262144 rows fit, but never more rows than exist.
    assert (
        mqa_logits_rows_per_chunk(num_rows=16_384, row_bytes=4096, budget_bytes=one_gib)
        is None
    )
    assert (
        mqa_logits_rows_per_chunk(num_rows=2**19, row_bytes=4096, budget_bytes=one_gib)
        == 262_144
    )


def test_row_bytes_uses_deep_gemm_256_column_alignment():
    # DeepGEMM pads the logits row stride to 1024 bytes (256 fp32 columns).
    assert mqa_logits_row_bytes(0) == 0
    assert mqa_logits_row_bytes(1) == 256 * 4
    assert mqa_logits_row_bytes(256) == 256 * 4
    assert mqa_logits_row_bytes(257) == 512 * 4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
