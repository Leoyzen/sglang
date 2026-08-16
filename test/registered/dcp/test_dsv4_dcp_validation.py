"""Unit tests for ``validate_deepseek_v4_dcp``.

The validator runs inside ``ServerArgs.__post_init__`` and reads only
server_args fields (no parallel state).  These tests use ``SimpleNamespace``
stubs to exercise every branch: early-return, rejection paths, and the
``attn_tp_size % dcp_size`` divisibility check under different DP-attention
and CP configurations.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v4_dcp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _make_args(**overrides) -> SimpleNamespace:
    """Build a minimal ServerArgs stub with DSV4-relevant defaults."""
    defaults = dict(
        dcp_size=2,
        dcp_indexer_backend="replicated",
        enable_prefill_cp=False,
        enable_hierarchical_cache=False,
        enable_dp_attention=False,
        dp_size=1,
        tp_size=4,
        attn_cp_size=1,
        speculative_algorithm=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Early return: dcp_size ≤ 1 ──────────────────────────────────────────


def test_dcp_size_1_returns_early() -> None:
    """dcp_size=1 is a no-op (DCP effectively disabled)."""
    validate_deepseek_v4_dcp(_make_args(dcp_size=1))


def test_dcp_size_none_returns_early() -> None:
    """dcp_size=None must not raise."""
    validate_deepseek_v4_dcp(_make_args(dcp_size=None))


# ── Rejection paths ─────────────────────────────────────────────────────


def test_rejects_prefill_cp_with_dcp() -> None:
    """DCP + prefill CP is rejected in the initial release."""
    with pytest.raises(ValueError, match="prefill CP"):
        validate_deepseek_v4_dcp(_make_args(enable_prefill_cp=True))


def test_rejects_hierarchical_cache_with_dcp() -> None:
    """DCP + HiCache is rejected in the initial release."""
    with pytest.raises(ValueError, match="HiCache"):
        validate_deepseek_v4_dcp(_make_args(enable_hierarchical_cache=True))


def test_rejects_non_replicated_indexer_backend() -> None:
    """owner_sharded indexer backend is DSA-only and rejected for DSV4 DCP."""
    with pytest.raises(ValueError, match="replicated"):
        validate_deepseek_v4_dcp(_make_args(dcp_indexer_backend="owner_sharded"))


def test_rejects_attn_tp_not_divisible_by_dcp() -> None:
    """attn_tp_size=4, dcp_size=3 → 4 % 3 != 0 → ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        validate_deepseek_v4_dcp(
            _make_args(tp_size=4, dp_size=1, dcp_size=3, attn_cp_size=1)
        )


# ── Acceptance paths ────────────────────────────────────────────────────


def test_accepts_valid_config() -> None:
    """tp_size=4, dcp_size=2, everything else default → no exception."""
    validate_deepseek_v4_dcp(
        _make_args(tp_size=4, dp_size=1, dcp_size=2, attn_cp_size=1)
    )


def test_attn_tp_accounts_for_dp_attention() -> None:
    """Under DP-attention, attn_tp = tp_size // dp_size // attn_cp_size.

    tp_size=8, dp_size=4, enable_dp_attention=True → attn_tp=2.
    dcp_size=2 divides 2 → valid.
    dcp_size=4 does not divide 2 → ValueError.
    """
    # Valid: attn_tp=2, dcp_size=2
    validate_deepseek_v4_dcp(
        _make_args(
            tp_size=8,
            dp_size=4,
            enable_dp_attention=True,
            attn_cp_size=1,
            dcp_size=2,
        )
    )
    # Invalid: attn_tp=2, dcp_size=4
    with pytest.raises(ValueError, match="divisible"):
        validate_deepseek_v4_dcp(
            _make_args(
                tp_size=8,
                dp_size=4,
                enable_dp_attention=True,
                attn_cp_size=1,
                dcp_size=4,
            )
        )


def test_speculative_not_rejected() -> None:
    """The validator deliberately does NOT reject speculative algorithms.

    DSV4 DCP + speculative draft support is a planned follow-up, so
    setting speculative_algorithm should not cause this validator to raise.
    """
    validate_deepseek_v4_dcp(_make_args(speculative_algorithm="DSPARK"))


def test_attn_cp_size_none_treated_as_one() -> None:
    """When attn_cp_size is None, the validator uses ``or 1`` so it should
    not crash with a TypeError."""
    validate_deepseek_v4_dcp(
        _make_args(tp_size=4, dp_size=1, dcp_size=2, attn_cp_size=None)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
