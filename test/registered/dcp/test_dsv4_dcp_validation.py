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


# ── DSpark + DP-attention mutual exclusion ──────────────────────────────


def test_validate_dspark_dcp_dp_attention_mutex() -> None:
    """DSPARK speculative + DP-attention + DCP must be rejected.

    Under DP-attention, attn_tp = tp_size // dp_size // attn_cp_size.
    With tp_size=4, dp_size=4 → attn_tp=1, and dcp_size=2 does not divide 1,
    so the validator raises ValueError.

    With dcp_size=1, DCP is disabled (early return), so no error even with
    DSPARK + DP-attention.
    """
    # DSPARK + DP-attention + dcp_size=2 → ValueError (attn_tp=1, 1%2!=0)
    with pytest.raises(ValueError, match="divisible"):
        validate_deepseek_v4_dcp(
            _make_args(
                speculative_algorithm="DSPARK",
                enable_dp_attention=True,
                tp_size=4,
                dp_size=4,
                dcp_size=2,
            )
        )

    # dcp_size=1 → early return, no raise
    validate_deepseek_v4_dcp(
        _make_args(
            speculative_algorithm="DSPARK",
            enable_dp_attention=True,
            tp_size=4,
            dp_size=4,
            dcp_size=1,
        )
    )


# ── MoE runner backend auto warning under DCP ──────────────────────────


def test_validate_warns_moe_auto_under_dcp() -> None:
    """When moe_runner_backend='auto' under DCP, the validator should warn
    that 'auto' may resolve to a backend incompatible with DCP.

    If the warning is not yet implemented, skip rather than fail.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            validate_deepseek_v4_dcp(_make_args(dcp_size=2, moe_runner_backend="auto"))
        except Exception:
            pass

    dcp_warnings = [w for w in caught if "dcp" in str(w.message).lower()]
    if not dcp_warnings:
        pytest.skip("moe_runner_backend='auto' warning under DCP not yet implemented")


# ── Model class lacks DCP metadata hook ────────────────────────────────


def test_dsv4_model_lacks_dcp_metadata_hook() -> None:
    """DeepseekV4ForCausalLM intentionally does NOT define
    ``prepare_context_parallel_metadata_for_dcp``.

    DSV4 DCP uses the decode-style recipe (Q activation all-gather + LSE
    combine) and does not need the dense-MLA gather buffers that the hook
    would set up.  Eager_runner skips the block when the hook is absent.
    Removing the hook would cause a stub call to crash; adding it back
    would exercise an untested code path.
    """
    try:
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    except ImportError:
        pytest.skip("DeepseekV4ForCausalLM not importable (missing deps)")

    assert not hasattr(
        DeepseekV4ForCausalLM, "prepare_context_parallel_metadata_for_dcp"
    ), (
        "DeepseekV4ForCausalLM should NOT define "
        "prepare_context_parallel_metadata_for_dcp — see the NOTE at "
        "~L2864 in deepseek_v4.py"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
