from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "mode, expected",
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.IDLE, True),
        (ForwardMode.EXTEND, False),
        (ForwardMode.MIXED, False),
        (ForwardMode.TARGET_VERIFY, False),
        (ForwardMode.DRAFT_EXTEND_V2, False),
    ],
)
def test_dcp_fuses_decode_only(mode: ForwardMode, expected: bool) -> None:
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.use_fused_topk = True
    backend.dcp_enabled = True
    backend.hisparse_coordinator = None

    forward_batch = SimpleNamespace(forward_mode=mode)
    assert backend._use_fused_topk_for_batch(forward_batch) is expected


def test_fused_topk_disabled_is_shared_by_producer_and_consumer() -> None:
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.use_fused_topk = False
    backend.dcp_enabled = True
    backend.hisparse_coordinator = None
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    assert not backend._use_fused_topk_for_batch(forward_batch)


def test_non_dcp_extend_keeps_existing_fused_behavior() -> None:
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.use_fused_topk = True
    backend.dcp_enabled = False
    backend.hisparse_coordinator = None
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

    assert backend._use_fused_topk_for_batch(forward_batch)
