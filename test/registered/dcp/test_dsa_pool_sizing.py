from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.model_executor.pool_configurator import (
    DefaultPoolConfigurator,
    _get_dsa_indexer_cache_token_multiplier,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _published_context():
    # _compute_dsa_indexer_cell_size reads the resolved memory config
    # (via _should_elide_dsa_index_k), so the context must be published.
    override = get_context().override_server_args(model_path="dummy")
    override.install()
    try:
        yield
    finally:
        override.restore()


def _kvc(*, enable_hisparse: bool = False):
    return SimpleNamespace(server_args=SimpleNamespace(enable_hisparse=enable_hisparse))


def test_dsa_indexer_cache_is_replicated_across_dcp_ranks() -> None:
    with get_parallel().override(dcp_enabled=True, attn_dcp_size=8):
        assert _get_dsa_indexer_cache_token_multiplier(_kvc()) == 8


def test_dsa_indexer_cache_has_no_multiplier_without_dcp() -> None:
    with get_parallel().override(dcp_enabled=False, attn_dcp_size=1):
        assert _get_dsa_indexer_cache_token_multiplier(_kvc()) == 1


def test_dsa_cell_size_accounts_for_dcp_indexer_replication() -> None:
    kvc = SimpleNamespace(
        use_mla_backend=True,
        kv_cache_dtype=torch.float8_e4m3fn,
        model_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            hf_config=SimpleNamespace(),
        ),
        server_args=SimpleNamespace(enable_hisparse=False),
        is_draft_worker=False,
        mambaish_config=None,
        layer_info=SimpleNamespace(start_layer=0, end_layer=2),
    )
    configurator = object.__new__(DefaultPoolConfigurator)

    with (
        get_parallel().override(attn_tp_size=8, dcp_enabled=True, attn_dcp_size=8),
        patch(
            "sglang.srt.layers.cp.utils.get_glm_dsa_layer_split_effective_num_layers",
            return_value=2,
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.is_deepseek_dsa",
            return_value=True,
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.get_dsa_index_head_dim",
            return_value=128,
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.dsa_layer_skips_topk",
            return_value=False,
        ),
    ):
        cell_size = configurator._compute_cell_size(kvc, num_layers=2)

    # Per layer: 576 bytes of MLA KV plus 132 bytes of replicated indexer KV
    # for each of the eight DCP ranks.
    assert cell_size == (576 + 132 * 8) * 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
