"""GLM-5.3-Flash (glm5_next, KDA hybrid) decode-CP pool sizing — CPU only.

Covers the three pool-side contracts DCP relies on for the KDA hybrid:

1. The KDA/GDN recurrent state is per-request and slot-addressed through the
   Mamba pool: under DCP every rank holds a full replica of the same slot
   space, so the pool geometry must NOT widen with ``attn_dcp_size`` (unlike
   the DSA indexer K cache, which is global-slot addressed and does).
2. The draft KV/indexer pools of a DSA-family target are DCP-replicated (the
   draft worker indexes the shared allocator's widened virtual loc space
   raw), so the target's ``_cell_size`` budget must carry the ``×dcp_size``
   factor or the draft over-commits memory the target already reserved.
3. The hybrid linear-attention backend rejects prefill-CP + DCP at startup,
   mirroring the dense DSA guard in ``dsa_backend``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _published_context():
    # The configurator and parallel reads go through published bags; publish
    # a minimal server-args config so overrides resolve.
    override = get_context().override_server_args(model_path="dummy")
    override.install()
    try:
        yield
    finally:
        override.restore()


def _kda_cache_params(num_heads=8, head_dim=128, conv_kernel_size=4):
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
    )

    shape = KimiLinearStateShape.create(
        tp_world_size=1,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
    )
    return KimiLinearCacheParams(shape=shape, layers=[1, 3])


def _make_mamba_pool(device="cpu"):
    from sglang.srt.mem_cache.memory_pool import MambaPool

    params = _kda_cache_params()
    return MambaPool(
        size=16,
        spec_state_size=4,
        cache_params=params,
        mamba_layer_ids=list(params.layers),
        device=device,
    )


def test_kda_state_pool_is_replicated_not_widened_under_dcp() -> None:
    # Each DCP rank runs the same requests, so per-request state is a full
    # per-rank replica: the pool keeps one slot per request regardless of
    # attn_dcp_size. A widened pool would waste dcp_size-1 copies of state
    # that no rank addresses differently.
    pool_dcp1 = _make_mamba_pool()
    with get_parallel().override(dcp_enabled=True, attn_dcp_size=4):
        pool_dcp4 = _make_mamba_pool()

    assert pool_dcp1.size == pool_dcp4.size
    assert pool_dcp1.mamba_cache.conv[0].shape == pool_dcp4.mamba_cache.conv[0].shape
    assert pool_dcp1.mamba_cache.temporal.shape == pool_dcp4.mamba_cache.temporal.shape


def _kvc(*, eagle: bool, dsa: bool):
    hf_config = SimpleNamespace()
    hf_config.get_text_config = lambda: hf_config
    return SimpleNamespace(
        use_mla_backend=True,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_dtype_str="fp8_e4m3",
        model_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            context_len=8192,
            hf_config=hf_config,
            linear_attn_registry_result=None,
        ),
        server_args=SimpleNamespace(enable_hisparse=False),
        is_draft_worker=False,
        mambaish_config=None,
        layer_info=SimpleNamespace(start_layer=0, end_layer=2, num_effective_layers=2),
        spec_algorithm=SimpleNamespace(
            is_eagle=lambda: eagle,
            is_standalone=lambda: False,
            is_dflash=lambda: False,
            is_dflash_family=lambda: False,
            is_none=lambda: not eagle,
        ),
        spec_aux_config=SimpleNamespace(
            eagle_draft_num_layers=1,
            eagle_draft_swa_num_layers=None,
            dflash_draft_num_layers=None,
        ),
        ps=get_parallel(),
    )


def _cell_size(kvc, dcp_size: int) -> int:
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

    configurator = object.__new__(DefaultPoolConfigurator)
    with get_parallel().override(attn_tp_size=8, dcp_enabled=(dcp_size > 1), attn_dcp_size=dcp_size):
        with (
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
            configurator.__init__(kvc)
    return configurator._cell_size


def test_dsa_draft_pool_scales_with_dcp_size() -> None:
    # Component budget (per target max_total_num_tokens token), with
    # kv_lora_rank=512 / qk_rope_head_dim=64 and the non-DSA MLA dim of 576
    # (calculate_mla_kv_cache_dim's SimpleNamespace hf_config is not DSA):
    #   target KV    = 576 * 2 layers            (DCP-sharded, dcp-invariant)
    #   target index = 132 * 2 layers * dcp      (global-slot replication)
    #   draft KV     = 576 * 1 layer  * dcp      (draft pool spans the shared
    #                                             allocator's widened virtual
    #                                             loc space)
    #   draft index  = 132 * 1 layer  * dcp^2    (draft pool is sized by the
    #             widened virtual space AND keeps global-slot index_k, so the
    #             budget carries both factors to track the real allocation)
    kvc = _kvc(eagle=True, dsa=True)
    for dcp_size in (1, 4):
        cell_size = _cell_size(kvc, dcp_size)
        expected = 576 * 2 + 132 * 2 * dcp_size + 576 * dcp_size + 132 * dcp_size**2
        assert cell_size == expected, f"dcp_size={dcp_size}: {cell_size} != {expected}"


def test_target_only_cell_size_scales_only_by_indexer_replication() -> None:
    # No draft (is_none): the only DCP term left in the budget is the
    # target-side indexer replication; the latent KV pool is DCP-sharded and
    # must stay dcp-invariant.
    kvc = _kvc(eagle=False, dsa=True)
    cell_dcp1 = _cell_size(kvc, 1)
    cell_dcp4 = _cell_size(kvc, 4)
    # 2 indexer layers x 132 bytes x (4 - 1) extra replica ranks.
    assert cell_dcp4 - cell_dcp1 == 132 * 2 * 3


def test_hybrid_backend_rejects_prefill_cp_with_dcp() -> None:
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
        MambaAttnBackendBase,
    )

    full = SimpleNamespace(
        token_to_kv_pool=None,
        req_to_token_pool=None,
        kv_index_translator=None,
        needs_cpu_seq_lens=False,
    )

    class _FakeLinear(MambaAttnBackendBase):
        def __init__(self):
            pass

    linear = _FakeLinear()

    with get_parallel().override(dcp_enabled=True, attn_dcp_size=2):
        backend = HybridLinearAttnBackend(full, linear, [0, 2])
        assert backend.full_attn_layers == [0, 2]

        override = get_context().override_server_args(enable_prefill_cp=True)
        override.install()
        try:
            with pytest.raises(ValueError, match="prefill CP"):
                HybridLinearAttnBackend(full, linear, [0, 2])
        finally:
            override.restore()

    with get_parallel().override(dcp_enabled=False, attn_dcp_size=1):
        HybridLinearAttnBackend(full, linear, [0, 2])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
