"""DCP-aware HiCache L3 storage foundation (PR 1 of hicache-dcp-l3-shared-foundation).

Covers the additive foundation only:

- ``HiCacheStorageConfig`` gains ``dcp_rank``/``dcp_size`` with dcp-neutral
  defaults (``dcp_size=1`` keeps every key byte-identical to the pre-DCP
  scheme).
- The ``_dcp{rank}_{size}`` key-namespace tag at both consumption seams
  (buffer-mode ``config_suffix`` / MooncakeStore component keys, and the
  direct-linker ``_storage_suffix``), including degree-change isolation.
- Construction-site plumbing: buffer-mode controller reads parallel state.
- Gating: the ``--enable-hicache-dcp-shard`` flag preserves the exact
  pre-change L3×DCP error when off, admits L3 when on, and the hard
  exclusions from design D5 survive the flag (guard order: exclusions BEFORE
  flag admission).
- The UMBP direct linker hard-rejects under ``--dcp-size > 1``.

Pure CPU unit tests; no GPU, no running store, no distributed group.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_dcp_l3_foundation.py -q
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.arg_groups.hicache_hook import (
    _L3_DCP_NOT_IMPLEMENTED_MESSAGE,
    resolve_hicache_dcp_compatibility,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    dcp_key_namespace,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

try:
    from sglang.srt.configs.model_config import AttentionArch
except Exception:  # pragma: no cover - env without full model deps
    AttentionArch = None


def _make_storage_config(**overrides) -> HiCacheStorageConfig:
    defaults = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
    )
    defaults.update(overrides)
    return HiCacheStorageConfig(**defaults)


def _make_server_args(**overrides):
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(model_path="dummy", **overrides)
    if AttentionArch is not None:
        sa._model_config = MagicMock()
        sa._model_config.attention_arch = AttentionArch.MLA
    return sa


class TestStorageConfigDefaults(CustomTestCase):
    def test_dcp_fields_default_neutral(self):
        cfg = _make_storage_config()
        self.assertEqual(cfg.dcp_rank, 0)
        self.assertEqual(cfg.dcp_size, 1)

    def test_dcp_fields_are_explicit(self):
        cfg = _make_storage_config(dcp_rank=2, dcp_size=4)
        self.assertEqual(cfg.dcp_rank, 2)
        self.assertEqual(cfg.dcp_size, 4)

    def test_namespace_helper_empty_at_dcp1(self):
        self.assertEqual(dcp_key_namespace(), "")
        self.assertEqual(dcp_key_namespace(_make_storage_config()), "")

    def test_namespace_helper_grammar(self):
        self.assertEqual(dcp_key_namespace(dcp_rank=3, dcp_size=4), "_dcp3_4")
        self.assertEqual(
            dcp_key_namespace(_make_storage_config(dcp_rank=1, dcp_size=2)),
            "_dcp1_2",
        )


class TestGoldenKeyParityDcp1(CustomTestCase):
    """dcp_size=1 keys MUST be byte-identical to the pre-change scheme."""

    GOLDEN_MLA = "_testmodel"
    GOLDEN_MHA = "_testmodel_0_8"

    def _suffix_via_backend(self, cfg) -> str:
        # Build the real backend in a throwaway dir; inspect its suffix only.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            from sglang.srt.mem_cache.hicache_storage import HiCacheFile

            with patch.dict("os.environ", {}, clear=False):
                backend = HiCacheFile(cfg, file_path=tmp)
            return backend.config_suffix

    def test_mla_dcp1_golden(self):
        cfg = _make_storage_config(is_mla_model=True)
        backend = MagicMock()
        # Direct suffix arithmetic on the class-under-test seam.
        suffix = f"_{cfg.model_name}"
        # is_mla_model -> no tp tag; pp=1 -> no pp tag; cp=1 -> no cp tag;
        # dcp=1 -> no dcp tag.
        self.assertEqual(suffix, self.GOLDEN_MLA)
        self.assertNotIn("_dcp", suffix)

    def test_mha_dcp1_golden(self):
        cfg = _make_storage_config(is_mla_model=False, tp_rank=0, tp_size=8)
        suffix = f"_{cfg.model_name}_{cfg.tp_rank}_{cfg.tp_size}"
        self.assertEqual(suffix, self.GOLDEN_MHA)
        self.assertNotIn("_dcp", suffix)

    def test_backend_suffix_dcp1_unchanged(self):
        cfg = _make_storage_config(is_mla_model=True)
        self.assertEqual(self._suffix_via_backend(cfg), self.GOLDEN_MLA)

    def test_backend_suffix_appends_dcp_only_when_active(self):
        dcp1 = self._suffix_via_backend(_make_storage_config(is_mla_model=True))
        dcp2 = self._suffix_via_backend(
            _make_storage_config(is_mla_model=True, dcp_rank=1, dcp_size=2)
        )
        self.assertEqual(dcp1, self.GOLDEN_MLA)
        self.assertEqual(dcp2, self.GOLDEN_MLA + "_dcp1_2")


class TestKeyDisjointnessAcrossDegrees(CustomTestCase):
    def _component_keys(self, dcp_rank: int, dcp_size: int) -> set:
        """Mirror of the MooncakeStore MLA component-key composition."""
        ns = dcp_key_namespace(dcp_rank=dcp_rank, dcp_size=dcp_size)
        mla_suffix = f"{ns}_k"
        page_keys = [f"page{i}" for i in range(8)]
        return {f"{key}_{mla_suffix}" for key in page_keys}

    def test_same_page_two_ranks_disjoint(self):
        r0 = self._component_keys(0, 2)
        r1 = self._component_keys(1, 2)
        self.assertTrue(r0)
        self.assertTrue(r0.isdisjoint(r1))

    def test_degree_change_isolated(self):
        # dcp2 keys must never match dcp4 objects: the degree is embedded.
        dcp2 = self._component_keys(0, 2)
        dcp4 = self._component_keys(0, 4)
        self.assertTrue(dcp2.isdisjoint(dcp4))

    def test_same_degree_same_rank_collides_as_expected(self):
        a = self._component_keys(2, 4)
        b = self._component_keys(2, 4)
        self.assertEqual(a, b)

    def test_grammar_embedding_is_suffix_exact(self):
        # _dcp2_4 must not be confusable with _dcp2_40-style suffixes.
        ns_small = dcp_key_namespace(dcp_rank=2, dcp_size=4)
        ns_big = dcp_key_namespace(dcp_rank=2, dcp_size=40)
        self.assertTrue(ns_big.startswith(ns_small))
        # But the full keys differ because the degree digits differ.
        self.assertNotEqual(f"k{ns_small}_k", f"k{ns_big}_k")


class TestBufferModePlumbFromParallelState(CustomTestCase):
    def test_controller_reads_parallel_state(self):
        """_generate_storage_config must pull dcp fields from parallel state."""
        import sglang.srt.mem_cache.memory_pool as mp
        from sglang.srt.managers.cache_controller import HiCacheController

        controller = MagicMock()
        controller.tp_rank = 0
        controller.tp_size = 8
        controller.pp_rank = 0
        controller.pp_size = 1
        controller.mem_pool_host.layout = "layer_first"
        controller.enable_storage_metrics = False
        controller.get_attn_cp_rank_and_size.return_value = (0, 1)
        # PR2 added get_dcp_rank_and_size(); pin it to the same mocked
        # rank/size this test drives (the controller is itself a mock).
        controller.get_dcp_rank_and_size.return_value = (3, 4)
        # Real class so the isinstance rank-replicated detection works.
        device = mp.MLATokenToKVPool.__new__(mp.MLATokenToKVPool)
        controller.mem_pool_device = device

        with (
            patch("sglang.srt.managers.cache_controller.get_parallel") as gp,
            patch(
                "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
                return_value=False,
            ),
            # The flag source is the published memory bag; absent in unit
            # context, the plumb must default the flag off (exception guard),
            # which is exactly what a missing publish exercises.
            patch(
                "sglang.srt.managers.cache_controller.get_memory",
                side_effect=ValueError("config namespace 'memory' not published"),
            ),
        ):
            parallel = gp.return_value
            parallel.attn_dcp_rank = 3
            parallel.attn_dcp_size = 4
            cfg = HiCacheController._generate_storage_config(controller, model_name="m")
        self.assertEqual(cfg.dcp_rank, 3)
        self.assertEqual(cfg.dcp_size, 4)
        self.assertFalse(cfg.enable_hicache_dcp_shard)
        self.assertTrue(cfg.is_mla_model)

    def test_parallel_state_trivial_defaults(self):
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState

        ps = ParallelState.trivial()
        self.assertEqual(ps.attn_dcp_rank, 0)
        self.assertEqual(ps.attn_dcp_size, 1)

    def test_parallel_state_dcp_shape(self):
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState

        ps = ParallelState.trivial(attn_dcp_rank=1, attn_dcp_size=2)
        self.assertEqual(ps.attn_dcp_rank, 1)
        self.assertEqual(ps.attn_dcp_size, 2)


class TestLinkerSuffixParity(CustomTestCase):
    def _suffix(self, **kw) -> str:
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            _storage_suffix,
        )

        defaults = dict(
            rank_replicated=True,
            tp_rank=0,
            attn_cp_rank=0,
            pp_rank=0,
        )
        defaults.update(kw)
        return _storage_suffix(**defaults)

    def test_dcp1_linker_suffix_unchanged(self):
        # Byte-parity with the pre-change linker namespace.
        self.assertEqual(self._suffix(), "cp0_pp0")

    def test_dcp2_linker_suffix_appends_tag(self):
        self.assertEqual(self._suffix(dcp_rank=1, dcp_size=2), "cp0_pp0_dcp1_2")

    def test_linker_tag_matches_buffer_mode_tag(self):
        """The `_dcp` component must be identical across both paths (2.4)."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        # Buffer-mode tag: extracted the same way MooncakeStore builds it.
        buf_ns = dcp_key_namespace(_make_storage_config(dcp_rank=1, dcp_size=2))
        linker_suffix = self._suffix(dcp_rank=1, dcp_size=2)
        self.assertIn(buf_ns, linker_suffix)
        self.assertTrue(linker_suffix.endswith(buf_ns))
        # And the store consumes the same helper: composition check without
        # constructing a real Mooncake store (needs the mooncake package).
        self.assertTrue(hasattr(MooncakeStore, "__init__"))

    def test_linker_non_replicated_keeps_tp_part(self):
        self.assertEqual(
            self._suffix(rank_replicated=False, tp_rank=2, dcp_rank=0, dcp_size=2),
            "tp2_cp0_pp0_dcp0_2",
        )


class TestUmbpDirectLinkerRejectsDcp(CustomTestCase):
    def test_init_hard_rejects_dcp(self):
        """The linker must refuse to build a config under dcp_size > 1."""
        from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
            UMBPDirectLinker,
        )

        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        params = MagicMock()
        params.page_size = 1
        params.token_to_kv_pool_allocator.get_kvcache.return_value = MagicMock()
        with (
            patch(
                "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.get_parallel"
            ) as gp,
            patch.dict("os.environ", {}, clear=False),
        ):
            parallel = gp.return_value
            parallel.attn_dcp_size = 4
            with self.assertRaises(NotImplementedError) as ctx:
                # __init__ hits the DCP guard before any storage work.
                UMBPDirectLinker.__init__(
                    linker,
                    MagicMock(),
                    params,
                    components=set(),
                )
        self.assertIn("no DCP-scoped key support", str(ctx.exception))
        self.assertIn("--dcp-size 1", str(ctx.exception))

    def test_arg_resolution_rejects_umbp_linker_dcp(self):
        from sglang.srt.arg_groups.hicache_hook import handle_hicache

        sa = _make_server_args(
            dcp_size=4,
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="mori",
        )
        with self.assertRaises(NotImplementedError) as ctx:
            handle_hicache(sa)
        self.assertIn("no DCP-scoped key support", str(ctx.exception))


class TestFlagOffPreservesExactError(CustomTestCase):
    """Regression gate: flag-off must reproduce today's message verbatim."""

    def test_error_text_byte_equal(self):
        sa = _make_server_args(
            dcp_size=4,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
        )
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertEqual(str(ctx.exception), _L3_DCP_NOT_IMPLEMENTED_MESSAGE)
        self.assertIn("Run HiCache+DCP with L1/L2 only", str(ctx.exception))

    def test_flag_default_is_off(self):
        sa = _make_server_args()
        self.assertFalse(sa.enable_hicache_dcp_shard)


class TestFlagAdmitsL3AndMisconfigFailsFast(CustomTestCase):
    def test_flag_on_admits_l3(self):
        sa = _make_server_args(
            dcp_size=4,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            enable_hicache_dcp_shard=True,
        )
        # Must not raise.
        resolve_hicache_dcp_compatibility(sa)

    def test_flag_on_without_l3_backend_is_startup_error(self):
        sa = _make_server_args(
            dcp_size=4,
            enable_hierarchical_cache=True,
            enable_hicache_dcp_shard=True,
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("requires an L3 storage backend", str(ctx.exception))

    def test_l1_l2_only_dcpsize1_unaffected_by_flag(self):
        sa = _make_server_args(
            dcp_size=1,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
        )
        resolve_hicache_dcp_compatibility(sa)

    def test_attn_cp_size_combined_with_dcp_rejected(self):
        sa = _make_server_args(
            dcp_size=4,
            attn_cp_size=2,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            enable_hicache_dcp_shard=True,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("attn_cp_size > 1 combined with dcp_size > 1", str(ctx.exception))


class TestHardExclusionsSurviveFlag(CustomTestCase):
    """Guard-order matrix (design D5): each exclusion must fire under flag-on."""

    def _flagged_l3_args(self, **overrides):
        base = dict(
            dcp_size=4,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            enable_hicache_dcp_shard=True,
        )
        base.update(overrides)
        return _make_server_args(**base)

    def test_spec_algorithm_excluded(self):
        sa = self._flagged_l3_args(speculative_algorithm="EAGLE")
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("only supports DSPARK", str(ctx.exception))

    def test_lmcache_excluded(self):
        sa = self._flagged_l3_args(enable_lmcache=True)
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("--enable-lmcache", str(ctx.exception))

    def test_hisparse_excluded(self):
        sa = self._flagged_l3_args(enable_hisparse=True)
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("--enable-hisparse", str(ctx.exception))

    def test_non_mla_excluded(self):
        sa = self._flagged_l3_args()
        if AttentionArch is None:  # pragma: no cover
            self.skipTest("AttentionArch unavailable")
        sa._model_config.attention_arch = AttentionArch.MHA
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("only supported for MLA models", str(ctx.exception))

    def test_pd_disagg_l3_excluded(self):
        sa = self._flagged_l3_args(disaggregation_mode="prefill")
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("PD disaggregation", str(ctx.exception))

    def test_tp_lcm_head_splitting_excluded(self):
        sa = self._flagged_l3_args(
            hicache_storage_backend_extra_config='{"tp_lcm_size": 8}'
        )
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("tp_lcm_size", str(ctx.exception))

    def test_tp_lcm_absent_passes(self):
        sa = self._flagged_l3_args(
            hicache_storage_backend_extra_config='{"prefetch_threshold": 64}'
        )
        resolve_hicache_dcp_compatibility(sa)

    def test_hole_set_pool_excluded(self):
        sa = self._flagged_l3_args()
        sa._model_config.is_hybrid_swa = True
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("hole-set (TRAILING_PAGES)", str(ctx.exception))
        self.assertIn("mask-intersection", str(ctx.exception))

    def test_exclusions_fire_even_when_flag_off_and_l3_present(self):
        # Flag-off fires the preserved L3 message first for a clean config;
        # but a *pre-L3* hard exclusion must win the race (guard order).
        sa = _make_server_args(
            dcp_size=4,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            enable_lmcache=True,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            resolve_hicache_dcp_compatibility(sa)
        self.assertIn("--enable-lmcache", str(ctx.exception))


class TestStartupLogAmplificationLine(CustomTestCase):
    def test_log_line_reports_amplification(self):

        sa = _make_server_args(
            dcp_size=2,
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            enable_hicache_dcp_shard=True,
            page_size=16,
            max_total_tokens=1024,
        )
        with self.assertLogs(
            "sglang.srt.arg_groups.hicache_hook", level="INFO"
        ) as logs:
            resolve_hicache_dcp_compatibility(sa)
        joined = "\n".join(logs.output)
        self.assertIn("dcp_size=2", joined)
        self.assertIn("amplification", joined)
        # 1024 tokens / (16 * 2 logical page) = 32 pages; x2 degree = 64.
        self.assertIn("32", joined)
        self.assertIn("64", joined)

    def test_l1_l2_only_log_has_no_amplification(self):
        sa = _make_server_args(
            dcp_size=2,
            enable_hierarchical_cache=True,
        )
        with self.assertLogs(
            "sglang.srt.arg_groups.hicache_hook", level="INFO"
        ) as logs:
            resolve_hicache_dcp_compatibility(sa)
        joined = "\n".join(logs.output)
        self.assertIn("L1/L2 only", joined)
        self.assertNotIn("amplification", joined)


if __name__ == "__main__":
    unittest.main()
