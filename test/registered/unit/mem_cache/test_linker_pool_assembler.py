"""Unit tests for external-linker device pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDevicePoolEntry(CustomTestCase):
    def test_sparse_multi_component_layer_ranges(self):
        k0 = torch.zeros((8, 3), dtype=torch.uint8)
        k2 = torch.zeros((8, 5), dtype=torch.uint8)
        v0 = torch.zeros((8, 7), dtype=torch.uint8)
        v2 = torch.zeros((8, 11), dtype=torch.uint8)
        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[k0, k2], [v0, v2]],
            layer_mapping={0: 0, 2: 1},
            page_size=2,
            rows_are_pages=False,
            packed=False,
        )

        indices = torch.tensor([0, 1, 4, 5])
        locations = pool.prepare_locations(indices)
        self.assertEqual(locations, [0, 4])
        pointers, sizes = pool.get_page_buffer_meta(indices)
        self.assertEqual(
            pointers,
            [buffer[row].data_ptr() for row in locations for buffer in (k0, k2, v0, v2)],
        )
        self.assertEqual(sizes, [6, 10, 14, 22] * 2)
        self.assertIsNone(pool.get_prepared_layer_range_meta(locations, 1))

        pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
        self.assertEqual(
            pointers,
            [
                [k2[0].data_ptr()],
                [v2[0].data_ptr()],
                [k2[4].data_ptr()],
                [v2[4].data_ptr()],
            ],
        )
        self.assertEqual(sizes, [[10], [22], [10], [22]])
        self.assertEqual(offsets, [[6], [14], [6], [14]])

    def test_rejects_invalid_pages_and_empty_buffers(self):
        with self.assertRaisesRegex(ValueError, "has no storage buffers"):
            DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[],
                layer_mapping={},
                page_size=2,
                rows_are_pages=False,
            )

        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((8, 3), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=False,
        )
        for indices, error in (
            (torch.tensor([0]), "multiple of page_size"),
            (torch.tensor([1, 2]), "aligned contiguous pages"),
            (torch.tensor([0, 2]), "aligned contiguous pages"),
            (torch.tensor([8, 9]), "exceeds buffer shapes"),
        ):
            with self.subTest(indices=indices.tolist()):
                with self.assertRaisesRegex(ValueError, error):
                    pool.prepare_locations(indices)


class TestDevicePoolGroup(CustomTestCase):
    def test_resolve_transfers_expands_physical_pools(self):
        entries = [
            SimpleNamespace(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices,
            ),
            SimpleNamespace(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices + 100,
            ),
        ]
        group = DevicePoolGroup(entries, num_layers=2, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["a", "b"],
            device_indices=torch.tensor([0, 1, 4, 5]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        resolved = group.resolve_transfers([transfer])

        self.assertEqual([item.name for item in resolved], [PoolName.KV, PoolName.INDEXER])
        self.assertEqual(resolved[0].host_indices.tolist(), [0, 1, 4, 5])
        self.assertEqual(resolved[1].host_indices.tolist(), [100, 101, 104, 105])
        self.assertTrue(all(item.hit_policy == PoolHitPolicy.ALL_PAGES for item in resolved))

    def test_partial_side_pool_requires_explicit_opt_in(self):
        entry = SimpleNamespace(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            translate_indices=lambda indices: indices + 100,
        )
        group = DevicePoolGroup([entry], num_layers=1, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=["b", "d"],
            device_indices=torch.tensor([20, 21, 24, 25]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        self.assertEqual(group.resolve_transfers([transfer]), [])
        resolved = group.resolve_transfers([transfer], allow_partial=True, allow_missing_kv=True)

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].name, PoolName.SWA)
        self.assertEqual(resolved[0].keys, ["b", "d"])
        self.assertEqual(resolved[0].host_indices.tolist(), [120, 121, 124, 125])
        self.assertEqual(resolved[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)


class TestHybridDevicePoolAssembler(CustomTestCase):
    def test_deepseek_v4_maps_sparse_sidecars(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
            )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = False
        kvcache.start_layer = 1
        kvcache.end_layer = 4
        kvcache.swa_page_size = 2
        kvcache.swa_kv_pool = SimpleNamespace(kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)])
        kvcache.c4_kv_pool = SimpleNamespace(kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)])
        kvcache.c4_indexer_kv_pool = SimpleNamespace(index_k_with_scale_buffer=[torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)])
        kvcache.c128_kv_pool = SimpleNamespace(kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)])
        kvcache.layer_mapping = [
            DeepSeekV4LayerItem(0, -1),
            DeepSeekV4LayerItem(4, 0),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 1),
        ]
        kvcache.compress_state_pools = [None, state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [
            None,
            state_pool(),
            None,
            state_pool(),
        ]

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(mtp_draft_device_pools=()),
            components={ComponentType.FULL, ComponentType.SWA},
        )

        self.assertEqual(group.num_layers, 3)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(
            set(group.entry_map),
            {
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            },
        )
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4], PoolName.KV)
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4_STATE], PoolName.SWA)

        c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
        pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
        self.assertEqual(len(pointers), 2)
        self.assertEqual(sizes, [5, 5])
        _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
        self.assertEqual(sizes, [[5]])
        self.assertEqual(offsets, [[5]])
        self.assertIsNone(c4_pool.get_prepared_layer_range_meta([0], 1))

    def test_dsa_uses_hybrid_assembler_strategy(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        def dsa_pool(kv_width, index_width):
            pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
            pool.page_size = 2
            pool.layer_num = 1
            pool.kv_buffer = [torch.zeros((8, kv_width), dtype=torch.uint8)]
            pool.index_key_cache = SimpleNamespace(buffer=[torch.zeros((4, index_width), dtype=torch.uint8)])
            return pool

        kvcache = dsa_pool(3, 7)
        kvcache.layer_num = 2
        kvcache.kv_buffer.append(torch.zeros((8, 5), dtype=torch.uint8))
        kvcache.index_key_cache.buffer.append(torch.zeros((4, 11), dtype=torch.uint8))
        draft_pools = (dsa_pool(13, 17), dsa_pool(19, 23))

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(mtp_draft_device_pools=draft_pools),
            components={ComponentType.FULL},
        )

        self.assertEqual(group.num_layers, 2)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.INDEXER})
        self.assertEqual(
            group.sources,
            {
                PoolName.KV: PoolName.KV,
                PoolName.INDEXER: PoolName.KV,
            },
        )
        _, sizes, offsets = group.entry_map[PoolName.KV].get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[6, 26]])
        self.assertEqual(offsets, [[0, 16]])
        _, sizes, offsets = group.entry_map[PoolName.INDEXER].get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[7, 17]])
        self.assertEqual(offsets, [[0, 18]])
        _, sizes, offsets = group.entry_map[PoolName.INDEXER].get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[11, 23]])
        self.assertEqual(offsets, [[7, 35]])

    def test_linker_requires_packed_draft(self):
        """Do not accept draft state that the linker would omit from storage."""
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        draft = SimpleNamespace(
            token_to_kv_pool=object(),
            model_config=SimpleNamespace(
                num_nextn_predict_layers=0,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            ),
        )
        target = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.EAGLE)
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target),
            _draft_model_runners=lambda: (draft,),
        )
        for linker_enabled, nextn_layers in (
            (False, 0),
            (True, 0),
            (False, 1),
            (True, 1),
        ):
            draft.model_config.num_nextn_predict_layers = nextn_layers
            with (
                self.subTest(linker=linker_enabled, nextn=nextn_layers),
                patch.object(
                    spec,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=not linker_enabled,
                        enable_unified_cache_external_linker=linker_enabled,
                    ),
                ),
            ):
                if linker_enabled and not nextn_layers:
                    with self.assertRaisesRegex(NotImplementedError, "only supports packed"):
                        spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(target.mtp_draft_device_pools, ())
                else:
                    plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(
                        plan.mode,
                        spec.HiCacheDraftMode.PACKED if nextn_layers else spec.HiCacheDraftMode.SIDECAR,
                    )
                    self.assertEqual(plan.device_pools, (draft.token_to_kv_pool,))
                    self.assertEqual(
                        target.mtp_draft_device_pools,
                        plan.device_pools if nextn_layers else (),
                    )

    def test_mamba_accepts_tree_page_size(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        kvcache.full_attention_layer_id_mapping = {0: 0}
        kvcache.full_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((16, 5), dtype=torch.uint8)]
        )
        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(
                    mamba_cache=SimpleNamespace(
                        temporal=torch.zeros((1, 4, 2), dtype=torch.uint8),
                        conv=[torch.zeros((1, 4, 3), dtype=torch.uint8)],
                    )
                ),
                mamba_map={0: 0},
            ),
            mtp_draft_device_pools=(),
        )
        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=params,
            components={ComponentType.FULL, ComponentType.MAMBA},
        )
        # The tree page size is carried by the KV entry; the MAMBA entry
        # stays slot-granular regardless.
        self.assertEqual(group.entry_map[PoolName.KV].page_size, 2)
        mamba = group.entry_map[PoolName.MAMBA]
        self.assertEqual(mamba.page_size, 1)
        # slot-granular rows: internal page_size==1 while the KV entry carries
        # the tree page size; _row_span == 1 means rows_are_pages semantics.
        self.assertEqual(mamba._row_span, 1)

    def _mamba_assembler_target(self):
        """Latent-per-layer (MLA/DSA-shaped) mamba-hybrid target — the ONLY
        shape the packed-draft mapping supports. Models the GLM-5.3-Flash
        family: one flat latent buffer per full-attention layer, so the
        flattened buffer count equals len(full_layer_mapping) and the packed
        tuple (target_comp, N + depth) indexes draft latents, not v-buffers."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        kvcache.full_attention_layer_id_mapping = {0: 0, 2: 1, 4: 2}
        kvcache.full_kv_pool = SimpleNamespace(
            # Flat latent list: 1 buffer per mapped layer (MLA/DSA layout).
            kv_buffer=[torch.zeros((16, 5), dtype=torch.uint8) for _ in range(3)],
        )
        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(
                    mamba_cache=SimpleNamespace(
                        temporal=torch.zeros((2, 8, 4), dtype=torch.uint8),
                        conv=[torch.zeros((2, 8, 3), dtype=torch.uint8)],
                    )
                ),
                mamba_map={1: 0, 3: 1},
            ),
            mtp_draft_device_pools=(),
        )
        return kvcache, params

    def test_mamba_packs_dsa_draft_pool_flat_buffers(self):
        """DSA-shaped draft pool (GLM-5.3-Flash NextN: DSATokenToKVPool with
        flat .kv_buffer + .index_k_with_scale_buffer) packs its KV rows into
        the KV entry and its index rows into the INDEXER entry."""
        kvcache, params = self._mamba_assembler_target()
        params.mtp_draft_device_pools = (
            SimpleNamespace(
                page_size=1,
                kv_buffer=[torch.zeros((16, 9), dtype=torch.uint8)],
                index_k_with_scale_buffer=[torch.zeros((4, 11), dtype=torch.uint8)],
            ),
        )
        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=1,
            params=params,
            components={ComponentType.FULL, ComponentType.MAMBA},
        )
        # Draft index rows only land when the TARGET carries its own DSA
        # index sidecar AND the tree page_size is 64 (see the INDEXER entry
        # gating in _build_mamba_device_pool_group); this latent target at
        # page_size=1 has none, so only KV + MAMBA exist.
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.MAMBA})
        kv = group.entry_map[PoolName.KV]
        # Draft depth 0 attaches to the first mapped target layer (gid 0);
        # device layer 3 is the draft latent (3 target latents + depth 0).
        self.assertEqual(kv.layer_mapping[0], (0, 3))
        self.assertEqual(kv.layer_mapping[2], 1)
        _, sizes, _ = kv.get_prepared_layer_range_meta(kv.prepare_locations(torch.tensor([0])), 0)
        # Layer 0 resolves TWO buffers: target latent + draft latent.
        self.assertEqual(len(sizes[0]), 2)
        # The packed draft pointer must be the DRAFT buffer, not a v-buffer:
        # with an MHA-shaped target this index would silently resolve into
        # the target's k/v split (regression pinned by
        # test_mamba_mha_layout_rejects_packed_draft).
        draft_buffer = params.mtp_draft_device_pools[0].kv_buffer[0]
        self.assertIs(kv.kv_buffer[3], draft_buffer)

    def test_mamba_packs_hybrid_wrapper_draft_pool_without_flat_attrs(self):
        """Regression: a mamba-family NextN draft (Qwen3Next / NemotronH /
        Kimi / Bailing) hands over the HybridLinearKVPool WRAPPER, which has
        NO .kv_buffer / .index_k_with_scale_buffer (everything delegates to
        full_kv_pool). The assembler must unwrap the wrapper's flat latent
        rows instead of crashing with AttributeError, and a sidecar-less
        draft must be accepted (parity only enforced when both sides exist)."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache, params = self._mamba_assembler_target()
        wrapper = HybridLinearKVPool.__new__(HybridLinearKVPool)
        wrapper.full_kv_pool = SimpleNamespace(
            # Flat per-layer latent list, the same shape the packed group
            # consumes (DSA-style single buffer per draft layer).
            kv_buffer=[torch.zeros((16, 9), dtype=torch.uint8)],
        )
        params.mtp_draft_device_pools = (wrapper,)
        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=1,
            params=params,
            components={ComponentType.FULL, ComponentType.MAMBA},
        )
        # No INDEXER entry: the wrapper draft carries no index sidecar.
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.MAMBA})
        kv = group.entry_map[PoolName.KV]
        self.assertEqual(kv.layer_mapping[0], (0, 3))
        _, sizes, _ = kv.get_prepared_layer_range_meta(kv.prepare_locations(torch.tensor([0])), 0)
        self.assertEqual(len(sizes[0]), 2)

        # Component-GROUP full pool (nested per-layer lists) flattens too.
        kvcache2, params2 = self._mamba_assembler_target()
        wrapper2 = HybridLinearKVPool.__new__(HybridLinearKVPool)
        wrapper2.full_kv_pool = SimpleNamespace(
            kv_buffer=[[torch.zeros((16, 5), dtype=torch.uint8)], [torch.zeros((16, 7), dtype=torch.uint8)]],
        )
        params2.mtp_draft_device_pools = (wrapper2,)
        group2 = resolve_hybrid_device_pool_group(
            kvcache=kvcache2,
            page_size=1,
            params=params2,
            components={ComponentType.FULL, ComponentType.MAMBA},
        )
        kv2 = group2.entry_map[PoolName.KV]
        # 2 draft components attach to the first two mapped target layers.
        self.assertEqual(kv2.layer_mapping[0], (0, 3))
        self.assertEqual(kv2.layer_mapping[2], (1, 4))
        self.assertEqual(kv2.layer_mapping[4], 2)

    def test_mamba_mha_layout_rejects_packed_draft(self):
        """An MHA-layout target (separate k_buffer/v_buffer groups → 2N flat
        buffers) must fail LOUDLY at assembly instead of mis-indexing: the
        packed tuple (target_comp, N + depth) would resolve into the target's
        v-buffers (index 3 = v[0]) instead of the draft buffers, corrupting
        every packed restore. Startup must refuse, not mis-store."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache, params = self._mamba_assembler_target()
        # Swap the latent list for the k/v split: 3 k + 3 v = 6 flat buffers
        # for 3 mapped layers.
        kvcache.full_kv_pool = SimpleNamespace(
            k_buffer=[torch.zeros((16, 5), dtype=torch.uint8) for _ in range(3)],
            v_buffer=[torch.zeros((16, 7), dtype=torch.uint8) for _ in range(3)],
        )
        params.mtp_draft_device_pools = (
            SimpleNamespace(
                page_size=1,
                kv_buffer=[torch.zeros((16, 9), dtype=torch.uint8)],
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "MHA-layout mamba-hybrid targets"):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=1,
                params=params,
                components={ComponentType.FULL, ComponentType.MAMBA},
            )
        # The no-draft path stays untouched: MHA-layout targets without draft
        # pools must keep assembling (their unpumped mapping is index-exact).
        params.mtp_draft_device_pools = ()
        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=1,
            params=params,
            components={ComponentType.FULL, ComponentType.MAMBA},
        )
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.MAMBA})

    def test_hicache_draft_plan_reaches_build_kv_cache(self):
        """The EAGLE draft pools must flow from the plan into tree-cache params.

        Pins the scheduler's init seam (ordering verified identical to
        upstream main): Scheduler.__init__ runs init_model_worker() ->
        init_memory_pools() -> draft_worker.init_hicache_draft_plan()
        (scheduler.py:554/989) BEFORE kv_cache_builder.build_kv_cache()
        (scheduler.py:560). build_kv_cache must then read
        tp_worker.model_runner.mtp_draft_device_pools -- the very attribute
        _build_hicache_draft_plan mutated -- into CacheInitParams, so the
        direct linker's strategy sees non-empty draft pools. A regression in
        either hop silently drains mamba P2 draft packing (spec_accept_rate
        collapse with empty draft pools at the assembler).
        """
        from sglang.srt.configs.model_config import ModelImpl
        from sglang.srt.runtime_context import get_memory, reset_context
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.mem_cache import kv_cache_builder

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
        self.addCleanup(reset_context)

        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"],
                get_text_config=lambda: SimpleNamespace(),
            ),
            linear_attn_registry_result=None,
            _resolved_model_impl=ModelImpl.SGLANG,
            is_multimodal=False,
        )
        draft_pool = SimpleNamespace(
            page_size=1,
            kv_buffer=[torch.zeros((16, 9), dtype=torch.uint8)],
        )
        target_runner = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            mtp_draft_device_pools=(),
            model_config=model_config,
        )
        draft_runner = SimpleNamespace(
            token_to_kv_pool=draft_pool,
            model_config=SimpleNamespace(
                num_nextn_predict_layers=1,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLMNextN"]),
            ),
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target_runner),
            draft_worker=SimpleNamespace(draft_runners=[draft_runner]),
            _draft_model_runners=lambda: (draft_runner,),
        )
        # Real-system aliasing: the draft worker's target_worker IS the
        # scheduler's tp_worker, so model_runner here must be the SAME object
        # the plan mutates (scheduler.py:944 target_worker=self.tp_worker).
        tp_worker = SimpleNamespace(
            is_hybrid_swa=False,
            model_runner=target_runner,
            get_memory_pool=lambda: (object(), SimpleNamespace(get_kvcache=lambda: object())),
        )

        captured = {}

        def fake_create_tree_cache(ctx):
            captured["ctx"] = ctx
            return SimpleNamespace(cache_controller=None)

        with (
            get_memory().override(enable_unified_cache_external_linker=True),
            patch.object(
                kv_cache_builder,
                "create_tree_cache",
                side_effect=fake_create_tree_cache,
            ),
            patch.object(kv_cache_builder, "maybe_register_hicache_draft"),
        ):
            plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
            self.assertEqual(plan.mode, spec.HiCacheDraftMode.PACKED)
            self.assertEqual(plan.device_pools, (draft_pool,))
            # Hop 1: the plan injected the draft pools onto the target runner.
            self.assertEqual(target_runner.mtp_draft_device_pools, (draft_pool,))

            # Hop 2: build_kv_cache forwards them into CacheInitParams exactly
            # as the scheduler's call site does.
            kv_cache_builder.build_kv_cache(
                server_args=ServerArgs(model_path="dummy", page_size=1),
                model_config=model_config,
                tp_worker=tp_worker,
                page_size=1,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                attn_tp_cpu_group=None,
                tp_cpu_group=None,
                attn_cp_cpu_group=None,
                enable_metrics=False,
                enable_kv_cache_events=False,
                ps=SimpleNamespace(pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1, tp_size=1, tp_rank=0),
                tp_group=None,
                pp_group=SimpleNamespace(cpu_group=None),
                enable_hierarchical_cache=False,
                hicache_draft_plan=plan,
            )
        params = captured["ctx"].params
        self.assertEqual(params.mtp_draft_device_pools, (draft_pool,))


if __name__ == "__main__":
    unittest.main()
