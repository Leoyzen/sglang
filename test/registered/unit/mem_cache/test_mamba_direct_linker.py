"""Unit tests for the Mamba direct external linker path."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    MambaDevicePoolEntry,
    _build_mamba_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_NUM_MAMBA_LAYERS = 2
_NUM_FULL_LAYERS = 3
_SLOTS = 8


def _fake_mamba_pool():
    """Device MambaPool stand-in: temporal (L, slots, ...) + conv list."""
    return SimpleNamespace(
        mamba_cache=SimpleNamespace(
            temporal=torch.zeros((_NUM_MAMBA_LAYERS, _SLOTS, 4), dtype=torch.uint8),
            conv=[torch.zeros((_NUM_MAMBA_LAYERS, _SLOTS, 3), dtype=torch.uint8)],
        )
    )


def _fake_hybrid_kvcache():
    """HybridLinearKVPool stand-in with MHA full pool buffers."""

    class _FullPool:
        k_buffer = [torch.zeros((16, 5), dtype=torch.uint8) for _ in range(_NUM_FULL_LAYERS)]
        v_buffer = [torch.zeros((16, 7), dtype=torch.uint8) for _ in range(_NUM_FULL_LAYERS)]

    kvcache = SimpleNamespace(
        full_attention_layer_id_mapping={gid: i for i, gid in enumerate([0, 2, 4])},
        full_kv_pool=_FullPool(),
    )
    return kvcache


def _fake_params():
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_pool=_fake_mamba_pool(),
            mamba_map={1: 0, 3: 1},  # global mamba layer ids -> pool layer idx
        )
    )


class TestMambaDevicePoolGroup(CustomTestCase):
    def test_build_group_shape_and_order(self):
        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=1, params=_fake_params())

        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.MAMBA})
        self.assertEqual(group.num_layers, 5)  # full {0,2,4} | mamba {1,3}
        self.assertFalse(group.rank_replicated)
        self.assertEqual(group.sources[PoolName.KV], PoolName.KV)
        self.assertEqual(group.sources[PoolName.MAMBA], PoolName.MAMBA)

        mamba = group.entry_map[PoolName.MAMBA]
        self.assertIsInstance(mamba, MambaDevicePoolEntry)
        self.assertEqual(mamba.page_size, 1)
        self.assertFalse(mamba.packed)
        # temporal (2 layers) first, then conv_0 (2 layers): 2 component groups
        self.assertEqual(len(mamba.components), 2)
        self.assertEqual(len(mamba.components[0]), _NUM_MAMBA_LAYERS)
        self.assertEqual(len(mamba.components[1]), _NUM_MAMBA_LAYERS)
        self.assertEqual(mamba.temporal_state_elem_size, 4)
        self.assertEqual(len(mamba.conv_buffer), 1)

    def test_mamba_layer_mapping_over_sorted_union(self):
        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=1, params=_fake_params())
        mamba = group.entry_map[PoolName.MAMBA]
        # Union of {0,2,4} and {1,3} = [0,1,2,3,4]; mamba gids 1,3 -> ranks 1,3
        self.assertEqual(mamba.layer_mapping, {1: 0, 3: 1})

    def test_key_pointer_parity_per_layer(self):
        """Keys expanded per (slot, component-group) must align 1:1 with the
        per-layer pointers emitted by get_prepared_layer_range_meta."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=1, params=_fake_params())
        mamba = group.entry_map[PoolName.MAMBA]

        store = SimpleNamespace()
        # Bind the unbound method with a minimal `self` substitute providing
        # the attributes the key composer reads.
        composer = MooncakeStore._get_hybrid_page_component_keys
        keys = ["slot-key"]
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=keys,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        class _Self:
            registered_pools = {PoolName.MAMBA: mamba}
            mha_suffix = "r0_tp0"

        component_keys, multiplier = composer(_Self(), keys, transfer)
        # temporal (elem size 4 > 0) + conv_0 => 2 keys per slot
        self.assertEqual(component_keys, ["slot-key_r0_tp0_temporal", "slot-key_r0_tp0_conv_0"])
        self.assertEqual(multiplier, 2)

        # Local transfer layer 1 -> gid 1 -> mamba pool layer 0;
        # local layer 3 -> gid 3 -> mamba pool layer 1.
        temporal_view = mamba.components[0][0]
        conv_view = mamba.components[1][0]
        ptrs, sizes, offsets = mamba.get_prepared_layer_range_meta(mamba.prepare_locations(torch.tensor([3])), 1)
        self.assertEqual(ptrs[0][0], temporal_view[3].data_ptr())
        self.assertEqual(ptrs[1][0], conv_view[3].data_ptr())
        temporal_view_hi = mamba.components[0][1]
        conv_view_hi = mamba.components[1][1]
        ptrs, _, _ = mamba.get_prepared_layer_range_meta(mamba.prepare_locations(torch.tensor([3])), 3)
        self.assertEqual(ptrs[0][0], temporal_view_hi[3].data_ptr())
        self.assertEqual(ptrs[1][0], conv_view_hi[3].data_ptr())

    def test_offload_pack_uniform_groups(self):
        """get_page_buffer_meta emits layers×groups per slot; mooncake's
        _pack_multi_buffer_meta must re-group them per key uniformly."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=1, params=_fake_params())
        mamba = group.entry_map[PoolName.MAMBA]

        indices = torch.tensor([2, 3])
        ptrs, sizes = mamba.get_page_buffer_meta(indices)
        # 2 slots × (2 temporal + 2 conv) layer views = 8 flat ptrs
        self.assertEqual(len(ptrs), 8)

        component_keys = [f"k{row}_{suffix}" for row in range(2) for suffix in ("temporal", "conv_0")]
        packed_ptrs, packed_sizes = MooncakeStore._pack_multi_buffer_meta(component_keys, ptrs, sizes)
        self.assertEqual(len(packed_ptrs), 4)  # 2 keys × 2 slots
        # First key of slot 0 holds both temporal layers, contiguous
        self.assertEqual(packed_ptrs[0], ptrs[0:2])
        self.assertEqual(packed_ptrs[1], ptrs[2:4])
        self.assertEqual(packed_ptrs[2], ptrs[4:6])
        self.assertEqual(packed_ptrs[3], ptrs[6:8])


class _FakeAllocator:
    def __init__(self):
        self.next_slot = 0
        self.live = set()
        self.freed = []

    def alloc(self, n):
        if self.next_slot >= 4:
            return None
        slots = torch.arange(self.next_slot, self.next_slot + n, dtype=torch.int64)
        self.next_slot += n
        self.live.update(slots.tolist())
        return slots

    def free(self, slots):
        self.freed.extend(slots.tolist())
        self.live.difference_update(slots.tolist())


class TestMambaComponentTransfers(CustomTestCase):
    def _component(self, allocator):
        component = MambaComponent.__new__(MambaComponent)
        component.component_type = ComponentType.MAMBA
        component.cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_allocator=allocator),
            evict_for_alloc=lambda params: None,
        )
        return component

    def test_offload_uses_node_boundary_key(self):
        component = self._component(_FakeAllocator())
        node = SimpleNamespace(
            hash_value=["h0", "h1", "h2"],
            component_data={ComponentType.MAMBA: SimpleNamespace(value=torch.tensor([5], dtype=torch.int64))},
        )
        transfer = component.build_external_linker_transfer(LinkerTransferPhase.OFFLOAD, node, None)
        self.assertEqual(transfer.name, PoolName.MAMBA)
        self.assertEqual(transfer.keys, ["h2"])
        self.assertEqual(transfer.device_indices.tolist(), [5])
        self.assertEqual(transfer.hit_policy, PoolHitPolicy.TRAILING_PAGES)

    def test_lookup_reduces_to_trailing_hash(self):
        component = self._component(_FakeAllocator())
        transfer = component.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, ["h0", "h1", "h2"])
        self.assertEqual(transfer.keys, ["h2"])
        self.assertIsNone(transfer.device_indices)
        self.assertEqual(transfer.hit_policy, PoolHitPolicy.TRAILING_PAGES)

    def test_load_allocates_with_eviction_fallback(self):
        allocator = _FakeAllocator()
        allocator.next_slot = 4  # exhausted
        evictions = []

        component = MambaComponent.__new__(MambaComponent)
        component.component_type = ComponentType.MAMBA
        component.cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_allocator=allocator),
            evict_for_alloc=lambda params: evictions.append(params) or setattr(allocator, "next_slot", 0),
        )
        transfer = component.build_external_linker_transfer(LinkerTransferPhase.LOAD, None, ["h0", "h1"])
        self.assertEqual(transfer.keys, ["h1"])
        self.assertEqual(transfer.device_indices.tolist(), [0])
        self.assertEqual(len(evictions), 1)
        self.assertEqual(evictions[0].num_tokens, 0)
        self.assertEqual(evictions[0].mamba_num, 1)

    def test_load_returns_none_when_starved(self):
        allocator = _FakeAllocator()
        allocator.next_slot = 4  # exhausted, eviction cannot help

        component = MambaComponent.__new__(MambaComponent)
        component.component_type = ComponentType.MAMBA
        component.cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_allocator=allocator),
            evict_for_alloc=lambda params: None,
        )
        transfer = component.build_external_linker_transfer(LinkerTransferPhase.LOAD, None, ["h0"])
        self.assertIsNone(transfer)

    def test_update_load_passes_through(self):
        component = self._component(_FakeAllocator())
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["h0"],
            device_indices=torch.tensor([3], dtype=torch.int64),
        )
        for phase in ExternalLinkerLoadPhase:
            result = component.update_external_linker_load(
                phase,
                req=SimpleNamespace(),
                full_transfer=PoolTransfer(name=PoolName.KV, keys=["h0"]),
                transfer=transfer,
                prefix_len=8,
                insert_result=InsertResult(prefix_len=0),
            )
            self.assertIs(result, transfer)


class TestUpdateLoadCommitKeepsMamba(CustomTestCase):
    def test_commit_keeps_mamba_whole(self):
        cache = SimpleNamespace(page_size=2)
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache

        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["h0", "h1"],
            device_indices=torch.tensor([9], dtype=torch.int64),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        full_transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["h0", "h1"],
            device_indices=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        )
        mamba_component = MambaComponent.__new__(MambaComponent)
        mamba_component.component_type = ComponentType.MAMBA
        mamba_component.update_external_linker_load = lambda phase, req, full_transfer, transfer, prefix_len, **kwargs: transfer
        full_component = SimpleNamespace(component_type=ComponentType.FULL)
        full_component.update_external_linker_load = lambda phase, req, full_transfer, transfer, prefix_len, **kwargs: transfer

        result = wrapper._update_load(
            ExternalLinkerLoadPhase.COMMIT,
            SimpleNamespace(),
            [(full_component, full_transfer), (mamba_component, mamba_transfer)],
            prefix_len=4,
            insert_result=InsertResult(
                prefix_len=0,
                # No MAMBA adopted ranges recorded by the tree.
                adopted_ranges={ComponentType.FULL: [(2, 4)]},
            ),
            canonical_full=torch.tensor([0, 1, 12, 13], dtype=torch.int64),
        )

        self.assertIn(mamba_transfer, result)
        self.assertEqual(mamba_transfer.keys, ["h0", "h1"])
        self.assertEqual(mamba_transfer.device_indices.tolist(), [9])
        # FULL still went through adopted-range filtering.
        self.assertEqual(full_transfer.keys, ["h1"])
        self.assertEqual(full_transfer.device_indices.tolist(), [12, 13])

    def test_resolve_transfers_keeps_mamba_trailing_policy(self):
        """The MAMBA entry is self-sourced: its TRAILING_PAGES policy must
        survive resolve_transfers (unlike KV-derived entries, forced to
        ALL_PAGES)."""
        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=1, params=_fake_params())
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                keys=["a"],
                device_indices=torch.tensor([0, 1]),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["a"],
                device_indices=torch.tensor([3]),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]

        resolved = group.resolve_transfers(transfers)

        by_name = {transfer.name: transfer for transfer in resolved}
        self.assertEqual(by_name[PoolName.KV].hit_policy, PoolHitPolicy.ALL_PAGES)
        self.assertEqual(by_name[PoolName.MAMBA].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(by_name[PoolName.MAMBA].host_indices.tolist(), [3])
        # Self-keyed identity: the resolved MAMBA transfer probes its own key
        # space, not the KV page-hash array.
        self.assertEqual(by_name[PoolName.MAMBA].probe_source, PoolName.MAMBA)
        self.assertEqual(by_name[PoolName.KV].probe_source, PoolName.KV)


class TestSelfKeyedBatchExists(CustomTestCase):
    """batch_exists_v2 evaluates a self-keyed TRAILING_PAGES transfer over its
    own boundary keys, independent of the KV page count."""

    def _store_stub(self, mamba_entry):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        exist_calls = []

        def fake_batch_exist(keys):
            exist_calls.append(list(keys))
            # KV page keys ("page{i}_...") always exist; a self-keyed slot
            # exists iff its boundary key names the hit boundary ("boundary_hit").
            return [1 if (k.startswith("boundary_hit_") or not k.startswith("boundary_")) else 0 for k in keys]

        kv_entry = SimpleNamespace(
            page_size=64,
            conv_buffer=None,
            temporal_state_elem_size=0,
        )
        store = MooncakeStore.__new__(MooncakeStore)
        store.registered_pools = {PoolName.MAMBA: mamba_entry, PoolName.KV: kv_entry}
        store.mha_suffix = "r0_tp0"
        store.mla_suffix = "r0_tp0"
        store.config_prefix = None
        store._use_group_semantics = False
        store.mem_pool_host = SimpleNamespace(kv_buffer=None)
        store._batch_exist = fake_batch_exist
        store._exist_calls = exist_calls
        return store

    def test_self_keyed_probe_uses_transfer_keys(self):
        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=64, params=_fake_params())
        mamba = group.entry_map[PoolName.MAMBA]

        store = self._store_stub(mamba)
        kv_keys = [f"page{i}" for i in range(5)]
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                keys=kv_keys,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["boundary_hit"],  # one node-boundary slot, exists
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
                probe_source=PoolName.MAMBA,
            ),
        ]
        result = type(store).batch_exists_v2(store, kv_keys, transfers)
        # MAMBA was probed with its own key, not the 5 KV page keys.
        self.assertIn(
            ["boundary_hit_r0_tp0_temporal", "boundary_hit_r0_tp0_conv_0"],
            store._exist_calls,
        )
        # Boundary present -> final KV page is restorable.
        self.assertIn(5, result.restorable_prefix_pages)

    def test_self_keyed_miss_caps_restorable_prefix(self):
        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=64, params=_fake_params())
        mamba = group.entry_map[PoolName.MAMBA]

        store = self._store_stub(mamba)
        kv_keys = [f"page{i}" for i in range(5)]
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                keys=kv_keys,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["boundary_miss"],  # mamba state absent
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
                probe_source=PoolName.MAMBA,
            ),
        ]
        result = type(store).batch_exists_v2(store, kv_keys, transfers)
        self.assertEqual(result.restorable_prefix_pages, [])

    def test_offload_lookup_key_parity(self):
        """The OFFLOAD key (node.hash_value[-1]) and the LOOKUP boundary key
        (last tail hash) must be the identical string for the same boundary."""
        component = TestMambaComponentTransfers._component(self, _FakeAllocator())
        node = SimpleNamespace(
            hash_value=["h0", "h1", "h2"],
            component_data={ComponentType.MAMBA: SimpleNamespace(value=torch.tensor([7], dtype=torch.int64))},
        )
        offload = component.build_external_linker_transfer(LinkerTransferPhase.OFFLOAD, node, None)
        # A request whose device-uncached tail ends at this node's boundary
        # carries the same page-hash chain, so its last tail hash == h2.
        lookup = component.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, ["h0", "h1", "h2"])
        self.assertEqual(offload.keys, ["h2"])
        self.assertEqual(lookup.keys, offload.keys)


def _fake_draft_pool():
    """MTP draft HybridLinearKVPool wrapper: full_kv_pool carries MHA buffers
    plus a DSA index sidecar with one buffer per draft layer."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    draft_layer_num = 1

    # The packed-draft assembler reads FLAT per-layer tensor lists off the
    # draft pool object itself (pool.kv_buffer / pool.index_k_with_scale_buffer),
    # mirroring how the pure-DSA group consumes them. Packed drafts are
    # DSA-style: ONE latent buffer per layer, 1:1 with the index sidecar.
    class _DraftPool:
        kv_buffer = [torch.zeros((16, 9), dtype=torch.uint8) for _ in range(draft_layer_num)]
        index_k_with_scale_buffer = [torch.zeros((4, 11), dtype=torch.uint8) for _ in range(draft_layer_num)]
        use_dsa = True
        page_size = 1

    pool = _DraftPool()
    assert isinstance(HybridLinearKVPool, object)  # keep import meaningful
    return pool


class TestMambaPackedDraftMapping(CustomTestCase):
    """The packed-draft path must key draft depths by ACTUAL mapping keys.

    Regression: _with_packed_draft_mapping assumed contiguous local keys
    (0..N-1) and crashed with KeyError: 0 on the mamba-hybrid path, where
    full_attention_layer_id_mapping is keyed by interleaved global layer ids
    (layer 0 is a mamba layer). It also flattened component GROUPS as single
    buffers, crashing _row_count with AttributeError ('list' has no shape).
    """

    def test_packed_draft_group_resolves_all_layers(self):
        params = _fake_params()
        params.mtp_draft_device_pools = (draft_pool := _fake_draft_pool(),)
        group = _build_mamba_device_pool_group(_fake_hybrid_kvcache(), page_size=1, params=params, mtp_draft_device_pools=(draft_pool,))

        kv = group.entry_map[PoolName.KV]
        # All flattened components must be real tensors (not nested lists).
        for buffer in kv.kv_buffer:
            self.assertIsInstance(buffer, torch.Tensor)

        # Target full layers are global ids {0, 2, 4}; the single draft layer
        # attaches to the first (lowest-id) target layer.
        mapping = kv.layer_mapping
        self.assertIn(0, mapping)
        self.assertEqual(mapping[0], (0, 3))  # (target comp 0, device layer 3)
        self.assertEqual(mapping[2], 1)
        self.assertEqual(mapping[4], 2)

        # Every mapping key (global layer ids; the packed key resolves both its
        # target and draft buffers) must resolve to non-zero pointer views.
        locations = kv.prepare_locations(torch.tensor([2]))
        for key in kv.layer_mapping:
            meta = kv.get_prepared_layer_range_meta(locations, key)
            self.assertIsNotNone(meta, f"transfer layer {key} unresolved")
            ptrs, sizes, offsets = meta
            self.assertTrue(all(p != 0 for p in [x for row in ptrs for x in row]))

    def test_identity_key_path_unchanged(self):
        """Pure-DSA-style identity mappings must keep their key space."""
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _with_packed_draft_mapping,
        )

        mapping = _with_packed_draft_mapping(
            {l: l for l in range(4)},
            target_device_layer_num=4,
            draft_layer_num=1,
        )
        self.assertEqual(mapping, {0: (0, 4), 1: 1, 2: 2, 3: 3})


if __name__ == "__main__":
    unittest.main()
