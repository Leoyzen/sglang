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


if __name__ == "__main__":
    unittest.main()
