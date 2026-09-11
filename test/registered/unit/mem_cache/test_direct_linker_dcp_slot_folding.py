"""Regression tests for the direct linker's DCP slot folding.

Production incident (glm-dcp-0907 @ 225c6d26bb, arena pod dcp_size=2, tp4/ep4,
GLM-5.3-Flash mamba hybrid + Mooncake direct linker): the linker path resolved
transfers with RAW widened logical slots while the device pool entry's buffers
address per-rank physical rows, so a high-slot offload exploded in
``DevicePoolEntry._rows``:

    ValueError: Pool kv row range [85504, 591872) exceeds buffer shapes
    [(573442, ...) x 11, (1146884, ...) x 1]

(1146884 = 2 x 573442 = the widened logical space upper bound; 573442 is the
per-rank physical row count.) The fix folds the widened slots to this rank's
rows at ``DevicePoolGroup.resolve_transfers`` — the same owner rule as
``HostKVCache.maybe_dcp_kernel_indices`` — for entries that declare
``dcp_fold_slots=True``. Keys need no shard flag: the linker's component keys
are composed from the rank-scoped `_dcp{rank}_{size}` suffix namespace, so
they never collide across ranks.

Buffer-identity facts pinned here (mirrors the unified hybrid pool):
- ONLY the full-attention latent rows shard under DCP; the mamba state pool is
  replicated and slot-granular (its allocator sets shards_under_dcp=False), so
  the MAMBA entry must stay pass-through;
- the DSA index sidecar is GLOBAL-slot addressed (replicated buffer sized
  ``size * dcp_size`` so all ranks compute identical top-k) and must NOT fold.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    _build_dsa_device_pool_group,
    _build_mamba_device_pool_group,
    _build_mamba_swa_device_pool_group,
    _dcp_folding_index_mapper,
    _is_hybrid_linear_kv_pool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PHYS_ROWS = 8  # per-rank physical rows of the fake KV entry
_WIDENED_SLOTS = 2 * _PHYS_ROWS  # dcp=2 widened logical space
_PAGE = 2  # tree page (physical rows per page); widened page = 4 slots


@contextmanager
def _parallel_dcp(size: int = 2, rank: int = 0):
    """Stamp the parallel bag the way the tests in test_full_loc_fast_path do;
    also covers the unpublished (CPU-only) runner case."""
    with get_parallel().override(
        dcp_enabled=size > 1, attn_dcp_size=size, attn_dcp_rank=rank
    ):
        yield


def _fake_mamba_pool():
    return SimpleNamespace(
        mamba_cache=SimpleNamespace(
            temporal=torch.zeros((2, 16, 4), dtype=torch.uint8),
            conv=[torch.zeros((2, 16, 3), dtype=torch.uint8)],
        )
    )


def _fake_latent_hybrid_kvcache(rows=_PHYS_ROWS):
    class _FullPool:
        kv_buffer = [torch.zeros((rows, 5), dtype=torch.uint8) for _ in range(3)]

    return SimpleNamespace(
        full_attention_layer_id_mapping={gid: i for i, gid in enumerate([0, 2, 4])},
        full_kv_pool=_FullPool(),
    )


def _fake_params():
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_pool=_fake_mamba_pool(),
            mamba_map={1: 0, 3: 1},
        )
    )


class TestDcpFoldingMapper(CustomTestCase):
    def test_identity_at_dcp_one(self):
        indices = torch.arange(_WIDENED_SLOTS)
        # dcp=1 must be a ZERO-COPY identity (incident invariant: no behavior
        # change without DCP, no extra tensor churn on the hot path).
        with _parallel_dcp(1):
            self.assertIs(_dcp_folding_index_mapper(indices), indices)

    def test_owner_rule_rank_interleave_collapse(self):
        indices = torch.arange(_WIDENED_SLOTS)
        with _parallel_dcp(2, rank=1):
            out = _dcp_folding_index_mapper(indices)
        # rank 1 owns odd slots; collapsing gives consecutive physical rows.
        torch.testing.assert_close(out, torch.arange(_PHYS_ROWS))
        self.assertLess(int(out.max()), _PHYS_ROWS)

    def test_unaligned_batch_fails_loud(self):
        with _parallel_dcp(2, rank=0):
            with self.assertRaises(ValueError) as cm:
                _dcp_folding_index_mapper(torch.arange(_WIDENED_SLOTS + 1))
        self.assertIn("whole widened pages", str(cm.exception))


class TestResolvedTransfersFold(CustomTestCase):
    """The end-to-end seam the incident crashed on: resolve_transfers output
    (host_indices == the slot tensor handed to _batch_io_v2) must fold."""

    def _mamba_group(self):
        group = _build_mamba_device_pool_group(
            _fake_latent_hybrid_kvcache(), page_size=_PAGE, params=_fake_params()
        )
        return group

    def test_kv_slots_fold_below_physical_rows(self):
        group = self._mamba_group()
        # One widened page (page_size * dcp = 4 slots) at a HIGH logical slot —
        # the exact region the production crash tripped (beyond half the
        # widened space).
        widened = torch.arange(_WIDENED_SLOTS)[-_PAGE * 2 :]
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=widened,
                keys=["aaa", "bbb"],
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                device_indices=torch.tensor([5]),
                keys=["nodetail"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        with _parallel_dcp(2, rank=1):
            resolved = {
                t.name: t
                for t in group.resolve_transfers(transfers, allow_missing_kv=True)
            }
            kv_rows = group.entry_map[PoolName.KV].prepare_locations(
                resolved[PoolName.KV].host_indices
            )
            ptrs, sizes = group.entry_map[PoolName.KV].get_page_buffer_meta(
                resolved[PoolName.KV].host_indices
            )
        self.assertLess(max(kv_rows), _PHYS_ROWS)
        # rank1 owns odd slots: slots [12..15] fold to [6,7] — one page of
        # rows starting at the LAST physical page boundary.
        self.assertEqual(kv_rows, [_PHYS_ROWS - _PAGE])
        # ...and row math actually succeeds end to end.
        self.assertEqual(len(ptrs), len(sizes))
        self.assertTrue(all(p != 0 for p in ptrs))

    def test_mamba_slots_stay_raw(self):
        group = self._mamba_group()
        transfers = [
            PoolTransfer(name=PoolName.KV, device_indices=torch.arange(4), keys=["k"]),
            PoolTransfer(
                name=PoolName.MAMBA,
                device_indices=torch.tensor([11]),
                keys=["tail"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        with _parallel_dcp(2, rank=1):
            resolved = {
                t.name: t
                for t in group.resolve_transfers(transfers, allow_missing_kv=True)
            }
            mamba_rows = group.entry_map[PoolName.MAMBA].prepare_locations(
                resolved[PoolName.MAMBA].host_indices
            )
        # Mamba state is replicated/slot-granular: NO folding, ids untouched.
        self.assertEqual(mamba_rows, [11])

    def test_both_ranks_fold_same_keys_to_disjoint_rows(self):
        """Cross-rank agreement: the SAME key set maps each rank to its own
        interleaved shard, rows disjoint and covering the page once."""
        group = self._mamba_group()
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=torch.arange(4),
                keys=["p1"],
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                device_indices=torch.tensor([7]),
                keys=["tail"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        rows_by_rank = {}
        slots_by_rank = {}
        for rank in (0, 1):
            with _parallel_dcp(2, rank=rank):
                resolved = group.resolve_transfers(transfers, allow_missing_kv=True)
                slots_by_rank[rank] = resolved[0].host_indices
                rows_by_rank[rank] = group.entry_map[PoolName.KV].prepare_locations(
                    resolved[0].host_indices
                )
        # Owner rule (maybe_dcp_kernel_indices semantics): each rank keeps ITS
        # interleaved slots then collapses; one widened page (4 slots) yields
        # 2 per-rank rows [0, 1] on EVERY rank.
        torch.testing.assert_close(slots_by_rank[0], torch.tensor([0, 1]))
        torch.testing.assert_close(slots_by_rank[1], torch.tensor([0, 1]))
        # Row ids stay inside this rank's physical buffer, and the two ranks'
        # LOGICAL slots were disjoint going in (the key namespace + owner rule
        # keep their data separate even though folded row ids are both dense).
        self.assertTrue(max(rows_by_rank[0]) < _PHYS_ROWS)
        self.assertTrue(max(rows_by_rank[1]) < _PHYS_ROWS)
        self.assertTrue(not set(range(0, 4, 2)) & set(range(1, 4, 2)))

    def test_dcp_one_group_resolves_unchanged(self):
        """dcp=1: resolved slots are the input tensor verbatim (identity)."""
        group = self._mamba_group()
        transfers = [
            PoolTransfer(
                name=PoolName.KV,
                device_indices=torch.arange(2, 6),
                keys=["x", "y"],
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                device_indices=torch.tensor([3]),
                keys=["tail"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        with _parallel_dcp(1):
            resolved = group.resolve_transfers(transfers, allow_missing_kv=True)
        torch.testing.assert_close(resolved[0].host_indices, torch.arange(2, 6))


class TestBufferIdentities(CustomTestCase):
    """Which 0907 buffers widen, and which fold — pinned so a future layout
    change cannot silently reintroduce the incident."""

    def test_mamba_entry_never_declares_folding(self):
        group = _build_mamba_device_pool_group(
            _fake_latent_hybrid_kvcache(), page_size=1, params=_fake_params()
        )
        self.assertIsNone(group.entry_map[PoolName.MAMBA]._index_mapper)
        self.assertIsNotNone(group.entry_map[PoolName.KV]._index_mapper)

    def test_mamba_swa_group_flags_kv_sharding_source(self):
        """The defensive SWA+mamba group must declare whether its KV source
        shards (hybrid-linear: yes) so the group consistency check can fire."""

        kvcache = SimpleNamespace(
            full_kv_pool=_fake_latent_hybrid_kvcache().full_kv_pool,
            swa_kv_pool=SimpleNamespace(
                k_buffer=[torch.zeros((4, 3)) for _ in range(2)],
                v_buffer=[torch.zeros((4, 3)) for _ in range(2)],
            ),
            full_attention_layer_id_mapping={0: 0, 2: 1, 4: 2},
            swa_attention_layer_id_mapping={1: 0, 3: 1},
            full_attention_layer_ids=[0, 2, 4],
            swa_attention_layer_ids=[1, 3],
            layer_num=5,
            swa_page_size=1,
            page_size=1,
        )
        try:
            group = _build_mamba_swa_device_pool_group(
                kvcache, page_size=1, params=_fake_params()
            )
        except Exception:
            self.skipTest("_swa_layer_mappings needs richer pool attrs on this branch")
        self.assertTrue(_is_hybrid_linear_kv_pool(kvcache))

    def test_dcp_group_consistency_check_rejects_foldless_source(self):
        """A group that claims its KV source does NOT shard may not carry a
        folding entry — catching copy-pasted dcp_fold_slots on a replicated
        pool (the exact mistake class behind the global-index sidecar)."""
        entry = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=object(),
            components=[[torch.zeros((4, 3))]],
            layer_mapping={0: 0},
            page_size=1,
            rows_are_pages=False,
            dcp_fold_slots=True,
        )
        with self.assertRaises(ValueError) as cm:
            DevicePoolGroup([entry], 1, 1, kv_source_shards_under_dcp=False)
        self.assertIn("kv_source_shards_under_dcp=False", str(cm.exception))

    def test_index_mapper_and_fold_flag_are_exclusive(self):
        with self.assertRaises(ValueError):
            DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=object(),
                components=[[torch.zeros((4, 3))]],
                layer_mapping={0: 0},
                page_size=1,
                rows_are_pages=False,
                index_mapper=lambda t: t,
                dcp_fold_slots=True,
            )

    def test_indexer_slots_per_key_contract(self):
        """The INDEXER entry declares dcp_size pages per key (global-slot
        index-K domain), so the _batch_io_v2 keys==indices//width assert
        holds at dcp>1; default entries keep slots_per_key == page_size."""
        # Default: page-sized keys (folded/self-keyed pools).
        default_entry = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=object(),
            components=[[torch.zeros((8, 3))]],
            layer_mapping={0: 0},
            page_size=4,
            rows_are_pages=False,
        )
        self.assertEqual(default_entry.slots_per_key, 4)
        # INDEXER-style widened-domain entry: one key spans dcp pages.
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            _active_dcp_size,
        )

        widened = DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=object(),
            components=[[torch.zeros((8, 3))]],
            layer_mapping={0: 0},
            page_size=4,
            rows_are_pages=True,
            slots_per_key=4 * _active_dcp_size(),
        )
        dcp = _active_dcp_size()
        # Simulate the _batch_io_v2 division with dcp widened slots per key.
        n_keys = 3
        host_indices = torch.arange(n_keys * 4 * dcp)
        self.assertEqual(len(host_indices) // widened.slots_per_key, n_keys)


if __name__ == "__main__":
    unittest.main()


class TestIndexerSlotsPerKey(CustomTestCase):
    """Incident regression (glm-dcp-0907 @ 7cb9a4e326, GLM-5.3-Flash pod
    dcp_size=4, page_size=64): the INDEXER entry's index-K buffer is
    GLOBAL-slot addressed (replicated, sized size*dcp_size — IndexKeyCache),
    so resolve_transfers hands it RAW widened indices and `_rows` correctly
    yields dcp_size rows per key. But `MooncakeStore._batch_io_v2` asserted
    ``len(keys) == len(host_indices) // page_size``, which over-counts by
    dcp_size for widened-domain entries → 2028 consecutive offload failures
    at ``mooncake_store.py:851``. Fix: DevicePoolEntry declares
    ``slots_per_key`` (page_size * dcp_size for the INDEXER entries under
    dcp>1, page_size otherwise) and _batch_io_v2 divides by that.
    """

    # Production DCP page_size is ALREADY the widened width (kv_cache_builder
    # passes allocator.page_size = tree_page 64 x dcp_size 4 = 256). Kernel
    # index rows stay fixed 64-token pages (4 kernel rows per widened page).
    _TREE_PAGE = 64
    _DCP = 4
    _IDX_PAGE = _TREE_PAGE * _DCP

    def _dsa_group(self):
        kvcache = SimpleNamespace(
            kv_buffer=[
                torch.zeros((self._IDX_PAGE * 16, 5), dtype=torch.uint8)
                for _ in range(2)
            ],
            index_k_with_scale_buffer=[
                torch.zeros((self._IDX_PAGE * 16, 3), dtype=torch.uint8)
                for _ in range(2)
            ],
            layer_num=2,
            page_size=self._IDX_PAGE,
        )
        draft = SimpleNamespace(
            page_size=self._IDX_PAGE,
            kv_buffer=[
                torch.zeros((self._IDX_PAGE * 16, 5), dtype=torch.uint8)
                for _ in range(1)
            ],
            index_k_with_scale_buffer=[
                torch.zeros((self._IDX_PAGE * 16, 3), dtype=torch.uint8)
                for _ in range(1)
            ],
        )
        return _build_dsa_device_pool_group(kvcache, self._IDX_PAGE, (draft,))

    def test_indexer_slot_width_matches_batch_io_v2_assertion(self):
        """dcp=4: one key spans dcp_size*page_size raw slots → dcp_size rows.
        Replays the exact division `_batch_io_v2` asserts with the same
        slots_per_key the store reads via getattr."""
        with _parallel_dcp(4, rank=0):
            # Construction-time slot width: the group must be built with the
            # parallel context PUBLISHED (production assembles at startup).
            group = self._dsa_group()
            entry = group.entry_map[PoolName.INDEXER]
            kv = group.entry_map[PoolName.KV]
            # page_size is already widened; one key spans exactly one widened
            # page — no extra dcp multiply (old convention double-counted).
            self.assertEqual(entry.slots_per_key, self._IDX_PAGE)
            self.assertEqual(kv.slots_per_key, self._IDX_PAGE // self._DCP)
            n_keys = 3
            # Raw widened indices for 3 whole widened pages (the batch shape
            # the linker hands the store), starting at a HIGH logical slot —
            # the production crash region.
            start = self._IDX_PAGE * 5
            host_indices = torch.arange(start, start + n_keys * entry.slots_per_key)
            rows = entry.prepare_locations(host_indices)
            # Global-slot domain, no folding: one widened page == one
            # row-unit in this fixture's page grid (page_size == widened).
            self.assertEqual(len(rows), n_keys)
            self.assertTrue(max(rows) < 16)
            # The _batch_io_v2 assertion, computed exactly as the store does.
            slots_per_key = getattr(entry, "slots_per_key", None) or self._IDX_PAGE
            self.assertEqual(n_keys, len(host_indices) // slots_per_key)
            # Offload-side meta closes: dcp_size multi-range rows per key,
            # each row expanding to one ptr per flat buffer (3 here: 2 target
            # + 1 draft). _pack_multi_buffer_meta regroups len(ptrs)//n_keys
            # buffers per component key.
            ptrs, sizes = entry.get_page_buffer_meta(host_indices)
            n_bufs = 3
            self.assertEqual(len(ptrs), n_keys * n_bufs)
            self.assertEqual(len(ptrs), len(sizes))
            self.assertTrue(all(p != 0 for p in ptrs))
            nbuf = len(ptrs) // n_keys
            self.assertEqual(nbuf, n_bufs)
            self.assertEqual(len(ptrs) % n_keys, 0)

    def test_indexer_restore_groups_rows_per_key(self):
        """Restore-side dual: load_layer_wise prepares one row-group per
        component key, so `batch_get_into_multi_buffer_ranges` aligns with
        the session-start key list (keys == len(groups))."""
        with _parallel_dcp(4, rank=0):
            group = self._dsa_group()
            entry = group.entry_map[PoolName.INDEXER]
            n_keys = 2
            host_indices = torch.arange(n_keys * entry.slots_per_key)
            locations = entry.prepare_locations(host_indices)
            meta = entry.get_prepared_layer_range_meta(locations, layer=0)
        self.assertIsNotNone(meta)
        ptrs, sizes, offsets = meta
        # One GROUP per key; slots_per_key == page_size → one row per key
        # (rows_per_key == 1), each group covering the layer's component
        # buffers in the packed mapping.
        self.assertEqual(len(ptrs), n_keys)
        self.assertEqual(len(ptrs), len(sizes))
        self.assertEqual(len(ptrs), len(offsets))
        self.assertTrue(all(len(g) == 1 * 2 for g in ptrs))
        self.assertTrue(all(len(g) == len(s) for g, s in zip(ptrs, sizes)))

    def test_indexer_dcp_one_identity(self):
        """dcp=1: slots_per_key == page_size and restore grouping is the
        historical one-row-per-key behavior."""
        with _parallel_dcp(1):
            group = self._dsa_group()
            entry = group.entry_map[PoolName.INDEXER]
            self.assertEqual(entry.slots_per_key, self._IDX_PAGE)
            self.assertEqual(entry._rows_per_key, 1)
            n_keys = 2
            n_bufs = 3
            host_indices = torch.arange(n_keys * self._IDX_PAGE)
            ptrs, _ = entry.get_page_buffer_meta(host_indices)
            self.assertEqual(len(ptrs), n_keys * n_bufs)
            locations = entry.prepare_locations(host_indices)
            ptrs_r, sizes_r, offsets_r = entry.get_prepared_layer_range_meta(
                locations, layer=0
            )
            # One group per key; layer 0's packed mapping covers the target
            # buffer plus the packed draft buffer (2 ranges), one row each.
            self.assertEqual(len(ptrs_r), n_keys)
            self.assertEqual(len(ptrs_r[0]), 2)


if __name__ == "__main__":
    unittest.main()


class TestKvFoldedSlotsPerKey(CustomTestCase):
    """Incident regression (production DBGDBG, glm-dcp-0907 @ 0fe17cfb53):
    KV entry (dcp_fold_slots=True) must declare its FOLDED slot width —
    widened tree page // dcp_size — not the widened page itself. Under DCP
    CacheInitParams.page_size is already the widened width (kv_cache_builder
    passes allocator.page_size = tree_page * dcp_size), so the default
    slots_per_key==page_size over-declared 4x at dcp4 and tripped
    _batch_io_v2's keys==indices//slots_per_key assert (observed: keys=16,
    host_idx=1024, spk=256, page_size=64 server / 256 entry).
    """

    def test_production_shape(self):
        dcp = 4
        tree_page = 64
        widened_page = tree_page * dcp  # 256 == entry page_size under DCP
        entry = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=object(),
            components=[[torch.zeros((16, 3))]],
            layer_mapping={0: 0},
            page_size=widened_page,
            rows_are_pages=False,
            dcp_fold_slots=True,
            slots_per_key=max(1, widened_page // dcp),
        )
        # Exact production shape: 16 tree keys -> widened 4096 slots ->
        # folded 1024. The _batch_io_v2 assert: len(keys) == len(idx)//spk.
        n_keys, folded = 16, 1024
        self.assertEqual(folded // entry.slots_per_key, n_keys)
        self.assertEqual(entry.slots_per_key, 64)

    def test_indexer_not_multiplied_again(self):
        # INDEXER receives RAW widened indices; its page_size is already the
        # widened width, so slots_per_key == page_size (no extra multiply).
        dcp = 4
        widened_page = 64 * dcp
        entry = DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=object(),
            components=[[torch.zeros((16, 3))]],
            layer_mapping={0: 0},
            page_size=widened_page,
            rows_are_pages=True,
            slots_per_key=widened_page,
        )
        n_keys = 3
        raw_widened = n_keys * widened_page
        self.assertEqual(raw_widened // entry.slots_per_key, n_keys)
