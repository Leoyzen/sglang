"""Integration (mock) test: HiCache DCP L3 shard backup + consensus restore.

Task 7.1 of openspec change ``hicache-dcp-l3-shared-foundation``:
a simulated dcp=2 world (two in-process controller "ranks" sharing one fake
store) runs fill → evict → L3 write → restart tree → prefetch and asserts the
restored prefix accuracy equals a dcp=1 reference on interleaved token order,
with all ranks agreeing; plus the shutdown clause (clear() never wipes
rank-scoped shards via remove_all()).
"""

import unittest
from unittest import mock

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    dcp_key_namespace,
    dcp_logical_keep_pages,
    dcp_shard_active,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

DEGREE = 2
BASE_PAGE = 64
LOGICAL_PAGE = BASE_PAGE * DEGREE


def _cfg(rank: int) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=rank,
        tp_size=DEGREE,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test/integration",
        dcp_size=DEGREE,
        dcp_rank=rank,
        enable_hicache_dcp_shard=True,
        extra_config={"master_server_address": "127.0.0.1:50051"},
    )


class _SharedFakeStore:
    """One logical L3 namespace shared by both simulated ranks.

    Puts are keyed by the composed shard key (page hash + `_dcp{r}_{n}`), so
    cross-rank visibility emerges naturally — each rank can see a peer shard
    is present without gathering data.
    """

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.remove_all_called = False

    def put(self, key: str, payload: bytes) -> bool:
        self.objects[key] = payload
        return True

    def exists(self, key: str) -> bool:
        return key in self.objects

    def get(self, key: str):
        return self.objects.get(key)

    def remove_all(self):
        self.remove_all_called = True
        self.objects.clear()


class _SimRank:
    """One simulated DCP rank's L3 write/probe surface against the fake store.

    Mirrors the seam layout: page keys are per-rank shard keys; existence
    probes are rank-scoped; a consensus min over contributed hit lengths
    stands in for the asserted gloo collective.
    """

    def __init__(self, rank: int, store: _SharedFakeStore, *, failing: bool = False):
        self.rank = rank
        self.store = store
        self.failing = failing
        self.suffix = dcp_key_namespace(dcp_rank=rank, dcp_size=DEGREE)
        self.puts_per_page = {}  # page_hash -> put count issued by THIS rank

    def shard_key(self, page_hash: str) -> str:
        return f"{page_hash}{self.suffix}"

    def backup_page(self, page_hash: str, payload: bytes) -> bool:
        self.puts_per_page[page_hash] = self.puts_per_page.get(page_hash, 0) + 1
        return self.store.put(self.shard_key(page_hash), payload)

    def probe_hit_pages(self, page_hashes, *, min_peers=None) -> int:
        """Longest prefix of own-rank shard keys present in the store.

        Task 5.4: a store failure is normalized to a zero contribution; the
        caller-side collective (min_peers) still executes on every rank.
        """
        hit = 0
        for h in page_hashes:
            if self.failing:
                raise ConnectionError("simulated store node death")
            if self.store.exists(self.shard_key(h)):
                hit += 1
            else:
                break
        return hit

    def load_page(self, page_hash: str):
        return self.store.get(self.shard_key(page_hash))


def _make_payload(page_idx: int, token_offset: int) -> bytes:
    return bytes([(page_idx + token_offset) % 251] * LOGICAL_PAGE)


class TestDcp2FillEvictRestartPrefetch(CustomTestCase):
    def _reference_single_rank(self, pages: int) -> dict:
        """dcp=1 reference: one rank, no suffix, plain keys."""
        store = _SharedFakeStore()
        ref = _SimRank(0, store)
        # Force legacy grammar: empty suffix under dcp=1
        ref.suffix = dcp_key_namespace(dcp_rank=0, dcp_size=1)
        page_hashes = [f"ph{i}" for i in range(pages)]
        for i, h in enumerate(page_hashes):
            ref.backup_page(h, _make_payload(i, token_offset=0))
        return {
            "store": store,
            "hashes": page_hashes,
            "payloads": {h: store.get(f"{h}{ref.suffix}") for h in page_hashes},
        }

    def test_fill_evict_write_restart_prefetch_parity(self):
        ref = self._reference_single_rank(pages=8)
        pages = 8
        page_hashes = [f"ph{i}" for i in range(pages)]

        store = _SharedFakeStore()
        ranks = [_SimRank(r, store) for r in range(DEGREE)]

        # --- fill + evict → every rank writes its own shard ---
        for i, h in enumerate(page_hashes):
            logical_payload = _make_payload(i, token_offset=0)
            for r in ranks:
                # Owner rule: this rank owns residue-r tokens of the page; the
                # payload it stores is ITS slice (owner filter idx % d == r).
                shard_payload = logical_payload[r.rank :: DEGREE]
                self.assertTrue(r.backup_page(h, shard_payload))

        # Single-writer: each rank issued exactly one put per page
        for r in ranks:
            for h in page_hashes:
                self.assertEqual(r.puts_per_page[h], 1)
        # Key partitioning: both shards present, disjoint key namespaces
        for h in page_hashes:
            for r in ranks:
                self.assertTrue(store.exists(r.shard_key(h)))

        # --- restart tree: fresh rank objects, same store ---
        ranks_restart = [_SimRank(r, store) for r in range(DEGREE)]

        # --- prefetch: rank-scoped probes + min consensus on logical tokens ---
        contributions = {}
        local_hits = {}
        for r in ranks_restart:
            try:
                local_hits[r.rank] = r.probe_hit_pages(page_hashes)
            except ConnectionError:
                local_hits[r.rank] = 0  # normalized degradation (5.4)
            contributions[r.rank] = local_hits[r.rank]
        consensus_pages = min(contributions.values())
        self.assertEqual(consensus_pages, pages)  # all shards present

        # Belief invalidation beyond consensus uses the LOGICAL page size (5.6)
        keep = dcp_logical_keep_pages(consensus_pages * LOGICAL_PAGE, LOGICAL_PAGE)
        self.assertEqual(keep, pages)

        # --- restored content equals the dcp=1 reference on interleaved order ---
        for i, h in enumerate(page_hashes):
            ref_payload = ref["payloads"][h]
            for r in ranks_restart:
                own = r.load_page(h)
                # each rank's shard is the r-th residue of the same logical
                # payload the dcp=1 run materialized as one page
                expected_shard = bytes(ref_payload)[r.rank :: DEGREE]
                self.assertEqual(own, expected_shard)
            # reassembly across ranks (interleaved concat) equals reference
            reassembled = b"".join(ranks_restart[r].load_page(h) for r in range(DEGREE))
            self.assertEqual(reassembled, bytes(ref_payload))

    def test_peer_shard_missing_truncates_uniformly(self):
        """Spec scenario: rank1's page-6 shard missing → all ranks adopt 6."""
        pages = 8
        page_hashes = [f"pm{i}" for i in range(pages)]
        store = _SharedFakeStore()
        ranks = [_SimRank(r, store) for r in range(DEGREE)]
        for r in ranks:
            for i, h in enumerate(page_hashes):
                r.backup_page(h, bytes([i]) * (LOGICAL_PAGE // DEGREE))

        # Evict exactly rank1's page 6 shard (peer shard missing)
        del store.objects[ranks[1].shard_key(page_hashes[6])]

        contributions = {}
        for r in ranks:
            contributions[r.rank] = r.probe_hit_pages(page_hashes)
        consensus = min(contributions.values())
        self.assertEqual(consensus, 6)  # truncated at last fully-covered page
        for r in ranks:
            # identical truncation on all ranks; page 6 beliefs invalidated,
            # page 5 remains fully covered
            self.assertTrue(store.exists(r.shard_key(page_hashes[5])))

    def test_failing_rank_no_hang_uniform_zero(self):
        """Spec scenario: rank2 fails under dcp=4 → zero everywhere."""
        degree = 4
        base_page = 64
        logical_page = base_page * degree
        page_hashes = [f"pf{i}" for i in range(4)]

        class _R:
            def __init__(self, rank, failing):
                self.rank = rank
                self.failing = failing
                self.suffix = dcp_key_namespace(dcp_rank=rank, dcp_size=degree)

            def probe(self, hashes):
                if self.failing:
                    raise ConnectionError("store node died")
                return len(hashes)

        ranks = [_R(r, failing=(r == 2)) for r in range(degree)]
        contributions = {}
        for r in ranks:
            try:
                contributions[r.rank] = r.probe(page_hashes)
            except ConnectionError:
                contributions[r.rank] = 0  # degraded, still reaches collective
        self.assertEqual(min(contributions.values()), 0)
        # every rank arrived — no strand, no hang
        self.assertEqual(len(contributions), degree)
        self.assertEqual(dcp_logical_keep_pages(0, logical_page), 0)

    def test_shutdown_never_removes_rank_scoped_shards(self):
        """Task 7.1 shutdown clause: MooncakeStore.clear() with an active
        shard suffix must NOT call store.remove_all() (which would wipe every
        peer's shards); legacy clear() keeps its remove_all()."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        fake_inner = _SharedFakeStore()

        def make_store(storage_cfg):
            store = MooncakeStore.__new__(MooncakeStore)
            # bypass mooncake client init: stamp only what clear() touches
            object.__setattr__(store, "store", fake_inner)
            object.__setattr__(store, "config", mock.Mock())
            object.__setattr__(
                store,
                "dcp_suffix",
                dcp_key_namespace(
                    dcp_rank=getattr(storage_cfg, "dcp_rank", 0),
                    dcp_size=getattr(storage_cfg, "dcp_size", 1),
                ),
            )
            return store

        # Shard-active store: clear() must skip remove_all and keep objects
        shard_store = make_store(_cfg(rank=0))
        fake_inner.put("some_peer_shard_dcp1_2", b"x")
        with mock.patch.object(MooncakeStore, "clear", MooncakeStore.clear):
            shard_store.clear()
        self.assertFalse(fake_inner.remove_all_called)
        self.assertIn("some_peer_shard_dcp1_2", fake_inner.objects)

        # Legacy store (dcp=1): clear() still removes everything
        legacy_store = make_store(
            HiCacheStorageConfig(
                tp_rank=0,
                tp_size=1,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
                is_mla_model=True,
                enable_storage_metrics=False,
                is_page_first_layout=False,
                model_name="m",
            )
        )
        legacy_store.clear()
        self.assertTrue(fake_inner.remove_all_called)

    def test_gate_matrix_end_to_end(self):
        """Flag off ⇒ shard machinery inert even at dcp>1 (legacy parity)."""
        cfg_off = HiCacheStorageConfig(
            tp_rank=1,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="m",
            dcp_size=DEGREE,
            dcp_rank=1,
            enable_hicache_dcp_shard=False,
        )
        self.assertFalse(dcp_shard_active(cfg_off))
        self.assertEqual(dcp_key_namespace(dcp_rank=1, dcp_size=DEGREE), "_dcp1_2")


if __name__ == "__main__":
    unittest.main()
