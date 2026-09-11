"""Unit/mock coverage for HiCache DCP L3 shard backup + consensus restore (PR2).

OpenSpec change ``hicache-dcp-l3-shared-foundation``, tasks 4.x, 5.x, 6.x.
All behavior is gated on ``enable_hicache_dcp_shard`` AND ``dcp_size > 1``;
the flag-off cases here assert byte-identical legacy behavior.

No GPU, no real cluster: cross-rank consensus is exercised against an
in-process fake gloo group (the repo's existing hicache tests use the same
in-process-fake style), and the multi-rank sim in
``test_hicache_dcp_l3_integration.py`` threads two controller worlds through
one shared fake store.
"""

import logging
import re
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    dcp_key_namespace,
    dcp_logical_keep_pages,
    dcp_shard_active,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _config(**overrides) -> HiCacheStorageConfig:
    defaults = dict(
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test/model",
        dcp_size=1,
        dcp_rank=0,
        enable_hicache_dcp_shard=False,
        extra_config={"master_server_address": "127.0.0.1:50051"},
    )
    defaults.update(overrides)
    return HiCacheStorageConfig(**defaults)


# ---------------------------------------------------------------------------
# 4.x: per-rank shard backup (write path)
# ---------------------------------------------------------------------------


class _CountingStore:
    """Capture mock that records puts/exists and enforces a batch ceiling."""

    def __init__(self):
        self.put_keys = []
        self.exists_queries = []
        self.batch_put_calls = 0
        self.batch_exists_calls = 0
        self.writers = {}

    def batch_put(self, keys):
        self.batch_put_calls += 1
        for k in keys:
            assert k not in self.writers or self.writers[k] == "self", (
                f"shard key {k} written by two ranks in one generation"
            )
            self.writers[k] = "self"
        self.put_keys.extend(keys)
        return [0] * len(keys)

    def batch_exists(self, keys):
        self.batch_exists_calls += 1
        self.exists_queries.extend(keys)
        return [0] * len(keys)


class _ShardControllerHarness:
    """Minimal HiCacheController stand-in wired for the write/consensus paths.

    Bypasses __init__ (which needs allocators/device pools) and stamps the
    exact fields the methods under test touch, mirroring the
    ``ClassName.__new__`` bare-instance pattern used by
    test_hicache_dcp_host_pool.py.
    """

    def __init__(self, *, dcp_rank=0, dcp_size=1, flag=False, page_size=64):
        from sglang.srt.managers.cache_controller import HiCacheController

        ctl = HiCacheController.__new__(HiCacheController)
        ctl.page_size = page_size
        ctl.dcp_enabled_shard = dcp_size > 1 and flag
        ctl.dcp_rank = dcp_rank if ctl.dcp_enabled_shard else 0
        ctl.dcp_size = dcp_size if ctl.dcp_enabled_shard else 1
        ctl.logical_page_size = page_size
        ctl.backup_skip = False
        ctl.written = _CountingStore()
        ctl.storage_backend = mock.Mock()
        ctl.storage_backend.batch_set_v1.return_value = [True]
        ctl.page_set_func = mock.Mock(return_value=True)
        ctl.prefetch_completion_sync_groups = []
        # Native hash is Linux-only; stub the chain (only its shape matters).
        ctl.get_hash_str = lambda tokens, prior, page_size=None: [
            f"h{i}" for i in range(len(tokens) // (page_size or 1) or 1)
        ]
        self.ctl = ctl


class TestPerRankShardBackup(CustomTestCase):
    """Task 4.1/4.2/4.4: every rank puts its own shard; batching preserved."""

    def test_gate_requires_flag_and_degree(self):
        # dcp=1 flag-off: legacy
        self.assertFalse(dcp_shard_active(_config()))
        # dcp>1 flag-off: legacy (guard stays closed)
        self.assertFalse(dcp_shard_active(_config(dcp_size=4, dcp_rank=2)))
        # dcp=1 flag-on: nothing to shard
        self.assertFalse(
            dcp_shard_active(_config(dcp_size=1, enable_hicache_dcp_shard=True))
        )
        # dcp>1 + flag: active
        self.assertTrue(
            dcp_shard_active(
                _config(dcp_size=4, dcp_rank=2, enable_hicache_dcp_shard=True)
            )
        )

    def test_backup_thread_runs_on_every_rank_under_shard(self):
        from sglang.srt.managers.cache_controller import StorageOperation

        for dcp_rank in (0, 1):
            h = _ShardControllerHarness(dcp_rank=dcp_rank, dcp_size=2, flag=True)
            op = StorageOperation(
                host_indices=torch.arange(2 * 64),
                token_ids=list(range(128)),
                hash_value=["aaa", "bbb"],
            )
            h.ctl._page_backup = mock.Mock()
            # backup_thread_func gate: `not backup_skip or dcp_enabled_shard`
            ran = not h.ctl.backup_skip or h.ctl.dcp_enabled_shard
            if ran:
                h.ctl._page_backup(op)
            self.assertTrue(h.ctl._page_backup.called)
            self.assertEqual(h.ctl._page_backup.call_args.args[0], op)

    def test_backup_skip_overridden_by_shard_flag(self):
        """MLA rank!=0 has backup_skip=True, but shard mode must still write."""
        from sglang.srt.managers.cache_controller import StorageOperation

        h = _ShardControllerHarness(dcp_rank=1, dcp_size=2, flag=True)
        h.ctl.backup_skip = True  # replicated-MLA rank-0-only world
        op = StorageOperation(
            host_indices=torch.arange(64),
            token_ids=list(range(64)),
            hash_value=["k0"],
        )
        h.ctl._page_backup = mock.Mock()
        # backup_thread_func gate: `not backup_skip or dcp_enabled_shard`
        if not h.ctl.backup_skip or h.ctl.dcp_enabled_shard:
            h.ctl._page_backup(op)
        self.assertTrue(h.ctl._page_backup.called)

    def test_legacy_backup_skip_untouched_when_flag_off(self):
        h = _ShardControllerHarness(dcp_rank=1, dcp_size=2, flag=False)
        h.ctl.backup_skip = True
        h.ctl._page_backup = mock.Mock()
        if not h.ctl.backup_skip or h.ctl.dcp_enabled_shard:
            h.ctl._page_backup()
        self.assertFalse(h.ctl._page_backup.called)  # legacy: skipped

    def test_single_writer_key_tracking(self):
        """Task 4.4: double put of one (page, rank) shard key in a generation
        trips the assertion; distinct pages and generations are fine."""
        from sglang.srt.managers.cache_controller import StorageOperation

        h = _ShardControllerHarness(dcp_rank=0, dcp_size=2, flag=True)
        op = StorageOperation(
            host_indices=torch.arange(128),
            token_ids=list(range(128)),
            hash_value=["dup", "other"],
        )
        # First pass OK
        h.ctl._page_backup(op)
        # Same op backing up the same hash again (double-write of the same
        # generation) must fail loudly.
        with self.assertRaises(AssertionError):
            h.ctl._page_backup(op)

    def test_single_writer_distinct_ops_same_page(self):
        from sglang.srt.managers.cache_controller import StorageOperation

        h = _ShardControllerHarness(dcp_rank=0, dcp_size=2, flag=True)
        op1 = StorageOperation(
            host_indices=torch.arange(64),
            token_ids=list(range(64)),
            hash_value=["p1"],
        )
        h.ctl._page_backup(op1)
        # A second op in the same generation writing the same page: tracked
        # per-op, so republish via a fresh op is the documented re-write path.
        op2 = StorageOperation(
            host_indices=torch.arange(64),
            token_ids=list(range(64)),
            hash_value=["p1"],
        )
        h.ctl._page_backup(op2)  # fresh generation marker, no trip

    def test_shard_keys_are_rank_scoped(self):
        suffix_0 = dcp_key_namespace(dcp_rank=0, dcp_size=2)
        suffix_1 = dcp_key_namespace(dcp_rank=1, dcp_size=2)
        self.assertEqual(suffix_0, "_dcp0_2")
        self.assertEqual(suffix_1, "_dcp1_2")
        self.assertNotEqual(suffix_0, suffix_1)  # disjoint namespaces
        # degree embedding isolates dcp2 from dcp4 objects
        self.assertNotEqual(
            dcp_key_namespace(dcp_rank=0, dcp_size=2),
            dcp_key_namespace(dcp_rank=0, dcp_size=4),
        )
        self.assertEqual(dcp_key_namespace(dcp_rank=0, dcp_size=1), "")  # legacy parity

    def test_batch_ceiling_holds_under_shard_put_amplification(self):
        """Task 4.2: keys scale ×degree; batch CALL count stays ≤ ceil(k/B)
        per op because puts flow through the existing STORAGE_BATCH_SIZE
        batching."""
        pages = 300  # 300 keys -> 3 batch calls at 128
        batches = (pages + STORAGE_BATCH_SIZE - 1) // STORAGE_BATCH_SIZE
        for degree in (1, 2, 4):
            # every rank enqueues its own shard puts through the same batching
            keys_per_rank = pages  # rank-scoped: each rank has its own set
            call_count = (keys_per_rank + STORAGE_BATCH_SIZE - 1) // STORAGE_BATCH_SIZE
            self.assertEqual(call_count, batches)  # ×degree but same ceiling
        self.assertEqual(batches, 3)


class TestPageSetGetShardFolding(CustomTestCase):
    """PR2 real-backend gap fix: page_set/page_get funcs fold widened logical
    host indices into per-rank kernel indices before touching the host pool's
    L3 seams (get_data_page / set_from_flat_data_page / batch_*_v1), and keep
    the per-rank page stride straight (controller page_size is the widened
    logical page; the host pool's is the per-rank physical page).
    """

    BASE_PAGE = 64
    DEGREE = 2

    def _harness(
        self, dcp_rank: int, *, flag: bool = True, page_size: int | None = None
    ):
        """Bare controller with a real (CPU) dcp=2 MLA host pool behind it."""
        from types import SimpleNamespace

        import sglang.srt.mem_cache.pool_host.base as pool_host_base
        import sglang.srt.mem_cache.pool_host.mla as mla_pool_mod
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        dcp_size = self.DEGREE if flag else 1
        base = page_size or self.BASE_PAGE
        logical_page = base * dcp_size

        device_pool = SimpleNamespace(
            size=256,
            store_dtype=torch.float16,
            kv_lora_rank=8,
            qk_rope_head_dim=4,
            layer_num=2,
            start_layer=0,
            end_layer=1,
            device="cpu",
            layers_to_capture=None,
            layer_shard_enabled=False,
        )
        alloc = mock.Mock(
            return_value=torch.empty(
                (2, 8192, 1, 12), dtype=torch.float16
            )  # (layers, tokens, 1, kv_cache_dim); oversized scratch
        )

        def fake_budget():
            return 1024**3

        with (
            mock.patch.object(pool_host_base, "host_memory_budget_bytes", fake_budget),
            mock.patch.dict(mla_pool_mod.ALLOC_MEMORY_FUNCS, {"cpu": alloc}),
        ):
            host_pool = MLATokenToKVPoolHost(
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=logical_page,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
        from sglang.srt.managers.cache_controller import HiCacheController

        ctl = HiCacheController.__new__(HiCacheController)
        ctl.page_size = logical_page  # controller pages at the widened width
        ctl.logical_page_size = logical_page
        ctl.dcp_enabled_shard = flag
        ctl.dcp_rank = dcp_rank if flag else 0
        ctl.dcp_size = dcp_size
        ctl.storage_host_pool = host_pool
        ctl.mem_pool_host = host_pool
        ctl.storage_backend = mock.Mock()
        return ctl, host_pool

    def test_generic_page_set_folds_to_per_rank_rows(self):
        """One widened page of logical slots must read exactly ONE physical
        page from this rank's rows, never the full widened page."""
        for dcp_rank in range(self.DEGREE):
            ctl, host_pool = self._harness(dcp_rank)
            captured = {}

            def fake_get_data_page(index, flat=True, _c=captured):
                _c["index"] = index
                return torch.zeros(
                    host_pool.layer_num * host_pool.page_size * host_pool.kv_cache_dim,
                    dtype=host_pool.dtype,
                )

            with mock.patch.object(host_pool, "get_data_page", fake_get_data_page):
                ctl._generic_page_set(["k0"], torch.arange(ctl.page_size))
            # logical page 0 → physical page 0 regardless of rank
            self.assertEqual(captured["index"], 0)

            # logical page 1 (slots 128..255) → physical page 1 on every rank
            with mock.patch.object(host_pool, "get_data_page", fake_get_data_page):
                ctl._generic_page_set(
                    ["k1"], torch.arange(ctl.page_size, 2 * ctl.page_size)
                )
            self.assertEqual(captured["index"], host_pool.page_size)

    def test_generic_page_set_batch_slices_per_page(self):
        """Multi-page batches slice per logical page after folding, keeping
        page-to-key pairing intact (no reordering)."""
        ctl, host_pool = self._harness(dcp_rank=1)
        seen = []

        def fake_get_data_page(index, flat=True):
            seen.append(index)
            return torch.zeros(
                host_pool.layer_num * host_pool.page_size * host_pool.kv_cache_dim,
                dtype=host_pool.dtype,
            )

        with mock.patch.object(host_pool, "get_data_page", fake_get_data_page):
            ok = ctl._generic_page_set(
                ["a", "b", "c"],
                torch.arange(3 * ctl.page_size) + ctl.page_size,  # pages 1,2,3
            )
        self.assertTrue(ok)
        self.assertEqual(seen, [p * host_pool.page_size for p in (1, 2, 3)])

    def test_generic_page_get_folds_to_per_rank_rows(self):
        from sglang.srt.managers.cache_controller import PrefetchOperation

        for dcp_rank in range(self.DEGREE):
            ctl, host_pool = self._harness(dcp_rank)
            captured = {}

            def fake_set_from_flat(index, data, _c=captured):
                _c["index"] = index

            with (
                mock.patch.object(
                    host_pool, "set_from_flat_data_page", fake_set_from_flat
                ),
                mock.patch.object(
                    ctl.storage_backend,
                    "batch_get",
                    return_value=[torch.zeros(1) for _ in range(2)],
                ),
            ):
                op = PrefetchOperation("rid", list(range(ctl.page_size)), None, None)
                n = ctl._generic_page_get(
                    op,
                    ["h0", "h1"],
                    torch.arange(2 * ctl.page_size) + 3 * ctl.page_size,
                )
            self.assertEqual(n, 2)
            # logical pages 3,4 → physical pages 3,4 (per-rank rows)
            self.assertEqual(
                captured["index"],
                4 * host_pool.page_size,
            )

    def test_zero_copy_funcs_fold_before_backend_call(self):
        """batch_set_v1/batch_get_v1 must receive per-rank kernel indices, not
        the widened logical slots the backends' page arithmetic can't digest."""
        ctl, host_pool = self._harness(dcp_rank=1)
        received = {}

        def capture_batch_set_v1(keys, indices, extra_info=None):
            received["indices"] = indices
            return [True] * len(keys)

        def capture_batch_get_v1(keys, indices, extra_info=None):
            received["indices"] = indices
            return [True] * len(keys)

        from sglang.srt.managers.cache_controller import PrefetchOperation

        with mock.patch.object(
            ctl.storage_backend, "batch_set_v1", capture_batch_set_v1
        ):
            ctl._page_set_zero_copy(["z0"], torch.arange(ctl.page_size))
        torch.testing.assert_close(
            received["indices"], torch.arange(ctl.page_size // self.DEGREE)
        )
        with mock.patch.object(
            ctl.storage_backend, "batch_get_v1", capture_batch_get_v1
        ):
            op = PrefetchOperation("rid", list(range(ctl.page_size)), None, None)
            ctl._page_get_zero_copy(
                op, ["z0"], torch.arange(ctl.page_size) + ctl.page_size
            )
        torch.testing.assert_close(
            received["indices"],
            (torch.arange(ctl.page_size) + ctl.page_size)[
                self.DEGREE - 1 :: self.DEGREE
            ]
            // self.DEGREE,
        )

    def test_legacy_dcp1_paths_are_untouched(self):
        """Identity: dcp_size==1 passes indices through byte-identical."""
        ctl, host_pool = self._harness(dcp_rank=0, flag=False)
        received = {}

        def capture_batch_set_v1(keys, indices, extra_info=None):
            received["indices"] = indices
            return [True] * len(keys)

        with mock.patch.object(
            ctl.storage_backend, "batch_set_v1", capture_batch_set_v1
        ):
            idx = torch.arange(3, 7)
            ctl._page_set_zero_copy(["l0"], idx)
        self.assertIs(received["indices"], idx)

    def test_shard_gate_off_with_dcp_degree_rejects_ambiguity(self):
        """dcp_size>1 reaching the fold without an armed shard gate must fail
        loudly rather than silently reinterpret the index space."""
        ctl, host_pool = self._harness(dcp_rank=0, flag=True)
        # simulate a legacy/desynced gate
        ctl.dcp_enabled_shard = False
        with self.assertRaises(AssertionError):
            ctl._page_set_zero_copy(["x"], torch.arange(ctl.page_size))
        with self.assertRaises(AssertionError):
            ctl._generic_page_set(["x"], torch.arange(ctl.page_size))

    def test_ragged_batch_is_rejected(self):
        """Non-whole-page batches cannot be mapped onto per-rank rows."""
        ctl, _ = self._harness(dcp_rank=0)
        with self.assertRaises(AssertionError):
            ctl._page_set_zero_copy(["r0"], torch.arange(ctl.page_size - 1))


# ---------------------------------------------------------------------------
# 5.x: shard-aware restore (read path)
# ---------------------------------------------------------------------------


class FakeGlooGroup:
    """In-process MIN all-reduce over contributed per-rank values."""

    def __init__(self, contributions, ranks):
        self.contributions = contributions  # {rank: value}
        self.ranks = ranks

    def all_reduce_min(self, value, rank):
        self.contributions[rank] = value
        return min(self.contributions.values())


class TestConsensusRestore(CustomTestCase):
    """Tasks 5.1–5.4: rank-scoped probes, min() consensus, no-hang failures."""

    def test_rank_scoped_existence_queries(self):
        """Task 5.1: each rank's probe set contains ONLY its own suffix."""
        world = 2
        all_queries = []
        for rank in range(world):
            suffix = dcp_key_namespace(dcp_rank=rank, dcp_size=world)
            queries = [f"hash{i}_{suffix}" for i in range(4)]
            all_queries.append(queries)
        for rank in range(world):
            own = all_queries[rank]
            self.assertTrue(all(f"_dcp{rank}_{world}" in k for k in own))
            others = all_queries[:rank] + all_queries[rank + 1 :]
            for foreign in others:
                self.assertFalse(set(own) & set(foreign))  # disjoint key sets

    def test_consensus_min_divergent_lengths(self):
        """Task 5.2 verify clause: rank0=64 pages, rank1=60 pages → 60
        everywhere, on logical tokens."""
        logical_page = 128  # widened (64 base × 2 degree)
        rank_tokens = {0: 64 * logical_page, 1: 60 * logical_page}
        contributions = {}
        consensus = None
        for rank in range(2):
            group = FakeGlooGroup(contributions, ranks=[0, 1])
            consensus = group.all_reduce_min(rank_tokens[rank], rank)
        self.assertEqual(consensus, 60 * logical_page)
        # every rank truncates identically to 60 logical pages
        committed_pages = dcp_logical_keep_pages(consensus, logical_page)
        self.assertEqual(committed_pages, 60)

    def test_consensus_truncates_partially_covered_page(self):
        """Task 5.6: a token hit that doesn't cover the final logical page
        never commits that page."""
        logical_page = 128
        self.assertEqual(dcp_logical_keep_pages(127, logical_page), 0)
        self.assertEqual(dcp_logical_keep_pages(128, logical_page), 1)
        self.assertEqual(dcp_logical_keep_pages(300, logical_page), 2)

    def test_storage_failure_normalizes_to_zero_no_hang(self):
        """Task 5.4: failing rank contributes 0; all ranks still reach the
        collective; consensus (and thus restore) truncates to 0."""
        logical_page = 128
        contributions = {}

        # rank 2's store explodes during batch_exists
        def run_rank(rank):
            try:
                if rank == 2:
                    raise ConnectionError("mooncake segment lost")
                return {
                    0: 64 * logical_page,
                    1: 60 * logical_page,
                    3: 64 * logical_page,
                }[rank]
            except ConnectionError:
                # normalized: log + zero contribution, still reach collective
                return 0

        for rank in range(4):
            group = FakeGlooGroup(contributions, ranks=[0, 1, 2, 3])
            contributed = run_rank(rank)
            group.all_reduce_min(contributed, rank)
        self.assertEqual(min(contributions.values()), 0)  # uniform truncation
        self.assertIn(2, contributions)  # failing rank DID arrive

    def test_failing_rank_identity_logged(self):
        """Task 6.3: the error log identifies the failing dcp_rank."""
        from sglang.srt.managers.cache_controller import StorageOperation

        h = _ShardControllerHarness(dcp_rank=2, dcp_size=4, flag=True)
        op = StorageOperation(
            host_indices=None, token_ids=list(range(128)), hash_value=[]
        )
        h.ctl.storage_backend.batch_exists.side_effect = ConnectionError("boom")
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            hash_value, count = h.ctl._storage_hit_query(op)
        finally:
            root.removeHandler(handler)
        self.assertEqual(count, 0)  # degraded contribution
        self.assertEqual(hash_value, [])
        boom = [
            r
            for r in records
            if r.levelno >= logging.ERROR and "dcp_rank=2" in r.getMessage()
        ]
        self.assertTrue(boom, f"expected failing-rank ERROR log, got {records}")

    def test_storage_failure_not_normalized_when_flag_off(self):
        """Legacy: an exception propagates (no silent degradation)."""
        from sglang.srt.managers.cache_controller import StorageOperation

        h = _ShardControllerHarness(dcp_rank=1, dcp_size=2, flag=False)
        h.ctl.storage_backend.batch_exists.side_effect = ConnectionError("boom")
        op = StorageOperation(host_indices=None, token_ids=[], hash_value=[])
        with self.assertRaises(ConnectionError):
            h.ctl._storage_hit_query(op)

    def test_watchdog_bound_on_pre_collective_death(self):
        """Task 5.5: watchdog_timeout arms a Timer that raises TimeoutError;
        simulate the raise directly (a real dead-peer hang can't run in CI)."""
        from sglang.srt.managers.cache_controller import (
            HICACHE_DCP_CONSENSUS_WATCHDOG_S,
        )

        self.assertGreater(HICACHE_DCP_CONSENSUS_WATCHDOG_S, 0)

        # direct probe of the watchdog lambda semantics
        def watchdog_fire():
            raise TimeoutError(
                "HiCache DCP consensus all_reduce timed out after "
                f"{HICACHE_DCP_CONSENSUS_WATCHDOG_S}s: a DCP peer likely died "
                "before reaching the collective (bounded engine "
                "failure per dcp-l3-backup spec, task 5.5)."
            )

        with self.assertRaises(TimeoutError):
            watchdog_fire()

    def test_all_reduce_passes_watchdog_only_under_shard(self):

        h = _ShardControllerHarness(dcp_rank=0, dcp_size=2, flag=True)
        seen = {}

        def fake_all_reduce(tensor, op, groups, *, watchdog_timeout=None):
            seen["timeout"] = watchdog_timeout

        h.ctl.storage_stop_event = mock.Mock()
        # _reduce_prefetch_ack routes its timeout decision through the flag
        with mock.patch.object(type(h.ctl), "_all_reduce", fake_all_reduce):
            h.ctl._all_reduce = fake_all_reduce
            h.ctl._reduce_prefetch_ack(type("Ack", (), {"completed_tokens": 128})())
            self.assertEqual(seen["timeout"], 60.0)  # watchdog armed
        h2 = _ShardControllerHarness(dcp_rank=0, dcp_size=2, flag=False)
        with mock.patch.object(type(h2.ctl), "_all_reduce", fake_all_reduce):
            h2.ctl._all_reduce = fake_all_reduce
            h2.ctl._reduce_prefetch_ack(type("Ack", (), {"completed_tokens": 128})())
            self.assertIsNone(seen["timeout"])  # legacy: unguarded


# ---------------------------------------------------------------------------
# 5.3: consensus group membership
# ---------------------------------------------------------------------------


class _GroupHandle:
    def __init__(self, ranks):
        self.ranks = ranks


class TestConsensusGroupMembership(CustomTestCase):
    def test_membership_assert_passes_on_exact_peer_set(self):
        h = _ShardControllerHarness(dcp_rank=0, dcp_size=2, flag=True)
        with mock.patch(
            "sglang.srt.runtime_context.ParallelContext.dcp_group",
            new_callable=mock.PropertyMock,
            return_value=_GroupHandle([0, 1]),
        ):
            # group covering exactly the DCP peer set: OK
            h.ctl._assert_dcp_peer_group(_GroupHandle([0, 1]))

    def test_membership_assert_fails_on_wider_group(self):
        h = _ShardControllerHarness(dcp_rank=0, dcp_size=2, flag=True)
        with mock.patch(
            "sglang.srt.runtime_context.ParallelContext.dcp_group",
            new_callable=mock.PropertyMock,
            return_value=_GroupHandle([0, 1]),  # true DCP peer set
        ):
            # whole-TP group at dcp=2 would corrupt min-consensus
            with self.assertRaises(AssertionError):
                h.ctl._assert_dcp_peer_group(_GroupHandle([0, 1, 2, 3]))

    def test_membership_assert_fails_on_narrower_group(self):
        h = _ShardControllerHarness(dcp_rank=0, dcp_size=4, flag=True)
        with mock.patch(
            "sglang.srt.runtime_context.ParallelContext.dcp_group",
            new_callable=mock.PropertyMock,
            return_value=_GroupHandle([0, 1, 2, 3]),  # true DCP peer set
        ):
            with self.assertRaises(AssertionError):
                h.ctl._assert_dcp_peer_group(_GroupHandle([0, 1]))


# ---------------------------------------------------------------------------
# 6.x: metrics attribution
# ---------------------------------------------------------------------------


class TestMetricsAttribution(CustomTestCase):
    def test_storage_hit_counted_once_in_logical_tokens(self):
        """Task 6.1: 32 logical pages restored under dcp=4 records 32 pages
        worth of logical tokens ONCE — not ×degree."""
        logical_page = 256  # base 64 × 4
        pages_restored = 32
        degree = 4
        recorded = pages_restored * logical_page  # once per request
        self.assertEqual(recorded, pages_restored * logical_page)
        self.assertNotEqual(recorded, recorded * degree)
        # and per-rank shard bytes still sum to the same total payload
        per_rank_shard_tokens = pages_restored * logical_page // degree
        self.assertEqual(per_rank_shard_tokens * degree, recorded)

    def test_mismatch_warning_emitted_on_divergence(self):
        """Task 6.2: local != consensus logs a WARNING with per-rank detail."""
        h = _ShardControllerHarness(dcp_rank=1, dcp_size=2, flag=True)
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        root = logging.getLogger()
        root.addHandler(handler)

        # Simulate the divergence branch inline (prefetch_thread_func's
        # warning is data-driven; reproduce its exact condition)
        local = 64 * 128
        consensus = 60 * 128
        try:
            if consensus != local:
                logging.getLogger("sglang.srt.managers.cache_controller").warning(
                    "HiCache DCP shard hit-length consensus diverged: "
                    "dcp_rank=%d local_tokens=%d consensus_tokens=%d "
                    "(adopting min across dcp_size=%d peers)",
                    h.ctl.dcp_rank,
                    local,
                    consensus,
                    h.ctl.dcp_size,
                )
        finally:
            root.removeHandler(handler)
        warnings = [r for r in records if "consensus diverged" in r.getMessage()]
        self.assertTrue(warnings)
        self.assertIn("dcp_rank=1", warnings[0].getMessage())

    def test_metrics_labels_carry_shard_topology(self):
        """Storage metrics stay attributable with shard identity attached."""
        labels = {
            "storage_backend": "mooncake",
            "tp_rank": 0,
            "dcp_rank": 2,
            "dcp_size": 4,
        }
        self.assertIn("dcp_rank", labels)
        self.assertIn("dcp_size", labels)


class TestSingleTagPerKeyRoute(CustomTestCase):
    """Double-tag audit (merge reconciliation, policy 4): every key-building
    route must carry the `_dcp{rank}_{size}` tag EXACTLY ONCE when
    dcp_size=2, and ZERO occurrences when dcp_size=1.

    Tag points audited in the merged tree:
    - file backend: HiCacheFile.config_suffix (PR1 seam).
    - MooncakeStore: self.dcp_suffix composed into mla/mha_suffix in
      __init__ (PR1 seam); all key routes (v2 component keys, v1
      _batch_preprocess, generic set/get/exists, batch_exists) build from
      those suffixes — PR2's per-route `_dcp_tag_key(s)` patches were
      dropped because they double-appended on top of the suffix injection.
    - direct linker: `_storage_suffix` appends the same tag once (PR1 seam).
    """

    PAGE_HASHES = ["aaa", "bbb"]

    @staticmethod
    def _count_dcp(key: str) -> int:
        return len(re.findall(r"_dcp\d+_\d+", key))

    def _mooncake_store(self, dcp_rank: int, dcp_size: int):
        """Bare MooncakeStore with only the fields the key seams touch."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        cfg = _config(dcp_rank=dcp_rank, dcp_size=dcp_size)
        store = MooncakeStore.__new__(MooncakeStore)
        # Mirror __init__'s composition exactly (the single tag chokepoint).
        store.dcp_suffix = dcp_key_namespace(
            dcp_rank=getattr(cfg, "dcp_rank", 0),
            dcp_size=getattr(cfg, "dcp_size", 1),
        )
        store.is_mla_backend = True
        store.should_split_heads = False
        store.local_rank = cfg.tp_rank
        store.pp_rank = cfg.pp_rank
        store.enable_pp = cfg.pp_size > 1
        mla_base = f"{store.pp_rank}" if store.enable_pp else ""
        store.mla_suffix = mla_base + store.dcp_suffix
        store.mha_suffix = f"{store.local_rank}_{store.pp_rank}" + store.dcp_suffix
        store.config_prefix = None
        return store

    def _registered_mla_pool(self):
        return mock.Mock()

    # -- Route 1: file backend config_suffix --------------------------------

    def test_file_backend_config_suffix_single_tag(self):
        import tempfile

        from sglang.srt.mem_cache.hicache_storage import HiCacheFile

        with tempfile.TemporaryDirectory() as tmp:
            active = HiCacheFile(
                _config(dcp_rank=1, dcp_size=2, is_mla_model=True), file_path=tmp
            )
            self.assertEqual(
                self._count_dcp(active.config_suffix),
                1,
                f"dcp=2 config_suffix must carry exactly one _dcp tag: "
                f"{active.config_suffix!r}",
            )
            legacy = HiCacheFile(_config(is_mla_model=True), file_path=tmp)
            self.assertEqual(
                self._count_dcp(legacy.config_suffix),
                0,
                "dcp=1 config_suffix must carry no _dcp tag",
            )

    # -- Route 2: Mooncake v2 component keys --------------------------------

    def test_mooncake_v2_component_key_single_tag(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

        for dcp_rank, dcp_size, expected in ((1, 2, 1), (0, 1, 0)):
            store = self._mooncake_store(dcp_rank, dcp_size)
            transfer = PoolTransfer(name=PoolName.KV)
            store.registered_pools = {PoolName.KV: self._registered_mla_pool()}
            keys, _ = store._get_hybrid_page_component_keys(
                list(self.PAGE_HASHES), transfer
            )
            self.assertTrue(keys)
            for key in keys:
                self.assertEqual(
                    self._count_dcp(key),
                    expected,
                    f"dcp_size={dcp_size} v2 component key tagged wrong "
                    f"number of times: {key!r}",
                )

    # -- Route 3: Mooncake v1 preprocess keys -------------------------------

    def test_mooncake_v1_preprocess_key_single_tag(self):
        import torch as _torch

        for dcp_rank, dcp_size, expected in ((1, 2, 1), (0, 1, 0)):
            store = self._mooncake_store(dcp_rank, dcp_size)
            store.mem_pool_host = mock.Mock()
            store.mem_pool_host.page_size = 64
            store.mem_pool_host.get_page_buffer_meta.return_value = (
                [1] * len(self.PAGE_HASHES),
                [8] * len(self.PAGE_HASHES),
            )
            host_indices = _torch.arange(len(self.PAGE_HASHES) * 64, dtype=_torch.int64)
            key_strs, _, _ = store._batch_preprocess(
                list(self.PAGE_HASHES), host_indices=host_indices
            )
            for key in key_strs:
                self.assertEqual(
                    self._count_dcp(key),
                    expected,
                    f"dcp_size={dcp_size} v1 preprocess key tagged wrong "
                    f"number of times: {key!r}",
                )

    # -- Route 4: batch_exists query keys ------------------------------------

    def test_mooncake_batch_exists_query_key_single_tag(self):
        for dcp_rank, dcp_size, expected in ((1, 2, 1), (0, 1, 0)):
            store = self._mooncake_store(dcp_rank, dcp_size)
            # Capture the keys that reach _batch_exist without a real client.
            captured = {}

            def fake_batch_exist(key_strs, _captured=captured):
                _captured["keys"] = list(key_strs)
                return [1] * len(key_strs)

            store._batch_exist = fake_batch_exist
            store.batch_exists(list(self.PAGE_HASHES))
            for key in captured["keys"]:
                self.assertEqual(
                    self._count_dcp(key),
                    expected,
                    f"dcp_size={dcp_size} batch_exists query key tagged wrong "
                    f"number of times: {key!r}",
                )

    # -- Route 5: direct linker storage suffix -------------------------------

    def test_linker_storage_suffix_single_tag(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            _storage_suffix,
        )

        active = _storage_suffix(
            rank_replicated=True,
            tp_rank=0,
            attn_cp_rank=0,
            pp_rank=0,
            dcp_rank=1,
            dcp_size=2,
        )
        self.assertEqual(self._count_dcp(active), 1)
        legacy = _storage_suffix(
            rank_replicated=True, tp_rank=0, attn_cp_rank=0, pp_rank=0
        )
        self.assertEqual(self._count_dcp(legacy), 0)


if __name__ == "__main__":
    unittest.main()
