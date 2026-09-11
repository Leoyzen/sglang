"""Non-mock end-to-end test: DCP L3 shard write/read against the real file backend.

PR2 real-GPU-integration gap reproduction (V2-Lite + dcp2 + file L3): the
crash chain was ``_page_backup → _generic_page_set → get_data_page → the
dcp_size == 1 assert`` because the controller never folded widened logical
host indices into per-rank kernel indices. These tests run the REAL
``HiCacheController._generic_page_set`` / ``_generic_page_get`` /
``_page_set_zero_copy`` machinery against a REAL ``HiCacheFile`` backend on a
tmp dir, with a real dcp_size=2 ``MLATokenToKVPoolHost``, asserting:

1. write → read roundtrip restores the rank's own shard bytes;
2. per-rank isolation: each rank's shard files carry only its shard data and
   share the store namespace without colliding;
3. keys land with exactly one ``_dcp{rank}_{size}`` tag (inside PR1's
   config_suffix seam);
4. the pre-fix crash path (get_data_page on dcp>1) no longer raises and never
   yields the full widened page.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEGREE = 2
BASE_PAGE = 8
LOGICAL_PAGE = BASE_PAGE * DEGREE
LAYERS = 2
KV_DIM = 12  # kv_lora_rank + qk_rope_head_dim


def _storage_cfg(rank: int) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test/dcp-l3-real",
        dcp_rank=rank,
        dcp_size=DEGREE,
        enable_hicache_dcp_shard=True,
        extra_config={"enable_metadata_cache": "false"},
    )


class _World:
    """One rank: real host pool + real HiCacheFile + bare controller."""

    def __init__(self, rank: int, file_path: str):
        import sglang.srt.mem_cache.pool_host.base as pool_host_base
        import sglang.srt.mem_cache.pool_host.mla as mla_pool_mod
        from sglang.srt.managers.cache_controller import HiCacheController
        from sglang.srt.mem_cache.hicache_storage import HiCacheFile
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        device_pool = mock.Mock()
        device_pool.size = 256
        device_pool.store_dtype = torch.float16
        device_pool.dtype = torch.float16
        device_pool.kv_lora_rank = 8
        device_pool.qk_rope_head_dim = 4
        device_pool.layer_num = LAYERS
        device_pool.start_layer = 0
        device_pool.end_layer = LAYERS - 1
        device_pool.device = "cpu"
        device_pool.layers_to_capture = None
        device_pool.layer_shard_enabled = False
        device_pool.index_head_dim = None

        alloc = mock.Mock(
            return_value=torch.empty((LAYERS, 4096, 1, KV_DIM), dtype=torch.float16)
        )
        with (
            mock.patch.object(
                pool_host_base, "host_memory_budget_bytes", lambda: 1024**3
            ),
            mock.patch.dict(mla_pool_mod.ALLOC_MEMORY_FUNCS, {"cpu": alloc}),
        ):
            self.host_pool = MLATokenToKVPoolHost(
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=LOGICAL_PAGE,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                dcp_size=DEGREE,
                dcp_rank=rank,
            )

        self.backend = HiCacheFile(_storage_cfg(rank), file_path=file_path)
        self.backend.register_mem_pool_host(self.host_pool)

        self.rank = rank
        ctl = HiCacheController.__new__(HiCacheController)
        ctl.page_size = LOGICAL_PAGE  # controller pages at the widened width
        ctl.logical_page_size = LOGICAL_PAGE
        ctl.dcp_enabled_shard = True
        ctl.dcp_rank = rank
        ctl.dcp_size = DEGREE
        ctl.storage_host_pool = self.host_pool
        ctl.mem_pool_host = self.host_pool
        ctl.storage_backend = self.backend
        ctl.storage_backend_type = "file"
        self.ctl = ctl

    def stage_shard(self, logical_page_starts, page_files_seed=1):
        """Fill this rank's physical rows for the given logical pages with
        deterministic, rank-and-page-distinct values.

        Widened logical page ``p`` (slots ``[p*L, (p+1)*L)`` with
        ``L = LOGICAL_PAGE``) folds to per-rank physical rows
        ``[p*BASE_PAGE, (p+1)*BASE_PAGE)`` on EVERY rank (owner-rule stride
        filtering within the slot-aligned run then ``// DEGREE``) — the same
        translation ``HostKVCache.maybe_dcp_kernel_indices`` applies.
        """
        for start in logical_page_starts:
            kernel_start = start * BASE_PAGE
            page = self.host_pool.kv_buffer[
                :, kernel_start : kernel_start + BASE_PAGE, :, :
            ]
            page.copy_(
                torch.arange(page.numel(), dtype=torch.float16).reshape(page.shape)
                + 1000.0 * (self.rank + 1)
                + 10.0 * start
                + page_files_seed
            )


class TestRealFileBackendShardRoundtrip(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self._tmp.name, "l3")
        os.makedirs(self.file_path, exist_ok=True)
        # Two ranks share one file dir, mirroring the co-located deployment.
        self.r0 = _World(0, self.file_path)
        self.r1 = _World(1, self.file_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_read_roundtrip_per_rank_shard(self):
        hashes = ["aa", "bb"]
        logical_pages = [1, 2]  # widened pages 1 and 2
        host_indices = torch.cat(
            [
                torch.arange(p * LOGICAL_PAGE, (p + 1) * LOGICAL_PAGE)
                for p in logical_pages
            ]
        )
        extra = HiCacheStorageExtraInfo(prefix_keys=None)

        for world in (self.r0, self.r1):
            world.stage_shard(logical_pages)
            ok = world.ctl._generic_page_set(hashes, host_indices, extra)
            self.assertTrue(ok, f"rank {world.rank}: backup must succeed")

        # Keys landed rank-scoped: the file backend composes
        # ``{key}{config_suffix}`` and PR1's seam embeds ``_dcp{rank}_{size}``
        # inside config_suffix — exactly one _dcp occurrence on disk.
        for world in (self.r0, self.r1):
            for h in hashes:
                self.assertTrue(
                    world.backend.exists(h),
                    f"rank {world.rank} missing shard file for {h}",
                )
                expected = f"{h}{world.backend.config_suffix}"
                self.assertTrue(
                    os.path.exists(os.path.join(self.file_path, expected + ".bin")),
                    f"expected on-disk file {expected}.bin",
                )
                self.assertEqual(
                    expected.count("_dcp"),
                    1,
                    f"key {expected!r} must carry exactly one _dcp tag",
                )

        # Cross-rank namespace isolation: rank0's file content differs from
        # rank1's for the same page hash (distinct shard payloads).
        f0 = os.path.join(self.file_path, f"aa{self.r0.backend.config_suffix}.bin")
        f1 = os.path.join(self.file_path, f"aa{self.r1.backend.config_suffix}.bin")
        with open(f0, "rb") as fh:
            b0 = fh.read()
        with open(f1, "rb") as fh:
            b1 = fh.read()
        self.assertNotEqual(b0, b1, "shards of different ranks must differ")
        # Each shard holds 1/DEGREE of the widened page payload.
        widened_bytes = (
            LAYERS * BASE_PAGE * DEGREE * 1 * KV_DIM * torch.float16.itemsize
        )
        self.assertEqual(len(b0), widened_bytes // DEGREE)
        self.assertEqual(len(b1), widened_bytes // DEGREE)

        # Overwrite the host rows with garbage, then read back through the
        # real generic page-get path and verify shard bytes roundtrip.
        for world in (self.r0, self.r1):
            for start in logical_pages:
                ks = start * BASE_PAGE
                world.host_pool.kv_buffer[:, ks : ks + BASE_PAGE, :, :].fill_(-777.0)
            from sglang.srt.managers.cache_controller import PrefetchOperation

            op = PrefetchOperation(
                "rid", [0] * (len(hashes) * LOGICAL_PAGE), None, None
            )
            hit = world.ctl._generic_page_get(op, hashes, host_indices, extra)
            self.assertEqual(hit, len(hashes), f"rank {world.rank}: full hit expected")

            for start in logical_pages:
                ks = start * BASE_PAGE
                restored = world.host_pool.kv_buffer[
                    :, ks : ks + BASE_PAGE, :, :
                ].flatten()
                expected = (
                    torch.arange(restored.numel(), dtype=torch.float16).reshape(-1)
                    + 1000.0 * (world.rank + 1)
                    + 10.0 * start
                    + 1
                )
                torch.testing.assert_close(
                    restored,
                    expected.reshape(restored.shape),
                    msg=f"rank {world.rank} logical page {start} shard bytes "
                    "did not roundtrip",
                )

    def test_zero_copy_path_folds_against_real_backend_keys(self):
        """The v1 seam (nixl/mooncake/sim-class backends) does page arithmetic
        on host_indices: len(host_indices) must equal
        len(keys) * host_pool.page_size after the controller's fold — per-rank
        physical slots, never the widened logical slots. Drive the
        controller's zero-copy page-set func through a faithful v1 face over
        the real file backend, so the fold is verified against real storage.
        """
        hashes = ["z1"]
        logical_pages = [3]
        host_indices = torch.arange(3 * LOGICAL_PAGE, 4 * LOGICAL_PAGE)
        extra = HiCacheStorageExtraInfo(prefix_keys=None)

        class _FileV1Adapter:
            """v1 face over the real HiCacheFile: what backends with
            page-arithmetic on host_indices see from the controller."""

            def __init__(self, inner):
                self.inner = inner
                self.seen_index_lengths = []

            def batch_set_v1(self, keys, host_indices, extra_info=None):
                self.seen_index_lengths.append(host_indices.numel())
                pages = [
                    self.inner.mem_pool_host.get_data_page(
                        host_indices[i * self.inner.mem_pool_host.page_size]
                    )
                    for i in range(len(keys))
                ]
                ok = self.inner.batch_set(keys, pages)
                return [ok] * len(keys)

            def register_mem_pool_host(self, mem_pool_host):
                self.inner.register_mem_pool_host(mem_pool_host)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        for world in (self.r0, self.r1):
            world.stage_shard(logical_pages)
            adapter = _FileV1Adapter(world.backend)
            # Drive the controller's zero-copy page-set func (not raw backend
            # calls): the fold must happen INSIDE the controller.
            world.ctl.storage_backend = adapter
            ok = world.ctl._page_set_zero_copy(hashes, host_indices, extra)
            self.assertTrue(ok)
            # LOGICAL_PAGE logical slots fold to BASE_PAGE per-rank rows.
            self.assertEqual(adapter.seen_index_lengths, [BASE_PAGE])
            world.ctl.storage_backend = world.backend
            for h in hashes:
                self.assertTrue(world.backend.exists(h))
                # PR1's seam embeds _dcp{rank}_{size} inside config_suffix.
                expected_name = f"{h}{world.backend.config_suffix}.bin"
                self.assertEqual(expected_name.count("_dcp"), 1)
                self.assertTrue(
                    os.path.exists(os.path.join(self.file_path, expected_name))
                )
                # byte count proves only the per-rank shard was stored
                size = os.path.getsize(os.path.join(self.file_path, expected_name))
                self.assertEqual(
                    size, LAYERS * BASE_PAGE * 1 * KV_DIM * torch.float16.itemsize
                )

    def test_generic_page_set_does_not_hit_full_widened_page(self):
        """Regression for the original crash: get_data_page under dcp>1 on
        the real pool must return the per-rank page, not blow up or return
        dcp× the data."""
        captured = {}
        orig = self.r1.host_pool.get_data_page

        def spy(index, flat=True):
            out = orig(index, flat=flat)
            captured["numel"] = out.numel()
            captured["index"] = index
            return out

        with mock.patch.object(self.r1.host_pool, "get_data_page", spy):
            ok = self.r1.ctl._generic_page_set(
                ["w0"], torch.arange(LOGICAL_PAGE, 2 * LOGICAL_PAGE)
            )
        self.assertTrue(ok)
        self.assertEqual(
            captured["numel"],
            LAYERS * BASE_PAGE * 1 * KV_DIM,
            "get_data_page must yield ONE per-rank physical page",
        )
        # logical page 1 folds to physical page 1 (rows BASE_PAGE..2*BASE_PAGE)
        self.assertEqual(captured["index"], BASE_PAGE)


class TestHybridBackupSkipDcpBypass(CustomTestCase):
    """Regression for the TP1-zero-shard bug: the replicated-MLA
    ``backup_skip`` premise breaks under DCP sharding, and the fix landed in
    the BASE class while the runtime class is the HybridCacheController
    SUBCLASS whose overrides lacked the bypass — silently acking ops with
    zero bytes on rank>0. These tests pin the decision surface shared by
    both classes."""

    def _hybrid_ctl(self, backup_skip: bool, dcp_enabled_shard: bool):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        ctl = HybridCacheController.__new__(HybridCacheController)
        ctl.backup_skip = backup_skip
        ctl.dcp_enabled_shard = dcp_enabled_shard
        ctl._warned_backup_skip = False
        return ctl

    def test_should_write_kv_to_storage_true_for_shard_rank(self):
        # The exact TP1 configuration that produced zero shard files:
        # MLA (backup_skip=True) + tp_rank=1 + DCP shard active.
        ctl = self._hybrid_ctl(backup_skip=True, dcp_enabled_shard=True)
        self.assertTrue(ctl.should_write_kv_to_storage())

    def test_should_write_kv_to_storage_legacy_semantics_preserved(self):
        # Legacy replicated-MLA: rank>0 without the shard path keeps skipping.
        ctl = self._hybrid_ctl(backup_skip=True, dcp_enabled_shard=False)
        self.assertFalse(ctl.should_write_kv_to_storage())
        # Rank0 always writes.
        ctl0 = self._hybrid_ctl(backup_skip=False, dcp_enabled_shard=False)
        self.assertTrue(ctl0.should_write_kv_to_storage())

    def test_should_backup_permits_kv_under_dcp_shard(self):
        from sglang.srt.mem_cache.hicache_storage import PoolTransfer

        ctl = self._hybrid_ctl(backup_skip=True, dcp_enabled_shard=True)
        transfer = mock.Mock(spec=PoolTransfer)
        self.assertTrue(ctl.should_backup(transfer))

    def test_base_class_helper_agrees_with_subclass(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        base = HiCacheController.__new__(HiCacheController)
        sub = self._hybrid_ctl(backup_skip=True, dcp_enabled_shard=True)
        for skip in (True, False):
            for shard in (True, False):
                base.backup_skip = skip
                base.dcp_enabled_shard = shard
                sub.backup_skip = skip
                sub.dcp_enabled_shard = shard
                self.assertEqual(
                    base.should_write_kv_to_storage(),
                    sub.should_write_kv_to_storage(),
                    f"decision drift at backup_skip={skip}, dcp_shard={shard}",
                )


if __name__ == "__main__":
    unittest.main()
