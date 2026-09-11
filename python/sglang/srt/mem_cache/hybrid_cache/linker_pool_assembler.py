"""Materialize hybrid device pools for external cache linkers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    pass


def _is_hybrid_linear_kv_pool(pool: Any) -> bool:
    """Duck check for the HybridLinearKVPool wrapper (lazy import avoids a
    circular dependency with memory_pool)."""
    try:
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(pool, HybridLinearKVPool):
            return True
    except ImportError:
        pass
    return hasattr(pool, "full_kv_pool") and hasattr(
        pool, "full_attention_layer_id_mapping"
    )


_warned_packed_draft_under_dcp = False


def _active_dcp_size() -> int:
    """Best-effort published DCP degree for ENTRY CONSTRUCTION defaults.

    Returns 1 when no parallel context is published (unit-test harness) or
    on a RuntimeError from the reader — entry construction must not crash
    over bookkeeping; correctness of folding itself is still fail-closed in
    `_dcp_folding_index_mapper`.
    """
    try:
        return max(1, int(get_parallel().attn_dcp_size))
    except (RuntimeError, AttributeError, TypeError):
        return 1


def _warn_packed_draft_under_dcp(draft_buffers: Sequence[torch.Tensor]) -> None:
    """One-shot warning: packed MTP draft buffers share the KV entry's folded
    target row domain under DCP, so their persisted rows land on folded
    positions instead of the draft's raw virtual locs. Roundtrips are
    self-consistent (no aliasing, no OOB) but restored draft latents are
    misplaced — EAGLE verify keeps outputs correct; only speculative
    acceptance degrades. Full draft-split (dedicated DRAFT entry keyed on the
    widened page) is the follow-up. Identity risk-free at dcp=1 (no warning,
    packed layout unchanged)."""
    global _warned_packed_draft_under_dcp
    if _warned_packed_draft_under_dcp or not draft_buffers:
        return
    try:
        dcp_size = get_parallel().attn_dcp_size
    except RuntimeError:
        return
    if dcp_size <= 1:
        return
    _warned_packed_draft_under_dcp = True
    logger.warning(
        "Direct linker under DCP (dcp_size=%d): packed MTP draft KV/index rows "
        "share the folded target row domain, so persisted draft latents are "
        "misplaced on restore. Output correctness is preserved by verify; "
        "expect degraded speculative acceptance on L3-restored prefixes.",
        dcp_size,
    )


def _dcp_folding_index_mapper(indices: torch.Tensor) -> torch.Tensor:
    """Fold widened DCP logical slots into this rank's physical rows.

    Direct-linker keys are composed from the per-rank `mla_suffix`/`mha_suffix`
    (which carry the `_dcp{rank}_{size}` namespace), so the KEY space is already
    rank-scoped — no shard flag is needed. Only the SLOT indices still speak the
    allocator's widened logical space (`size = physical * dcp_size`, page
    `page_size * dcp_size`; see `UnifiedMambaTokenToKVPoolAllocator.__init__`)
    while the device pool entry's buffers address per-rank physical rows.
    Keep this rank's slots (`% dcp_size == dcp_rank`) then collapse
    (`// dcp_size`) — the same owner rule as
    ``HostKVCache.maybe_dcp_kernel_indices`` and
    ``write_loc_to_kernel_ids``. Identity when DCP is inactive (dcp=1 keys and
    slots are unchanged, byte-for-byte).

    Batch alignment requirement (fail-loud, not a silent reinterpretation): a
    transfer batch must be whole widened pages — each key covers
    ``page_size * dcp_size`` logical slots and every rank folds the same
    keys down to ITS ``page_size`` physical rows.

    Fail-closed on an unpublished parallel context in PRODUCTION: guessing
    "probably dcp=1" is precisely the silent reinterpretation this seam exists
    to prevent. Exception: a bare unit-test harness (no parallel config ever
    published, the process IS the single rank) runs as dcp=1 identity — same
    convention `MultiEndedAllocator` and every other `get_parallel()` reader
    effectively rely on their fixtures meeting.
    """
    try:
        dcp_size = get_parallel().attn_dcp_size
    except RuntimeError as error:
        if get_parallel()._config is None and not get_parallel()._derived:
            # Unit-test harness: no publish, no stamp — single-rank identity.
            return indices
        raise RuntimeError(
            "Direct linker DCP slot folding requires a published parallel "
            "context (get_parallel().attn_dcp_size); refusing to guess the "
            "degree and reinterpret the slot space. Publish the parallel "
            "config, or state it with get_parallel().override(attn_dcp_size=...)."
        ) from error
    if dcp_size <= 1:
        return indices
    dcp_rank = get_parallel().attn_dcp_rank
    if indices.numel() % dcp_size:
        raise ValueError(
            "Direct linker DCP slot folding got "
            f"{indices.numel()} logical slots, not a multiple of "
            f"dcp_size={dcp_size}; offload/load batches must be runs of "
            "whole widened pages."
        )
    return indices[dcp_rank::dcp_size] // dcp_size


class DevicePoolEntry:
    """Zero-copy linker view over one physical device pool."""

    def __init__(
        self,
        *,
        name: PoolName,
        indices_from_pool: PoolName,
        device_pool: Any,
        components: Sequence[Sequence[torch.Tensor]],
        layer_mapping: dict[int, int | Sequence[int]],
        page_size: int,
        rows_are_pages: bool,
        packed: bool = True,
        index_mapper: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dcp_fold_slots: bool = False,
        # Accepted for source compatibility with old call sites, but the
        # authoritative per-key slot width is derived LAZILY at I/O time via
        # `slots_per_key`/`_object_width()` (a construction-time value races
        # the parallel-context publish behind the prod incident — see
        # `_object_width` docstring).
        slots_per_key: int | None = None,
    ):
        if slots_per_key is not None:
            logger.warning(
                "DevicePoolEntry(%s): `slots_per_key` at construction is ignored; "
                "width is resolved lazily at I/O time (drop the argument).",
                name,
            )
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.rows_are_pages = rows_are_pages
        self.packed = packed
        self._index_mapper = index_mapper
        # Declare that this entry's slot indices arrive in the DCP-widened
        # logical space and must fold to per-rank rows before row arithmetic
        # (see `_dcp_folding_index_mapper`). Only the sharded target pools
        # carry it; replicated/slot-granular entries (mamba state, SWA,
        # replicated index sidecars) stay pass-through.
        if dcp_fold_slots:
            if index_mapper is not None:
                raise ValueError(
                    f"Device pool {name} cannot combine index_mapper with dcp_fold_slots."
                )
            self._index_mapper = _dcp_folding_index_mapper
        self.dcp_fold_slots = dcp_fold_slots

        if not self.components or any(not component for component in self.components):
            raise ValueError(f"Device pool {name} has no storage buffers.")
        self.kv_buffer = [buffer for group in self.components for buffer in group]
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)

    def _object_width(self) -> int:
        """Slots one transfer key covers in THIS entry's slot domain, resolved
        at I/O time because the DCP degree is only published AFTER entry
        construction.

        - folding entries (dcp_fold_slots: KV latent targets): the FOLDED
          physical width — `page_size // dcp_size`. Under DCP the entry's
          page_size is already the widened width (kv_cache_builder passes
          allocator.page_size = tree_page * dcp_size), while the folded
          indices it receives carry `page_size // dcp_size` slots per key
          (16 keys -> 1024 folded physical rows at dcp=4). Resolving at
          construction reads dcp=1 before the publish and silently keeps the
          widened width (the exact 3984231c25 regression that left the prod
          assert firing).
        - raw widened/global-domain entries (DSA INDEXER sidecar): `page_size`
          — its index-K rows are global-slot replicated, sized size*dcp_size
          (IndexKeyCache), addressed in raw widened slots; one key spans one
          widened page (= page_size slots). Do NOT multiply by dcp_size again
          (over-declares 4x under the old convention).
        - self-keyed pools: page_size.
        dcp=1 => page_size for every kind (byte-identical to today).
        """
        if self.dcp_fold_slots:
            dcp = max(1, _active_dcp_size())
            return max(1, self.page_size // dcp)
        return self.page_size

    @property
    def slots_per_key(self) -> int:
        """Width (slots) one transfer key covers in the SOURCE index space.
        This is what `MooncakeStore._batch_io_v2` divides len(host_indices)
        by (``len(keys) == len(indices) // slots_per_key``); must equal the
        per-key width of the indices actually handed to the store — folded
        width for fold entries, page_size otherwise."""
        return self._object_width()

    @property
    def _row_span(self) -> int:
        """Physical buffer rows per transferred object. page-granular pools
        (rows_are_pages, e.g. the INDEXER index-K) address one row per object;
        slot-granular pools (KV latents) address `_object_width` rows per
        object (the folded width for fold entries)."""
        return 1 if self.rows_are_pages else self._object_width()

    @property
    def _rows_per_key(self) -> int:
        """Rows one transfer key spans in THIS entry's row domain — always 1
        in the corrected convention: each key is folded/reshaped to one
        contiguous object (`_rows` emits one start row per key), so the
        restore path groups one multi-range group per key. Identity at
        dcp=1."""
        return self.slots_per_key // self._object_width()  # == 1

    def _io_buffers(self):
        """buffer_meta + component offsets, built lazily because the per-row
        byte size depends on the I/O-time `_row_span` (fold entries)."""
        row_span = self._row_span
        buffer_meta = [
            [
                (
                    buffer.data_ptr(),
                    buffer.stride(0) * buffer.element_size(),
                    buffer.nbytes // buffer.shape[0] * row_span,
                )
                for buffer in component
            ]
            for component in self.components
        ]
        component_offsets = []
        offset = 0
        for component in buffer_meta:
            if not self.packed:
                offset = 0
            offsets = []
            for _, _, size in component:
                offsets.append(offset)
                offset += size
            component_offsets.append(offsets)
        return buffer_meta, component_offsets

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._index_mapper(indices) if self._index_mapper else indices

    def _rows(self, indices: torch.Tensor) -> list[int]:
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        width = self._object_width()
        if slots.numel() % width:
            raise ValueError(
                f"Pool {self.name} got {slots.numel()} indices, expected a multiple of object width {width}."
            )
        if not slots.numel():
            return []

        pages = slots.reshape(-1, width)
        starts = pages[:, 0]
        if torch.any(starts.remainder(width)) or not torch.equal(
            pages, starts[:, None] + torch.arange(width)
        ):
            raise ValueError(f"Pool {self.name} requires aligned contiguous pages.")
        rows = (
            starts.div(width, rounding_mode="floor") if self._row_span == 1 else starts
        )
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            raise ValueError(
                f"Pool {self.name} row range [{first_row}, {last_row}) exceeds buffer shapes {[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )
        return rows.tolist()

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices)
        buffer_meta, _ = self._io_buffers()
        ptrs = [
            base_ptr + row * row_stride
            for row in rows
            for component in buffer_meta
            for base_ptr, row_stride, _ in component
        ]
        sizes = [
            size for _ in rows for component in buffer_meta for _, _, size in component
        ]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices)

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        mapped = self.layer_mapping.get(layer)
        if mapped is None:
            return None
        buffer_indices = [mapped] if isinstance(mapped, int) else list(mapped)
        buffer_meta, component_offsets = self._io_buffers()

        items = []
        for component, offsets in zip(buffer_meta, component_offsets):
            for buffer_index in buffer_indices:
                base_ptr, row_stride, size = component[buffer_index]
                items.append((base_ptr, row_stride, size, offsets[buffer_index]))

        ptrs, sizes, offsets = [], [], []
        # One multi-range GROUP per key (`_rows_per_key` is always 1 in the
        # corrected convention: every key is folded/reshaped to one contiguous
        # object, so `_rows`/`prepare_locations` emit one start row per key).
        # Mooncake aligns range groups to COMPONENT KEYS, so we group the
        # rows one per key — the same per-key list shape
        # `_pack_multi_buffer_meta` produces on the offload side. Identity at
        # dcp=1 (rows_per_key==1) reduces to the historical one-row-per-key
        # shape.
        rows_per_group = self._rows_per_key
        if len(locations) % rows_per_group:
            raise ValueError(
                f"Pool {self.name} got {len(locations)} prepared rows, not a multiple of rows_per_key={rows_per_group}."
            )
        for group_start in range(0, len(locations), rows_per_group):
            key_rows = locations[group_start : group_start + rows_per_group]
            # One multi-range GROUP per key: concatenate every row's ranges
            # (row-major over the component items) so Mooncake's
            # keys↔groups zip stays key-aligned.
            key_ptrs, key_sizes, key_offsets = [], [], []
            for row in key_rows:
                key_ptrs.extend(
                    base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items
                )
                key_sizes.extend(size for _, _, size, _ in items)
                key_offsets.extend(offset for _, _, _, offset in items)
            if self.packed:
                ptrs.append(key_ptrs)
                sizes.append(key_sizes)
                offsets.append(key_offsets)
            else:
                ptrs.extend([[value] for value in key_ptrs])
                sizes.extend([[value] for value in key_sizes])
                offsets.extend([[value] for value in key_offsets])
        return ptrs, sizes, offsets


class DevicePoolGroup:
    """Physical device pools sharing one logical linker layer range."""

    def __init__(
        self,
        entries: Sequence[DevicePoolEntry],
        num_layers: int,
        page_size: int,
        *,
        rank_replicated: bool = False,
        kv_source_shards_under_dcp: bool = True,
    ):
        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        if len(self.entries) != len(self.entry_map):
            raise ValueError("DevicePoolGroup contains duplicate pool names.")
        self.sources = {entry.name: entry.indices_from_pool for entry in self.entries}
        self.num_layers = num_layers
        self.page_size = page_size
        self.rank_replicated = rank_replicated
        self.kv_buffer = None
        # NOTE: index mappers stay PER-ENTRY (`resolve_transfers` translates
        # each derived entry through its own mapper). A KV-derived entry that
        # reads the same sharded row space carries the same folding mapper; a
        # replicated global-slot sidecar (DSA INDEXER) deliberately does not.
        # `kv_source_shards_under_dcp=False` marks a group whose KV-source
        # slots are NOT the sharded full-attention space (pure replicated
        # SWA/KV stacks): the DCP folding mapper must stay off there, because
        # folding applies only to the sharded latent rows.
        if not kv_source_shards_under_dcp:
            for name, entry in self.entry_map.items():
                if entry._index_mapper is _dcp_folding_index_mapper:
                    raise ValueError(
                        f"Device pool {name} declares DCP slot folding, but the "
                        "group's KV source does not shard under DCP "
                        "(kv_source_shards_under_dcp=False)."
                    )

    def resolve_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        allow_partial: bool = False,
        allow_missing_kv: bool = False,
    ) -> list[PoolTransfer]:
        """Expand logical component transfers into physical device pools."""
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if not any(transfer.keys for transfer in transfers):
            return []
        if not allow_missing_kv and (kv is None or not kv.keys):
            return []
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        resolved = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None or not source.keys:
                continue
            indices = source.device_indices
            resolved.append(
                replace(
                    source,
                    name=name,
                    host_indices=(
                        self.entry_map[name].translate_indices(indices)
                        if indices is not None
                        else None
                    ),
                    keys=list(source.keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                    # Keep the key-space identity: transfers sourced from a
                    # pool other than KV address their own keys (e.g. MAMBA
                    # node-boundary keys), not the KV page-hash array.
                    probe_source=source_name,
                )
            )
        return resolved


def _deepseek_v4_state_views(state_pools: list[Any], global_layers: list[int]):
    views = []
    for layer in global_layers:
        pool = state_pools[layer]
        state = pool.kv_score_buffer.kv_score
        ring = int(pool.ring_size)
        usable = state.shape[0] // ring * ring
        views.append(
            state.view(torch.uint8)
            .reshape(state.shape[0], -1)[:usable]
            .reshape(usable // ring, -1)
        )
    return views


def _with_packed_draft_mapping(
    layer_mapping: dict[int, int],
    *,
    target_device_layer_num: int,
    draft_layer_num: int,
) -> dict[int, int | Sequence[int]]:
    """Attach draft depth N to the same transfer layer as target layer N."""
    if draft_layer_num > len(layer_mapping):
        raise ValueError(
            f"Packed draft layers exceed the target transfer layer count: {draft_layer_num} > {len(layer_mapping)}."
        )
    # layer_mapping keys may be global layer ids (e.g. the mamba-hybrid
    # full_attention_layer_id_mapping interleaves with mamba layers), so pair
    # each draft depth with the Nth target component by enumeration order,
    # keyed back by its actual mapping key.
    mapping_keys = sorted(layer_mapping)
    result: dict[int, int | Sequence[int]] = dict(layer_mapping)
    for depth in range(draft_layer_num):
        result[mapping_keys[depth]] = (
            layer_mapping[mapping_keys[depth]],
            target_device_layer_num + depth,
        )
    return result


def _build_deepseek_v4_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _dsv4_indexer_regions,
        _resolve_deepseek_v4_layer_mappings,
    )

    mappings = _resolve_deepseek_v4_layer_mappings(kvcache)
    if getattr(kvcache, "_unified_kv", False) or isinstance(
        kvcache.c4_kv_pool, HiSparseC4DevicePool
    ):
        raise ValueError(
            "The direct external linker does not support unified-KV or HiSparse."
        )
    if kvcache.swa_page_size != page_size:
        raise ValueError(
            f"DeepSeek V4 SWA page size must match the tree page size: {kvcache.swa_page_size} != {page_size}."
        )

    draft_swa_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.swa_kv_pool.kv_buffer
    ]
    swa_mapping = _with_packed_draft_mapping(
        mappings.swa,
        target_device_layer_num=len(kvcache.swa_kv_pool.kv_buffer),
        draft_layer_num=len(draft_swa_buffers),
    )
    entries = [
        DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=kvcache.swa_kv_pool,
            components=[[*kvcache.swa_kv_pool.kv_buffer, *draft_swa_buffers]],
            layer_mapping=swa_mapping,
            page_size=page_size,
            rows_are_pages=True,
        )
    ]

    def add(name, source, pool, buffers, layer_mapping):
        if layer_mapping:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=source,
                    device_pool=pool,
                    components=[buffers],
                    layer_mapping=layer_mapping,
                    page_size=page_size,
                    rows_are_pages=True,
                )
            )

    add(
        PoolName.DEEPSEEK_V4_C4,
        PoolName.KV,
        kvcache.c4_kv_pool,
        kvcache.c4_kv_pool.kv_buffer,
        mappings.c4,
    )
    for region in _dsv4_indexer_regions(kvcache, page_size):
        add(
            region.name,
            PoolName.KV,
            kvcache.c4_indexer_kv_pool,
            region.device_buffers,
            mappings.c4,
        )
    add(
        PoolName.DEEPSEEK_V4_C128,
        PoolName.KV,
        kvcache.c128_kv_pool,
        kvcache.c128_kv_pool.kv_buffer,
        mappings.c128,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.SWA,
        kvcache.compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.compress_state_pools,
            mappings.c4_state_global_layers,
        ),
        mappings.c4_state,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        PoolName.SWA,
        kvcache.indexer_compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.indexer_compress_state_pools,
            mappings.c4_state_global_layers,
        ),
        mappings.c4_state,
    )
    return DevicePoolGroup(
        entries,
        mappings.transfer_layer_num,
        page_size,
        rank_replicated=True,
    )


def _build_dsa_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    if kvcache.page_size != page_size:
        raise ValueError(
            f"DSA KV page size must match the tree page size: {kvcache.page_size} != {page_size}."
        )
    num_layers = kvcache.layer_num
    if any(pool.page_size != page_size for pool in mtp_draft_device_pools):
        raise ValueError("DSA MTP page size must match the tree page size.")
    draft_kv_buffers = [
        buffer for pool in mtp_draft_device_pools for buffer in pool.kv_buffer
    ]
    draft_indexer_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.index_k_with_scale_buffer
    ]
    if len(draft_kv_buffers) != len(draft_indexer_buffers):
        raise ValueError("DSA MTP KV and indexer draft layer counts must match.")
    _warn_packed_draft_under_dcp(draft_kv_buffers)
    layer_mapping = _with_packed_draft_mapping(
        {layer: layer for layer in range(num_layers)},
        target_device_layer_num=num_layers,
        draft_layer_num=len(draft_kv_buffers),
    )
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.kv_buffer, *draft_kv_buffers]],
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
            # Per-rank sharded latent rows; see `_dcp_folding_index_mapper`.
            # (Only the target buffers fold; DSA draft pools above are
            # packed-replicated and not DCP-supported on the direct linker.)
            dcp_fold_slots=True,
            # Under DCP the tree page arriving here is the allocator's WIDENED
            # page (`params.page_size == tree page * dcp_size`, see
            # kv_cache_builder.py CacheInitParams) and translate_indices folds
            # it down to THIS entry's physical rows: one key covers
            # `page_size // dcp_size` folded slots (production DBGDBG:
            # keys=16 host_idx=1024 page_size=256 -> 64/key). The width is
            # derived LAZILY at I/O time by `_object_width()` — a
            # construction-time read races the parallel publish and silently
            # keeps the widened width (3984231c25 regression), so no
            # `slots_per_key` is declared here.
        ),
        DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.index_k_with_scale_buffer, *draft_indexer_buffers]],
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=True,
            # Index-K is GLOBAL-slot addressed under DCP (replicated buffer
            # sized `size * dcp_size` so all ranks compute identical top-k;
            # memory_pool.py DSATokenToKVPool.__init__) — its row space is
            # the raw virtual one and must NOT fold. (Production bug class:
            # the 0907 incident's over-folded INDEXER entry; see
            # `_dcp_folding_index_mapper`.) Because it does NOT fold, its
            # indices stay in the widened/global domain, where the entry's
            # page_size is ALREADY the widened width (`params.page_size ==
            # tree page * dcp_size`, kv_cache_builder.py CacheInitParams) —
            # one key spans exactly one widened page = `page_size` slots.
            # Multiplying by dcp_size again would over-declare the width
            # 4x (production DBGDBG round-trip evidence). The width is
            # derived LAZILY by `_object_width()` (== page_size for
            # non-folding entries), so no `slots_per_key` is declared.
        ),
    ]
    return DevicePoolGroup(
        entries,
        num_layers,
        page_size,
        rank_replicated=True,
        kv_source_shards_under_dcp=True,
    )


class MambaDevicePoolEntry(DevicePoolEntry):
    """MAMBA device pool entry carrying the duck-typed host attributes that
    ``MooncakeStore._get_hybrid_page_component_keys`` / ``_batch_io_v2`` read
    off a :class:`MambaPoolHost` in direct-linker mode."""

    def __init__(
        self,
        *,
        temporal_state_elem_size: int,
        conv_buffers: Sequence[torch.Tensor],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temporal_state_elem_size = temporal_state_elem_size
        self.conv_buffer = list(conv_buffers)


def _build_mamba_device_pool_group(
    kvcache: Any,
    page_size: int,
    params: Any,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    if page_size != 1:
        # The MAMBA entry addresses slots by node-boundary keys (page-aligned
        # node ends) and keeps page_size=1 rows internally, so any tree
        # page size works; the KV entry carries the tree page granularity.
        logger.warning(
            "Mamba direct linker running with tree page_size=%d (mamba slots stay slot-granular).",
            page_size,
        )

    mamba_pool = params.req_to_token_pool.mamba_pool
    mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
    full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
    union_layers = sorted(set(full_layer_mapping) | set(mamba_layer_mapping))

    state_components, conv_buffers, temporal_state_elem_size = (
        _build_mamba_state_components(mamba_pool)
    )

    # P2: pack the EAGLE draft pools alongside the target so restores cover
    # the draft's KV and index rows too. Without this, a store hit gives the
    # target model the context but the draft has never seen it (#31600 form):
    # speculative acceptance collapses toward token-by-token decoding.
    # Draft pools arrive as runner.token_to_kv_pool, whose flat-buffer shape
    # depends on the draft's OWN pool family:
    #   - DSA/MLA drafts (e.g. GLM-5.3-Flash NextN): DSATokenToKVPool with
    #     flat .kv_buffer (+ .index_k_with_scale_buffer sidecar).
    #   - Mamba-family drafts (Qwen3Next / NemotronH / Kimi / Bailing NextN):
    #     the HybridLinearKVPool WRAPPER, which exposes no .kv_buffer — the
    #     flat latent rows live on wrapper.full_kv_pool.kv_buffer (component
    #     groups are unwrapped into flat per-layer lists), and the wrapper
    #     carries no index sidecar. Unwrapping the WRAPPER (not the inner
    #     DSA pool) preserves the flat list shape the packed group consumes.
    def _draft_flat_kv_buffers(pool: Any) -> list[torch.Tensor]:
        kv_buffers = getattr(pool, "kv_buffer", None)
        if kv_buffers is None and _is_hybrid_linear_kv_pool(pool):
            inner_kv = getattr(pool.full_kv_pool, "kv_buffer", None)
            if inner_kv is not None:
                kv_buffers = [
                    buffer
                    for group in inner_kv
                    for buffer in (group if isinstance(group, list) else (group,))
                ]
        if kv_buffers is None:
            raise ValueError(
                f"Mamba-hybrid MTP draft pool exposes no flat KV buffers "
                f"(neither .kv_buffer nor .full_kv_pool.kv_buffer): {type(pool).__name__}."
            )
        return list(kv_buffers)

    def _validate_draft_row_coverage(
        target_buffers: Sequence[torch.Tensor],
        draft_buffers: Sequence[torch.Tensor],
        target_label: str,
        draft_label: str,
    ) -> None:
        """Reject undersized draft pools AT ASSEMBLY.

        The packed entry's row budget is the MINIMUM across its buffers, so a
        draft pool with fewer rows than the target would pass startup and only
        fail MID-FLIGHT (row-range ValueError during an offload/load burst at
        high concurrency). Fail loudly here instead, naming pools and shapes.
        """
        if not draft_buffers or not target_buffers:
            return
        target_rows = min(buffer.shape[0] for buffer in target_buffers)
        draft_rows = min(buffer.shape[0] for buffer in draft_buffers)
        if draft_rows < target_rows:
            raise ValueError(
                f"Packed {draft_label} rows ({draft_rows}) must cover {target_label} rows ({target_rows}); "
                f"target shapes={[tuple(buffer.shape) for buffer in target_buffers]}, "
                f"draft shapes={[tuple(buffer.shape) for buffer in draft_buffers]}."
            )

    draft_kv_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in _draft_flat_kv_buffers(pool)
    ]
    draft_indexer_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in getattr(pool, "index_k_with_scale_buffer", ())
    ]
    # A non-DSA draft legitimately has no index sidecar: requiring parity
    # would reject every mamba-family draft whose full pool is plain MHA/MLA.
    # Only enforce parity when BOTH buffer lists exist.
    if (
        draft_kv_buffers
        and draft_indexer_buffers
        and len(draft_kv_buffers) != len(draft_indexer_buffers)
    ):
        raise ValueError(
            "Mamba-hybrid MTP KV and indexer draft layer counts must match."
        )
    draft_layer_num = len(draft_kv_buffers)

    # Flatten the target's component groups BEFORE building the packed
    # mapping: the packed tuple indexes the flat [*target, *draft] buffer
    # list, so the draft domain must start at len(_target_kv_buffers). The
    # layer count coincides only for 1-latent-per-layer (MLA/DSA) targets;
    # an MHA-layout target (separate k/v groups) would flatten to 2*N buffers
    # and the packed tuple would index v-buffers instead of draft buffers.
    _target_kv_buffers = [
        buffer for group in _mamba_kv_components(kvcache) for buffer in group
    ]
    if draft_kv_buffers and len(_target_kv_buffers) != len(full_layer_mapping):
        raise NotImplementedError(
            "MHA-layout mamba-hybrid targets with the direct linker packed-draft "
            "mapping are not supported yet (k/v split buffers)"
        )
    kv_layer_mapping = (
        _with_packed_draft_mapping(
            dict(full_layer_mapping),
            target_device_layer_num=len(_target_kv_buffers),
            draft_layer_num=draft_layer_num,
        )
        if draft_kv_buffers
        else full_layer_mapping
    )

    # Pack target + draft per-layer buffers into ONE component group so the
    # packed mapping tuple (target_comp, N + depth) resolves both indices
    # inside the same group (same scheme as the pure-DSA KV entry).
    # Row coverage: a draft pool smaller than the target would only explode
    # mid-flight; refuse it at assembly.
    _validate_draft_row_coverage(
        _target_kv_buffers,
        draft_kv_buffers,
        target_label=PoolName.KV.value,
        draft_label="MTP draft KV",
    )
    _warn_packed_draft_under_dcp(draft_kv_buffers)
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*_target_kv_buffers, *draft_kv_buffers]],
            layer_mapping=kv_layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
            # The unified hybrid allocator's full side shards under DCP: slots
            # arrive in the widened logical space while this entry's latent
            # rows are per-rank. Keys are already rank-scoped via the
            # `_dcp{rank}_{size}` suffix namespace, so slot folding alone is
            # sound; dcp=1 mappers are the identity.
            dcp_fold_slots=True,
        ),
        MambaDevicePoolEntry(
            name=PoolName.MAMBA,
            indices_from_pool=PoolName.MAMBA,
            device_pool=mamba_pool,
            components=state_components,
            layer_mapping=_sorted_union_remapping(mamba_layer_mapping, union_layers),
            page_size=1,
            rows_are_pages=True,
            packed=False,
            temporal_state_elem_size=temporal_state_elem_size,
            conv_buffers=conv_buffers,
            # Mamba state is REPLICATED across DCP ranks and stays
            # slot-granular (its allocator sets shards_under_dcp=False), so
            # its slot ids are physical already — no folding.
        ),
    ]
    num_layers = len(union_layers)

    # DSA full-attention layers carry a persistent per-page index sidecar
    # (DSATokenToKVPool.index_k_with_scale_buffer). Without an INDEXER entry
    # here, an L3 restore would land latent KV rows on freshly allocated
    # slots whose index rows are zero/stale, and dsa_topk would then pick
    # wrong pages over the whole restored prefix (upstream #30057 form;
    # observed as gsm8k 0.94 -> 0.44 on pure-L3 restore). Mirror the pure-DSA
    # group: the INDEXER entry is KV-sourced (ALL_PAGES), so batch_exists_v2
    # intersects restorable prefixes with index availability at lookup time —
    # the native clamp-by-sidecar-hits equivalent of PR #31443. Index rows
    # live in fixed 64-token kernel pages (IndexKeyCache.KERNEL_PAGE_TOKENS),
    # so the entry only pages 1:1 with the tree when page_size == 64.
    full_kv_pool = kvcache.full_kv_pool
    index_buffers = getattr(full_kv_pool, "index_k_with_scale_buffer", None)
    if index_buffers and getattr(full_kv_pool, "use_dsa", False) and page_size == 64:
        _validate_draft_row_coverage(
            index_buffers,
            draft_indexer_buffers,
            target_label=PoolName.INDEXER.value,
            draft_label="MTP draft indexer",
        )
        entries.append(
            DevicePoolEntry(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                device_pool=full_kv_pool,
                components=[[*index_buffers, *draft_indexer_buffers]],
                layer_mapping=kv_layer_mapping,
                page_size=page_size,
                rows_are_pages=True,
                # Same global-slot index-K domain as the pure-DSA group above:
                # raw widened indices, no folding. The entry's page_size is
                # already the widened width (`params.page_size == tree page *
                # dcp_size`, kv_cache_builder.py CacheInitParams), so one key
                # = one widened page = `page_size` slots — do NOT multiply by
                # dcp_size again (over-declares 4x; see the DSA-group
                # comment). Width is derived LAZILY by `_object_width()`,
                # so no `slots_per_key` is declared.
            )
        )
        # num_layers stays len(union_layers): every entry's mapping keys are
        # drawn from that same union, and consumers iterate range(num_layers)
        # tolerating per-pool None mappings (mooncake_direct_linker.py
        # load_layer_wise: `meta is None -> continue`). The KV/INDEXER packed
        # mapping never exceeds len(full_layer_mapping)+draft layers, which
        # union (full keys + all mamba layers) always covers.
    elif index_buffers and getattr(full_kv_pool, "use_dsa", False):
        logger.warning(
            "Mamba direct linker: DSA index sidecar present but tree page_size=%d != IndexKeyCache kernel page 64; INDEXER entry skipped (L3 restores would be unsafe).",
            page_size,
        )
    return DevicePoolGroup(
        entries,
        num_layers,
        page_size,
        rank_replicated=False,
    )


def _mamba_kv_components(kvcache: Any) -> list[Sequence[torch.Tensor]]:
    """Full-attention buffers of a HybridLinearKVPool in component-group form."""
    pool = kvcache.full_kv_pool
    k_buffer = getattr(pool, "k_buffer", None)
    if k_buffer is not None:
        return [k_buffer, pool.v_buffer]
    return [pool.kv_buffer]


def _sorted_union_remapping(
    pool_mapping: dict[int, int], union_layers: Sequence[int]
) -> dict[int, int]:
    """Map transfer-layer ranks to pool-side layer indices.

    Transfer layers are numbered over the sorted union of component global
    layer ids; remap each global id to its rank in that union so
    ``get_prepared_layer_range_meta`` resolves it per transfer layer.
    """
    return {
        local: pool_mapping[gid]
        for local, gid in enumerate(union_layers)
        if gid in pool_mapping
    }


def _build_mamba_state_components(
    mamba_pool: Any,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor], int]:
    """Assemble the MAMBA device entry from the device MambaPool buffers."""
    state = mamba_pool.mamba_cache
    # Slot-first component order must match
    # MooncakeStore._get_hybrid_page_component_keys: temporal first, then
    # conv_0..conv_n; MambaPoolHost.get_page_buffer_meta drops the temporal
    # object for conv-only models (0-size state), mirror that here.
    temporal_state_elem_size = (
        int(
            state.temporal.numel()
            // state.temporal.shape[0]
            // max(1, state.temporal.shape[1])
        )
        if state.temporal.numel()
        else 0
    )
    components: list[list[torch.Tensor]] = []
    conv_buffers: list[torch.Tensor] = []
    if temporal_state_elem_size > 0:
        components.append([state.temporal[l] for l in range(state.temporal.shape[0])])
    for conv in state.conv:
        conv_buffers.append(conv)
        components.append([conv[l] for l in range(conv.shape[0])])
    if not components:
        raise ValueError("Mamba pool has neither temporal nor conv state buffers.")
    return (
        components,
        conv_buffers,
        temporal_state_elem_size,
    )


def _build_mamba_swa_device_pool_group(
    kvcache: Any, page_size: int, params: Any
) -> DevicePoolGroup:
    """Defensive variant for SWA + Mamba hybrid stacks (KV + SWA + MAMBA)."""
    if page_size != 1:
        logger.warning(
            "Mamba(SWA) direct linker running with tree page_size=%d (mamba slots stay slot-granular).",
            page_size,
        )

    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _swa_layer_mappings,
    )

    mamba_pool = params.req_to_token_pool.mamba_pool
    mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
    full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
    union_layers = sorted(
        set(full_layer_mapping) | set(swa_layer_mapping) | set(mamba_layer_mapping)
    )

    state_components, conv_buffers, temporal_state_elem_size = (
        _build_mamba_state_components(mamba_pool)
    )

    full_pool = kvcache.full_kv_pool
    swa_pool = kvcache.swa_kv_pool

    def kv_components(pool: Any) -> list[Sequence[torch.Tensor]]:
        k_buffer = getattr(pool, "k_buffer", None)
        if k_buffer is not None:
            return [k_buffer, pool.v_buffer]
        return [pool.kv_buffer]

    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=full_pool,
            components=kv_components(full_pool),
            layer_mapping=full_layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
        ),
        DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=swa_pool,
            components=kv_components(swa_pool),
            layer_mapping=swa_layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
            # SWA rows are replicated under DCP ("only FULL shards"; see
            # MultiEndedAllocator.__init__), so its slot ids are physical.
        ),
        MambaDevicePoolEntry(
            name=PoolName.MAMBA,
            indices_from_pool=PoolName.MAMBA,
            device_pool=mamba_pool,
            components=state_components,
            layer_mapping=_sorted_union_remapping(mamba_layer_mapping, union_layers),
            page_size=1,
            rows_are_pages=True,
            packed=False,
            temporal_state_elem_size=temporal_state_elem_size,
            conv_buffers=conv_buffers,
        ),
    ]
    return DevicePoolGroup(
        entries,
        len(union_layers),
        page_size,
        rank_replicated=False,
        # Mirror the plain-mamba group: the FULL side of a hybrid-linear stack
        # shards under DCP and needs slot folding. A pure-SWA stack keeps the
        # replicated KV slots untranslated; raising there would deny a
        # configuration DCP never promised (KV sharding is a
        # full-attention-latent property).
        kv_source_shards_under_dcp=_is_hybrid_linear_kv_pool(kvcache),
    )


def resolve_hybrid_device_pool_group(
    *,
    kvcache: Any,
    page_size: int,
    params: Any,
    components: set[ComponentType],
) -> DevicePoolGroup:
    """Materialize a direct-linker pool group through the hybrid registry."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _select_strategy,
    )

    return _select_strategy(kvcache, components).build_direct_linker_pool_group(
        kvcache=kvcache,
        params=params,
        page_size=page_size,
    )
