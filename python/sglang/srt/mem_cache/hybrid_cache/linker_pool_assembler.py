"""Materialize hybrid device pools for external cache linkers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)
from dataclasses import replace
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


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
    ):
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.packed = packed
        self._index_mapper = index_mapper
        self._page_offsets = torch.arange(page_size)
        self._row_span = 1 if rows_are_pages else page_size

        if not self.components or any(not component for component in self.components):
            raise ValueError(f"Device pool {name} has no storage buffers.")
        self.kv_buffer = [buffer for group in self.components for buffer in group]
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)

        self.buffer_meta = [
            [
                (
                    buffer.data_ptr(),
                    buffer.stride(0) * buffer.element_size(),
                    buffer.nbytes // buffer.shape[0] * self._row_span,
                )
                for buffer in component
            ]
            for component in self.components
        ]

        self._component_offsets = []
        offset = 0
        for component in self.buffer_meta:
            if not packed:
                offset = 0
            offsets = []
            for _, _, size in component:
                offsets.append(offset)
                offset += size
            self._component_offsets.append(offsets)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._index_mapper(indices) if self._index_mapper else indices

    def _rows(self, indices: torch.Tensor) -> list[int]:
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(f"Pool {self.name} got {slots.numel()} indices, expected a multiple of page_size={self.page_size}.")
        if not slots.numel():
            return []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(pages, starts[:, None] + self._page_offsets):
            raise ValueError(f"Pool {self.name} requires aligned contiguous pages.")
        rows = starts.div(self.page_size, rounding_mode="floor") if self._row_span == 1 else starts
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            raise ValueError(f"Pool {self.name} row range [{first_row}, {last_row}) exceeds buffer shapes {[tuple(buffer.shape) for buffer in self.kv_buffer]}.")
        return rows.tolist()

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices)
        ptrs = [base_ptr + row * row_stride for row in rows for component in self.buffer_meta for base_ptr, row_stride, _ in component]
        sizes = [size for _ in rows for component in self.buffer_meta for _, _, size in component]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices)

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        mapped = self.layer_mapping.get(layer)
        if mapped is None:
            return None
        buffer_indices = [mapped] if isinstance(mapped, int) else list(mapped)

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            for buffer_index in buffer_indices:
                base_ptr, row_stride, size = component[buffer_index]
                items.append((base_ptr, row_stride, size, offsets[buffer_index]))

        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items]
            row_sizes = [size for _, _, size, _ in items]
            row_offsets = [offset for _, _, _, offset in items]
            if self.packed:
                ptrs.append(row_ptrs)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
            else:
                ptrs.extend([[value] for value in row_ptrs])
                sizes.extend([[value] for value in row_sizes])
                offsets.extend([[value] for value in row_offsets])
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
                    host_indices=(self.entry_map[name].translate_indices(indices) if indices is not None else None),
                    keys=list(source.keys),
                    hit_policy=(PoolHitPolicy.ALL_PAGES if source_name == PoolName.KV else source.hit_policy),
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
        views.append(state.view(torch.uint8).reshape(state.shape[0], -1)[:usable].reshape(usable // ring, -1))
    return views


def _with_packed_draft_mapping(
    layer_mapping: dict[int, int],
    *,
    target_device_layer_num: int,
    draft_layer_num: int,
) -> dict[int, int | Sequence[int]]:
    """Attach draft depth N to the same transfer layer as target layer N."""
    if draft_layer_num > len(layer_mapping):
        raise ValueError(f"Packed draft layers exceed the target transfer layer count: {draft_layer_num} > {len(layer_mapping)}.")
    # layer_mapping keys may be global layer ids (e.g. the mamba-hybrid
    # full_attention_layer_id_mapping interleaves with mamba layers), so pair
    # each draft depth with the Nth target component by enumeration order,
    # keyed back by its actual mapping key.
    mapping_keys = sorted(layer_mapping)
    result: dict[int, int | Sequence[int]] = dict(layer_mapping)
    for depth in range(draft_layer_num):
        result[mapping_keys[depth]] = (layer_mapping[mapping_keys[depth]], target_device_layer_num + depth)
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
    if getattr(kvcache, "_unified_kv", False) or isinstance(kvcache.c4_kv_pool, HiSparseC4DevicePool):
        raise ValueError("The direct external linker does not support unified-KV or HiSparse.")
    if kvcache.swa_page_size != page_size:
        raise ValueError(f"DeepSeek V4 SWA page size must match the tree page size: {kvcache.swa_page_size} != {page_size}.")

    draft_swa_buffers = [buffer for pool in mtp_draft_device_pools for buffer in pool.swa_kv_pool.kv_buffer]
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
        raise ValueError(f"DSA KV page size must match the tree page size: {kvcache.page_size} != {page_size}.")
    num_layers = kvcache.layer_num
    if any(pool.page_size != page_size for pool in mtp_draft_device_pools):
        raise ValueError("DSA MTP page size must match the tree page size.")
    draft_kv_buffers = [buffer for pool in mtp_draft_device_pools for buffer in pool.kv_buffer]
    draft_indexer_buffers = [buffer for pool in mtp_draft_device_pools for buffer in pool.index_k_with_scale_buffer]
    if len(draft_kv_buffers) != len(draft_indexer_buffers):
        raise ValueError("DSA MTP KV and indexer draft layer counts must match.")
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
        ),
        DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.index_k_with_scale_buffer, *draft_indexer_buffers]],
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=True,
        ),
    ]
    return DevicePoolGroup(entries, num_layers, page_size, rank_replicated=True)


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
        logger.warning("Mamba direct linker running with tree page_size=%d (mamba slots stay slot-granular).", page_size)

    mamba_pool = params.req_to_token_pool.mamba_pool
    mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
    full_layer_mapping = dict(kvcache.full_attention_layer_id_mapping)
    union_layers = sorted(set(full_layer_mapping) | set(mamba_layer_mapping))

    state_components, conv_buffers, temporal_state_elem_size = _build_mamba_state_components(mamba_pool)

    # P2: pack the EAGLE draft pools alongside the target so restores cover
    # the draft's KV and index rows too. Without this, a store hit gives the
    # target model the context but the draft has never seen it (#31600 form):
    # speculative acceptance collapses toward token-by-token decoding.
    # Draft pools are HybridLinearKVPool wrappers. Their .kv_buffer /
    # .index_k_with_scale_buffer are flat per-layer tensor lists (same shape
    # the pure-DSA group consumes), so take them directly instead of unwrapping
    # to full_kv_pool — unwrapping changes kv_buffer into component groups.
    draft_kv_buffers = [buffer for pool in mtp_draft_device_pools for buffer in pool.kv_buffer]
    draft_indexer_buffers = [buffer for pool in mtp_draft_device_pools for buffer in getattr(pool, "index_k_with_scale_buffer", ())]
    if draft_kv_buffers and len(draft_kv_buffers) != len(draft_indexer_buffers):
        raise ValueError("Mamba-hybrid MTP KV and indexer draft layer counts must match.")
    draft_layer_num = len(draft_kv_buffers)
    kv_layer_mapping = (
        _with_packed_draft_mapping(
            dict(full_layer_mapping),
            target_device_layer_num=len(full_layer_mapping),
            draft_layer_num=draft_layer_num,
        )
        if draft_kv_buffers
        else full_layer_mapping
    )

    # Pack target + draft per-layer buffers into ONE component group so the
    # packed mapping tuple (target_comp, N + depth) resolves both indices
    # inside the same group (same scheme as the pure-DSA KV entry). NOTE:
    # _mamba_kv_components returns component GROUPS (lists of per-layer
    # tensors), so flatten groups first instead of unpacking them as units.
    _target_kv_buffers = [buffer for group in _mamba_kv_components(kvcache) for buffer in group]
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*_target_kv_buffers, *draft_kv_buffers]],
            layer_mapping=kv_layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
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
        entries.append(
            DevicePoolEntry(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                device_pool=full_kv_pool,
                components=[[*index_buffers, *draft_indexer_buffers]],
                layer_mapping=kv_layer_mapping,
                page_size=page_size,
                rows_are_pages=True,
            )
        )
        num_layers = max(num_layers, len(full_layer_mapping) + draft_layer_num)
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


def _sorted_union_remapping(pool_mapping: dict[int, int], union_layers: Sequence[int]) -> dict[int, int]:
    """Map transfer-layer ranks to pool-side layer indices.

    Transfer layers are numbered over the sorted union of component global
    layer ids; remap each global id to its rank in that union so
    ``get_prepared_layer_range_meta`` resolves it per transfer layer.
    """
    return {local: pool_mapping[gid] for local, gid in enumerate(union_layers) if gid in pool_mapping}


def _build_mamba_state_components(
    mamba_pool: Any,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor], int]:
    """Assemble the MAMBA device entry from the device MambaPool buffers."""
    state = mamba_pool.mamba_cache
    # Slot-first component order must match
    # MooncakeStore._get_hybrid_page_component_keys: temporal first, then
    # conv_0..conv_n; MambaPoolHost.get_page_buffer_meta drops the temporal
    # object for conv-only models (0-size state), mirror that here.
    temporal_state_elem_size = int(state.temporal.numel() // state.temporal.shape[0] // max(1, state.temporal.shape[1])) if state.temporal.numel() else 0
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


def _build_mamba_swa_device_pool_group(kvcache: Any, page_size: int, params: Any) -> DevicePoolGroup:
    """Defensive variant for SWA + Mamba hybrid stacks (KV + SWA + MAMBA)."""
    if page_size != 1:
        logger.warning("Mamba(SWA) direct linker running with tree page_size=%d (mamba slots stay slot-granular).", page_size)

    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _swa_layer_mappings,
    )

    mamba_pool = params.req_to_token_pool.mamba_pool
    mamba_layer_mapping = dict(params.req_to_token_pool.mamba_map)
    full_layer_mapping, swa_layer_mapping = _swa_layer_mappings(kvcache)
    union_layers = sorted(set(full_layer_mapping) | set(swa_layer_mapping) | set(mamba_layer_mapping))

    state_components, conv_buffers, temporal_state_elem_size = _build_mamba_state_components(mamba_pool)

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
