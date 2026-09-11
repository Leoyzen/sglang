# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the hierarchical KV cache."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
    use_mla_backend,
)

logger = logging.getLogger(__name__)


def handle_hicache(server_args: Any):
    """Normalize hicache-related knobs into a valid runtime configuration.

    Resolution order:
    1) Layout <-> I/O compatibility for direct conflicts.
    2) Storage <-> layout compatibility (may rewrite layout).
    """
    cfg = resolving_view(server_args)
    if cfg.enable_unified_cache_external_linker:
        if cfg.enable_hierarchical_cache:
            raise ValueError(
                "--enable-unified-cache-external-linker and "
                "--enable-hierarchical-cache are mutually exclusive."
            )
        if cfg.hicache_storage_backend is not None:
            raise ValueError(
                "--enable-unified-cache-external-linker does not use "
                "--hicache-storage-backend."
            )
        # Direct-linker DCP gating (task 1.4): the UMBP/mori linker lacks
        # DCP-scoped keys, so it must never default to dcp_rank=0 and write
        # rank-colliding shard objects. Checked at argument-resolution time,
        # before the linker class is ever instantiated.
        if cfg.dcp_size > 1 and cfg.unified_cache_external_linker_backend == "mori":
            raise NotImplementedError(
                "UMBP direct linker (--enable-unified-cache-external-linker "
                "with backend 'mori') does not support --dcp-size > 1: the "
                "UMBP store has no DCP-scoped key support, so shards from "
                "different ranks would collide on dcp_rank=0 keys. Run UMBP "
                "with --dcp-size 1."
            )
        # Mooncake direct linker under DCP is SUPPORTED (no shard flag
        # required): its component keys are composed from the per-rank
        # `mla_suffix`/`mha_suffix`, which carry the `_dcp{rank}_{size}`
        # namespace, so the KEY space is rank-scoped with no cross-rank
        # collisions; and the pool group folds the widened logical slots to
        # per-rank physical rows at `resolve_transfers`
        # (`linker_pool_assembler._dcp_folding_index_mapper`), mirroring
        # `HostKVCache.maybe_dcp_kernel_indices`. The mamba state pool is
        # replicated and slot-granular, so it is untouched by the folding.
        return

    # Skip all normalization when neither hicache nor decode-offload path is active.
    if not (
        cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
        or (
            cfg.disaggregation_mode == "decode"
            and cfg.disaggregation_decode_retraction_backup in (None, "host_pool")
        )
    ):
        return

    validate_hicache_host_memory_mode(server_args)

    # Step 1: Initial layout-io compatibility normalization.
    resolve_layout_io_compatibility(server_args)

    # Step 2: Storage-layout normalization without changing io backend.
    resolve_storage_layout_compatibility(server_args)

    # Step 3: DCP compatibility for the L2 (device<->host) path.
    resolve_hicache_dcp_compatibility(server_args)


def handle_hicache_ratio_default(server_args: Any):
    """Default the host/device ratio per host memory mode.

    Runs before the dummy-model boundary: direct HostKVCache consumers
    (unit fixtures, dummy-model launches) must never see a None ratio.
    buffer_only stages in flight rather than retaining, so it needs only
    enough to cover the write backlog plus parked prefetches.

    A decode server keeps the ratio unset here: kv_cache_builder resolves
    it against the retraction-backup backend (1.0 for host_pool, else 2.0).
    """
    cfg = resolving_view(server_args)
    if cfg.hicache_ratio is None and cfg.disaggregation_mode != "decode":
        declare_resolution(
            server_args,
            "_handle_hicache_ratio_default",
            hicache_ratio=(
                1.2 if cfg.hicache_host_memory_mode == "buffer_only" else 2.0
            ),
        )


# The exact message raised today when L3 meets --dcp-size > 1 with the flag
# off. Preserved VERBATIM as a regression gate: flipping the flag's default or
# editing this text silently changes a documented ops failure signal.
_L3_DCP_NOT_IMPLEMENTED_MESSAGE = (
    "--hicache-storage-backend (L3) with --dcp-size > 1 is not "
    "supported yet: under DCP each rank holds a distinct "
    "interleaved MLA KV shard, so the rank-0-only replicated-MLA "
    "backup and the storage keys must become dcp_rank-aware "
    "first. Run HiCache+DCP with L1/L2 only."
)


def resolve_hicache_dcp_compatibility(server_args: Any):

    cfg = resolving_view(server_args)
    if cfg.dcp_size <= 1 or not cfg.enable_hierarchical_cache:
        return

    # ---- Hard exclusions: checked BEFORE the flag can admit L3 ----
    # Per design D5, the feature flag only relaxes the L3×DCP fail-fast below;
    # every exclusion in this block stays unconditional so enabling the flag
    # can never silently unblock an unrelated gap.

    # Speculative draft/spec host pools have no DCP index translation.
    if cfg.speculative_algorithm not in (None, "DSPARK"):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 only supports DSPARK speculative "
            "decoding; other draft-model host pools have no DCP index "
            "translation."
        )
    if cfg.enable_lmcache:
        raise NotImplementedError(
            "--enable-lmcache with --dcp-size > 1 is not supported: "
            "LMCache has no DCP-aware index translation."
        )
    if cfg.enable_hisparse:
        raise NotImplementedError(
            "--enable-hisparse with --dcp-size > 1 is not supported: the "
            "HiSparse host pool is constructed without DCP translation."
        )
    if not use_mla_backend(server_args):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 is only supported for MLA models: "
            "the index translation lives in MLATokenToKVPoolHost, and the "
            "MHA host pool has none."
        )
    # PD disaggregation with L3 under DCP: the KV-transfer relayout and the
    # L3 backup path are not DCP-aware. L1/L2-only HiCache keeps today's
    # behavior (the PD hook warns on prefill DCP, rejects decode DCP).
    l3_backend = cfg.hicache_storage_backend is not None
    if l3_backend and cfg.disaggregation_mode != "null":
        raise NotImplementedError(
            "HiCache L3 storage with --dcp-size > 1 is not supported under "
            f"PD disaggregation (disaggregation_mode={cfg.disaggregation_mode!r}): "
            "KV transfer relayout and the L3 backup path are not DCP-aware. "
            "Run PD disaggregation without --hicache-storage-backend."
        )
    # UMBP direct-linker rejection lives in handle_hicache (the external-linker
    # route early-returns before this function) and in the linker's own
    # __init__ (defense in depth, umbp_direct_linker.py).
    # Head splitting (heterogeneous tp_lcm_size) rebases keys on head-shard
    # ranks; combined with DCP slot interleaving the two translations are not
    # jointly defined.
    if cfg.dcp_size > 1 and _hicache_tp_lcm_size(server_args) is not None:
        raise NotImplementedError(
            "--hicache-storage-backend-extra-config tp_lcm_size (head "
            "splitting) combined with --dcp-size > 1 is not supported: head "
            "sharding and DCP slot interleaving have no jointly defined "
            "key layout. Drop tp_lcm_size or run with --dcp-size 1."
        )
    # Hole-set (TRAILING_PAGES) pools — Mamba state, SWA windows — restore
    # per-rank SETS with holes, so scalar min() consensus over hit lengths
    # could commit an illegal page boundary (see PoolTransferResult).
    if _hicache_has_hole_set_pool(server_args):
        raise NotImplementedError(
            "HiCache L3 with --dcp-size > 1 does not support models with "
            "hole-set (TRAILING_PAGES) pools (Mamba state / SWA windows): "
            "per-rank restorable sets need a per-page mask-intersection "
            "consensus, planned as a follow-up. Track hicache-dcp-l3-shared-"
            "foundation for the mask-intersection change."
        )
    # Combined NSA context-parallel and DCP axes have undefined joint key
    # semantics under L3.
    if cfg.attn_cp_size > 1 and cfg.dcp_size > 1:
        raise NotImplementedError(
            "HiCache with attn_cp_size > 1 combined with dcp_size > 1 is "
            "not supported: NSA context parallel (page-slice sharding) and "
            "decode context parallel (slot-interleaved sharding) are two "
            "independent axes whose joint L3 key/backup semantics are "
            "undefined. Enable at most one."
        )

    # ---- Flag-gated L3 admission ----
    if l3_backend and not cfg.enable_hicache_dcp_shard:
        raise NotImplementedError(_L3_DCP_NOT_IMPLEMENTED_MESSAGE)
    if cfg.enable_hicache_dcp_shard and not l3_backend:
        # A silently-ignored flag is exactly the hidden behavior toggle this
        # capability forbids: fail the launch, don't warn.
        raise ValueError(
            "--enable-hicache-dcp-shard requires an L3 storage backend: "
            "pass --hicache-storage-backend (e.g. mooncake, file) or drop "
            "the flag."
        )

    amplification = _hicache_dcp_amplification_factor(server_args)
    logger.info(
        "HiCache + DCP enabled (L1/L2%s): host pool uses widened logical "
        "slot accounting with per-rank physical translation at the transfer "
        "boundary (dcp_size=%d)%s",
        "+L3 shard storage" if l3_backend else " only",
        cfg.dcp_size,
        (
            "; DCP-L3 key-count amplification ≈ degree × estimated pages = "
            f"{cfg.dcp_size} × {amplification['pages']} pages ⇒ ~"
            f"{amplification['factor']} objects (capacity planning knob: "
            "batching, see STORAGE_BATCH_SIZE)"
            if l3_backend
            else ""
        ),
    )


def _hicache_tp_lcm_size(server_args: Any) -> Optional[int]:
    """tp_lcm_size from the storage extra config, if one was supplied."""
    raw = resolving_view(server_args).hicache_storage_backend_extra_config
    if not raw:
        return None
    try:
        if raw.startswith("@"):
            import json as _json
            from pathlib import Path as _Path

            path = raw[1:]
            ext = _Path(path).suffix.lower()
            with open(path, "rb" if ext == ".toml" else "r") as f:
                if ext == ".json":
                    config = _json.load(f)
                elif ext == ".toml":
                    import tomllib

                    config = tomllib.load(f)
                elif ext in (".yaml", ".yml"):
                    import yaml

                    config = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"Unsupported config file {path} (config format: {ext})"
                    )
        else:
            import json as _json

            config = _json.loads(raw)
        value = config.get("tp_lcm_size")
        return int(value) if value is not None else None
    except (ValueError, TypeError, OSError):
        # Unparsable configs fail loudly in their own parser later; this
        # guard only feeds the head-splitting exclusion.
        return None


def _hicache_has_hole_set_pool(server_args: Any) -> bool:
    """Whether the model's cache contains Mamba/SWA (TRAILING_PAGES) pools.

    Mirror of the pool-population logic that builds the device pools: a hybrid
    SSM model owns a Mamba pool, a hybrid SWA model owns an SWA pool — both
    have hole-set (trailing-pages) existence semantics under batch_exists_v2.
    """
    try:
        model_config = model_config_of(server_args)
    except Exception:
        # Unresolvable model config is not this gate's problem; the model
        # loader will fail on its own terms later.
        return False
    from sglang.srt.configs.hybrid_arch import (
        glm5_next_config,
        hybrid_gdn_config,
        hybrid_lightning_config,
        kimi_linear_config,
        linear_attn_model_spec,
        mamba2_config,
    )

    spec = linear_attn_model_spec(model_config)
    # `is True` (not truthiness): these are real bools in production, and
    # strict comparison keeps mocked model configs from tripping the gate.
    return (
        hybrid_gdn_config(model_config) is not None
        or mamba2_config(model_config) is not None
        or (spec.uses_mamba_radix_cache is True if spec is not None else False)
        or kimi_linear_config(model_config) is not None
        or glm5_next_config(model_config) is not None
        or hybrid_lightning_config(model_config) is not None
        or getattr(model_config, "is_hybrid_swa", False) is True
    )


def _hicache_dcp_amplification_factor(server_args: Any) -> dict:
    """Object-count amplification estimate for the DCP-L3 startup log line.

    The stored-object count under shard-native keys scales with
    degree × pages; the page estimate uses the host-pool capacity proxy
    (max_total_tokens widened to logical slots) so operators can pre-size the
    store. These are estimates for log visibility, not hard caps.
    """
    cfg = resolving_view(server_args)
    # page_size is resolved later in the pipeline (Optional here), so fall
    # back to 1 — the estimate only feeds a log line, not sizing.
    page_size = max(1, cfg.page_size or 1) * cfg.dcp_size
    est_pages = max(1, cfg.max_total_tokens or 0) // page_size
    # Per-page objects: one KV shard object per component (k, or MLA's packed
    # k), plus one per auxiliary side pool — approximated as 1 object/page
    # per rank for the KV pool itself, the dominant term.
    return {
        "degree": cfg.dcp_size,
        "pages": est_pages,
        "factor": cfg.dcp_size * est_pages,
    }


def resolve_layout_io_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_mem_layout == "page_first_direct"
        and cfg.hicache_io_backend == "kernel"
    ):
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_io_backend="direct",
        )
        logger.warning(
            "Kernel io backend does not support page first direct layout, switching to direct io backend"
        )

    if cfg.hicache_mem_layout == "page_first" and cfg.hicache_io_backend == "direct":
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_mem_layout="page_first_direct",
        )
        logger.warning(
            "Page first layout is not supported with direct IO backend, switching to page first direct layout"
        )


def resolve_storage_layout_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_storage_backend != "mooncake"
        or cfg.hicache_mem_layout != "layer_first"
    ):
        return

    if cfg.hicache_io_backend == "direct":
        new_layout = "page_first_direct"
    elif cfg.hicache_io_backend == "kernel":
        new_layout = "page_first"
    else:
        # Keep current behavior for unknown backends (e.g., kernel_ascend).
        new_layout = cfg.hicache_mem_layout

    declare_resolution(
        server_args,
        "_resolve_storage_layout_compatibility",
        hicache_mem_layout=new_layout,
    )
    logger.warning(
        f"Mooncake storage backend does not support layer_first layout, "
        f"switching to {new_layout} layout for {cfg.hicache_io_backend} io backend"
    )


def validate_hicache_host_memory_mode(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.hicache_host_memory_mode not in ("cache", "buffer_only"):
        raise ValueError(
            "hicache_host_memory_mode must be 'cache' or 'buffer_only', "
            f"got {cfg.hicache_host_memory_mode!r}"
        )

    # Both modes are defaulted upstream (a decode server resolves the
    # ratio later, in kv_cache_builder), so this fires only if that
    # defaulting regresses -- never build an unsized host pool.
    if (
        cfg.hicache_size <= 0
        and cfg.hicache_ratio is None
        and cfg.disaggregation_mode != "decode"
    ):
        raise ValueError(
            f"--hicache-host-memory-mode {cfg.hicache_host_memory_mode} "
            "requires a host pool size: pass --hicache-size or "
            "--hicache-ratio."
        )

    if cfg.hicache_host_memory_mode == "cache":
        return

    if cfg.hicache_storage_backend is None:
        raise ValueError(
            "--hicache-host-memory-mode buffer_only requires a storage backend "
            "(--hicache-storage-backend): host memory is only a staging buffer "
            "and all cached data lives in storage."
        )
    if cfg.hicache_write_policy == "write_back":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only does not support "
            "--hicache-write-policy write_back; use write_through or "
            "write_through_selective."
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only is not supported on "
            "decode instances: the decode-side prefetch and offload paths "
            "bypass the buffer-mode pipeline, fetching without its prefix "
            "context and never consuming its staged holds. Prefill "
            "instances share the standard scheduler path and are supported."
        )
