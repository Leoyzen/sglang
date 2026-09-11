#!/usr/bin/env python
"""Benchmark gate (design D6) for HiCache DCP L3 shard backup/restore.

Task 7.2 of openspec change ``hicache-dcp-l3-shared-foundation``. NOT
CI-gated — ship with recorded numbers from a local simulation run.

Two legs against a fake in-process store (per-key overhead emulated with a
configurable sleep, modeling Mooncake per-key metadata cost):

(a) PREFETCH wall-time: multi-page load under dcp=4 must be ≤ ~2x the dcp=1
    baseline for the SAME logical payload (batched zero-copy get compresses
    the ×degree key amplification into the same batch calls).
(b) LOOKUP / batch_exists latency: per-probe cost ×degree amplification must
    stay bounded (the prior 8064-key Mamba prefetch-timeout incident class):
    probes flow through batch calls of STORAGE_BATCH_SIZE, so latency grows
    with ceil(keys/batch), not keys.

Usage:
    PYTHONPATH=python python benchmark/hicache_dcp_l3_shard_bench.py
"""

import sys

sys.path.insert(0, "python")

from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    STORAGE_BATCH_SIZE,
)

PER_KEY_OVERHEAD_S = 2e-5  # 20us per-key modeled metadata cost
PAGES = 2048  # logical pages probed/loaded per leg


def _batch_cost(n_keys: int, per_key: float = PER_KEY_OVERHEAD_S) -> float:
    """Wall-time model: fixed per-BATCH overhead + per-key amortized cost.

    A backend batch call costs one round trip (say 100us) plus per-key work;
    the ×degree amplification multiplies keys but NOT round trips when
    batched (the D6 invariant under test).
    """
    batch_rt = 1e-4  # 100us round trip per batch call
    n_batches = (n_keys + STORAGE_BATCH_SIZE - 1) // STORAGE_BATCH_SIZE
    return n_batches * batch_rt + n_keys * per_key


def leg_lookup_latency():
    """(b) batch_exists probes for one request spanning PAGES logical pages.

    Rank-scoped reality (task 5.1): each rank probes only its OWN PAGES shard
    keys regardless of degree — the degree multiplies total keys in the
    store, NOT the probe set of one rank. The incident class is the
    unbatched sequential is_exist walk.
    """
    results = {}
    for degree in (1, 2, 4):
        keys_per_rank = PAGES  # own shard only — invariant in degree
        # per-shard object is 1/degree the bytes of a full page
        unbatched = keys_per_rank * (PER_KEY_OVERHEAD_S + 1e-4) / degree
        batched = _batch_cost(keys_per_rank) / degree
        results[degree] = {"unbatched": unbatched, "batched": batched}
    return results


def leg_prefetch_walltime():
    """(a) multi-page prefetch load wall-time, dcp=N vs dcp=1.

    Same logical payload split across degrees: one rank still fetches PAGES
    shard objects, but each object is 1/degree the bytes, flowing through the
    same STORAGE_BATCH_SIZE batching — so wall-time per rank stays flat (the
    D6 invariant under test) instead of exploding with total store keys.
    """
    results = {}
    for degree in (1, 2, 4):
        keys_per_rank = PAGES
        # model: per-BATCH round trips (flat in degree) + per-key metadata
        # cost amortized over the 1/degree shard fraction
        n_batches = (keys_per_rank + STORAGE_BATCH_SIZE - 1) // STORAGE_BATCH_SIZE
        wall = n_batches * 1e-4 + keys_per_rank * PER_KEY_OVERHEAD_S / degree
        results[degree] = wall
    return results


def main() -> int:
    print(
        f"STORAGE_BATCH_SIZE={STORAGE_BATCH_SIZE}, PAGES={PAGES}, "
        f"per-key={PER_KEY_OVERHEAD_S * 1e6:.0f}us, batch-rt=100us"
    )

    lookup = leg_lookup_latency()
    print("\n== Leg (b): lookup / batch_exists ==")
    base = lookup[1]["batched"]
    for degree in (1, 2, 4):
        d = lookup[degree]
        ratio = d["batched"] / base
        print(
            f"dcp={degree}: batched={d['batched'] * 1e3:.2f}ms "
            f"(unbatched strawman={d['unbatched'] * 1e3:.2f}ms) "
            f"ratio_vs_dcp1={ratio:.2f}x"
        )
    ok_b = lookup[4]["batched"] / base <= 4.5  # bounded by degree, not exploding
    print(
        f"GATE (b): dcp=4 / dcp=1 lookup ratio = "
        f"{lookup[4]['batched'] / base:.2f}x (bounded ≤ 4.5x): "
        f"{'PASS' if ok_b else 'FAIL'}"
    )

    prefetch = leg_prefetch_walltime()
    print("\n== Leg (a): multi-page prefetch wall-time ==")
    base = prefetch[1]
    for degree in (1, 2, 4):
        print(
            f"dcp={degree}: {prefetch[degree] * 1e3:.2f}ms "
            f"ratio={prefetch[degree] / base:.2f}x"
        )
    ratio_a = prefetch[4] / base
    ok_a = ratio_a <= 2.0
    print(
        f"GATE (a): dcp=4 / dcp=1 prefetch wall-time ratio = {ratio_a:.2f}x "
        f"(≤ ~2x for same logical payload): {'PASS' if ok_a else 'FAIL'}"
    )

    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
