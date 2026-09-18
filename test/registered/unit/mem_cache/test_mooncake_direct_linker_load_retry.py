"""Unit tests for Mooncake direct-linker range-get retry fail-soft."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeStore:
    """Returns one outcome per call, sized to the keys requested.

    ``outcomes`` is a list of per-call dicts mapping key -> transferred bytes
    (missing key defaults to OK). The last outcome repeats if more calls come.
    """

    def __init__(self, outcomes, session_start_result=None):
        self.outcomes = list(outcomes)
        self.range_calls = []
        self.session_start_calls = []
        self.session_end_calls = []
        self.session_start_result = session_start_result

    def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        self.range_calls.append(list(keys))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        return [outcome.get(key, 74880) for key in keys]

    def batch_get_session_start(self, keys):
        self.session_start_calls.append(list(keys))
        if self.session_start_result is None:
            return [0] * len(keys)
        return self.session_start_result

    def batch_get_session_end(self, keys):
        self.session_end_calls.append(list(keys))
        return None


class _FakeFailedGetCache:
    def __init__(self):
        self.batches = []

    def update_batch(self, ready, failed):
        self.batches.append((list(ready), list(failed)))


def _make_linker(store, *, attempts=5, budget_s=30.0):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.storage = SimpleNamespace(store=store)
    linker.range_get_retry_attempts = attempts
    linker.range_get_retry_budget_s = budget_s
    return linker


class TestRangeGetRetry(CustomTestCase):
    def test_retry_recovers_transient_partial_failure(self):
        # First attempt: k1 fails with -600. Second: all succeed.
        store = _FakeStore([{"k1": -600}, {}])
        linker = _make_linker(store)
        meta = ([100, 200, 300], [[74880], [74880], [74880]], [0, 0, 0])

        with patch(
            "sglang.srt.mem_cache.storage.mooncake_store."
            "mooncake_direct_linker.time.sleep"
        ):
            linker._range_get_with_retry("deepseek_v4_c2", 2, ["k0", "k1", "k2"], meta)

        # Only the failed entry is re-issued on the retry.
        self.assertEqual(store.range_calls, [["k0", "k1", "k2"], ["k1"]])

    def test_only_failed_entries_are_reissued(self):
        store = _FakeStore([{"b": -600, "c": -600}, {"b": -600}, {}])
        linker = _make_linker(store)
        meta = ([0, 0, 0, 0], [[74880], [74880], [74880], [74880]], [0, 0, 0, 0])

        with patch(
            "sglang.srt.mem_cache.storage.mooncake_store."
            "mooncake_direct_linker.time.sleep"
        ):
            linker._range_get_with_retry("deepseek_v4_c2", 0, list("abcd"), meta)

        self.assertEqual(store.range_calls, [["a", "b", "c", "d"], ["b", "c"], ["b"]])

    def test_persistent_failure_raises_and_negative_caches(self):
        store = _FakeStore([{"k1": -600}])
        failed_cache = _FakeFailedGetCache()
        linker = _make_linker(store, attempts=3)
        linker.storage.failed_get_cache = failed_cache
        meta = ([0, 0, 0], [[74880], [74880], [74880]], [0, 0, 0])

        with patch(
            "sglang.srt.mem_cache.storage.mooncake_store."
            "mooncake_direct_linker.time.sleep"
        ):
            with self.assertRaisesRegex(RuntimeError, "attempts=3"):
                linker._range_get_with_retry(
                    "deepseek_v4_c2", 2, ["k0", "k1", "k2"], meta
                )

        self.assertEqual(len(store.range_calls), 3)
        self.assertEqual(failed_cache.batches, [([], ["k1"])])

    def test_session_restart_on_later_attempts(self):
        store = _FakeStore([{"k1": -600}, {"k1": -600}, {"k1": -600}, {}])
        linker = _make_linker(store)
        meta = ([0, 0], [[74880], [74880]], [0, 0])

        with patch(
            "sglang.srt.mem_cache.storage.mooncake_store."
            "mooncake_direct_linker.time.sleep"
        ):
            linker._range_get_with_retry("deepseek_v4_c2", 0, ["k0", "k1"], meta)

        # attempt 3 triggers session end+start for the failed key only.
        self.assertEqual(store.session_end_calls, [["k1"]])
        self.assertEqual(store.session_start_calls, [["k1"]])


class TestLayerWiseLoadCounter(CustomTestCase):
    def test_fail_propagates_as_runtime_error(self):
        counter = LayerWiseLoadCounter(num_layers=2)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, RuntimeError("boom"))
        with self.assertRaisesRegex(RuntimeError, "layer-wise KV load failed"):
            counter.wait_until(0)

    def test_complete_resolves_wait(self):
        counter = LayerWiseLoadCounter(num_layers=2)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.complete(index, 0)
        counter.complete(index, 1)
        counter.wait_until(0)
        counter.wait_until(1)


if __name__ == "__main__":
    unittest.main(verbosity=3)
