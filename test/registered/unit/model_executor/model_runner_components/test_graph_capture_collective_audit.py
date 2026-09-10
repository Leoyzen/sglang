# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for the capture-time collective audit (openspec
``enable-dcp-bcg-prefill-cudagraph`` task 1.1).

The audit must flag a deliberately in-graph collective (simulated here with
a mock that reports "we are capturing" via ``torch.cuda`` mocking, since the
audit itself does not need a GPU) and pass a clean capture simulation. It
runs on CPU-only CI: the capture-detection surface
(``torch.cuda.is_current_stream_capturing``) is patched, and the collectives
themselves are fakes that never touch NCCL.

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_graph_capture_collective_audit.py -x -q
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components import (
    graph_capture_collective_audit as audit_mod,
)
from sglang.srt.model_executor.model_runner_components.graph_capture_collective_audit import (
    CaptureCollectiveAuditor,
    CollectiveCaptureAuditError,
    audit_active_capture_segment,
    capture_collective_audit_enabled,
    get_active_capture_collective_auditor,
    instrumented_capture_scope,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _capturing(enabled: bool):
    """Patch the surfaces the wrapper reads to detect an active capture.

    On a CPU-only runner ``torch.cuda.is_available()`` is False, so the
    wrapper's short-circuit would never consult the capturing flag; patching
    both surfaces keeps the audit logic exercised identically on CPU CI and
    a GPU host.
    """
    return (
        patch.object(audit_mod.torch.cuda, "is_available", return_value=True),
        patch.object(
            audit_mod.torch.cuda,
            "is_current_stream_capturing",
            return_value=enabled,
        ),
    )


class TestCaptureCollectiveAuditEnabled(CustomTestCase):
    def test_disabled_by_default_without_env_or_flag(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertFalse(capture_collective_audit_enabled())

    def test_enabled_via_debug_flag_param(self):
        self.assertTrue(capture_collective_audit_enabled(debug_flag=True))

    def test_enabled_via_env_var(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_DEBUG_CAPTURE_COLLECTIVE_AUDIT.override(True):
            self.assertTrue(capture_collective_audit_enabled())


class TestCaptureCollectiveAuditor(CustomTestCase):
    def setUp(self):
        self.auditor = CaptureCollectiveAuditor()
        self.originals = {}
        import torch.distributed as dist

        self.dist = dist
        for name in (
            "all_reduce",
            "all_gather_into_tensor",
            "broadcast",
        ):
            self.originals[name] = getattr(dist, name)

    def tearDown(self):
        # Restore whatever entry points any auditor installed in this test,
        # plus the module-level singleton so ordering cannot leak state.
        for auditor in (self.auditor, get_active_capture_collective_auditor()):
            if auditor is not None:
                auditor.uninstall()
        audit_mod._ACTIVE_AUDITOR = None
        for name, fn in self.originals.items():
            setattr(self.dist, name, fn)

    def _install_fake_collective(self, name):
        """Replace the collective with a recording fake and audit-wrap it."""
        calls = []

        def fake_collective(*args, **kwargs):
            calls.append(args)

        setattr(self.dist, name, fake_collective)
        # Install (re)wraps whatever is currently on the module.
        self.auditor.install()
        return calls

    def test_flags_deliberate_in_graph_collective(self):
        calls = self._install_fake_collective("all_reduce")
        env_patches = _capturing(enabled=True)
        with env_patches[0], env_patches[1]:
            self.auditor.begin_capture_scope("bucket-16")
            # Simulated captured segment body: a stray DCP KV all-gather.
            self.dist.all_reduce(torch.zeros(1))
            self.auditor.end_capture_scope()

        self.assertEqual(len(calls), 1, "fake collective must still run")
        with self.assertRaises(CollectiveCaptureAuditError) as ctx:
            self.auditor.assert_clean("prefill bucket 16")
        # Error names the op so the failure is actionable.
        self.assertIn("all_reduce", str(ctx.exception))
        self.assertIn("prefill bucket 16", str(ctx.exception))

    def test_passes_clean_capture_simulation(self):
        self._install_fake_collective("all_reduce")
        env_patches = _capturing(enabled=True)
        with env_patches[0], env_patches[1]:
            self.auditor.begin_capture_scope("bucket-16")
            # Clean segment: no collective call at all.
            self.auditor.end_capture_scope()

        self.assertEqual(self.auditor.recorded_collectives(), [])
        # Must not raise.
        self.auditor.assert_clean("clean prefill capture")

    def test_collective_outside_capture_is_not_flagged(self):
        calls = self._install_fake_collective("broadcast")
        env_patches = _capturing(enabled=False)
        with env_patches[0], env_patches[1]:
            self.auditor.begin_capture_scope("bucket-16")
            # Eager (warmup) collective outside any captured segment.
            self.dist.broadcast(torch.zeros(1), src=0)
            self.auditor.end_capture_scope()

        self.assertEqual(len(calls), 1)
        self.auditor.assert_clean("warmup collectives are legal")

    def test_install_swaps_and_uninstall_restores_entry_points(self):
        self.auditor.install()
        wrapped = getattr(self.dist, "all_reduce")
        self.assertIsNot(wrapped, self.originals["all_reduce"])

        self.auditor.uninstall()
        restored = getattr(self.dist, "all_reduce")
        self.assertIs(restored, self.originals["all_reduce"])

    def test_uninstall_is_idempotent(self):
        self.auditor.install()
        self.auditor.uninstall()
        # Second uninstall must not raise.
        self.auditor.uninstall()
        self.assertIs(getattr(self.dist, "all_reduce"), self.originals["all_reduce"])

    def test_multiple_captures_accumulate_until_reset(self):
        calls = self._install_fake_collective("all_gather_into_tensor")
        env_patches = _capturing(enabled=True)
        with env_patches[0], env_patches[1]:
            for _ in range(2):  # two buckets
                self.auditor.begin_capture_scope("bucket")
                self.dist.all_gather_into_tensor(torch.zeros(2), torch.zeros(1))
                self.auditor.end_capture_scope()

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.auditor.recorded_collectives()), 2)
        with self.assertRaises(CollectiveCaptureAuditError):
            self.auditor.assert_clean()
        self.auditor.reset()
        self.auditor.assert_clean()


class TestInstrumentedCaptureScope(CustomTestCase):
    def setUp(self):
        self.originals = {}
        import torch.distributed as dist

        for name in ("all_reduce", "all_gather"):
            self.originals[name] = getattr(dist, name)

    def tearDown(self):
        auditor = get_active_capture_collective_auditor()
        if auditor is not None:
            auditor.uninstall()
        import torch.distributed as dist

        for name, fn in self.originals.items():
            setattr(dist, name, fn)

    def test_noop_when_audit_disabled(self):
        # A stale singleton from an earlier test in this process must not
        # keep recording once the scope says disabled: the scope factory
        # alone decides whether this capture segment is audited.
        with patch.dict("os.environ", {}, clear=False):
            scope = instrumented_capture_scope("label")
            self.assertIs(
                type(scope).__name__,
                "nullcontext",
                "disabled audit must hand back a no-op scope",
            )
            with scope:
                pass

    def test_scope_flags_collective_invoked_inside(self):
        import torch.distributed as dist

        def fake_all_reduce(*args, **kwargs):
            return args[0] if args else None

        setattr(dist, "all_reduce", fake_all_reduce)
        env_patches = _capturing(enabled=True)
        with env_patches[0], env_patches[1]:
            with instrumented_capture_scope("bucket-4", debug_flag=True) as audit:
                self.assertIsNotNone(audit)
                dist.all_reduce(torch.zeros(1))
            auditor = get_active_capture_collective_auditor()
            self.assertIsNotNone(auditor)
            with self.assertRaises(CollectiveCaptureAuditError):
                auditor.assert_clean("bucket-4")

    def test_scope_passes_clean_capture(self):

        env_patches = _capturing(enabled=True)
        with env_patches[0], env_patches[1]:
            with instrumented_capture_scope("bucket-4", debug_flag=True):
                pass  # clean segment
            auditor = get_active_capture_collective_auditor()
            self.assertIsNotNone(auditor)
            auditor.assert_clean("clean capture")

    def test_audit_active_capture_segment_helper_resets_after_raise(self):
        import torch.distributed as dist

        def fake_all_gather(tensor_list, tensor, *args, **kwargs):
            if tensor_list:
                tensor_list[0] = tensor
            return None

        setattr(dist, "all_gather", fake_all_gather)
        env_patches = _capturing(enabled=True)
        with env_patches[0], env_patches[1]:
            with instrumented_capture_scope("bucket-8", debug_flag=True):
                dist.all_gather([], torch.zeros(1))
                with self.assertRaises(CollectiveCaptureAuditError) as ctx:
                    audit_active_capture_segment("prefill bucket 8")
                self.assertIn("all_gather", str(ctx.exception))
            # The helper resets even when it raises, so the next segment
            # starts from a clean slate.
            auditor = get_active_capture_collective_auditor()
            self.assertEqual(auditor.recorded_collectives(), [])


if __name__ == "__main__":
    unittest.main()
