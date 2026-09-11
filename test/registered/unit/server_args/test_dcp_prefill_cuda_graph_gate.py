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
"""Pin today's default prefill CUDA graph behavior under decode context
parallelism (openspec ``enable-dcp-bcg-prefill-cudagraph`` task 1.2 /
spec "Unvalidated backend keeps today's behavior").

Pinned contract (against the unmodified hook, PR #31532's disable rules):

* ``dcp_size > 1`` with no explicit prefill-backend lock auto-disables
  prefill CUDA graph for BOTH tc_piecewise and breakable, emitting the
  existing log messages —
  ``"Breakable CUDA graph is incompatible with decode context parallel
  (dcp_size > 1)"`` and the tc_piecewise silent declare (tc_piecewise
  declares without its own message; the DCP rule shares the generic
  cascade).
* A user-locked prefill backend bypasses the cascade entirely (the
  ``_cuda_graph_config_locked`` escape in
  ``apply_cuda_graph_compatibility``), whichever value the user chose.
* With ``dcp_size == 1`` nothing changes: both backends stay enabled.

These are regression pins: a future allowlist change (task 4.1) rewrites
them deliberately, and any accidental behavior drift fails here first.

    python -m pytest test/registered/unit/server_args/test_dcp_prefill_cuda_graph_gate.py -x -q
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cuda_graph_hook import (
    apply_cuda_graph_compatibility,
    disable_breakable_cudagraph_if_incompatible,
    disable_tc_piecewise_cudagraph_if_incompatible,
    handle_cuda_graph_config,
)
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_model_config(**extra):
    """Minimal stand-in satisfying model_config_of() reads in the hook.

    Llama-shaped: not KDA, not DeepSeek-V4, not multimodal, plain GQA —
    so the ONLY rule that fires is the one under test.
    """
    base = dict(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        is_piecewise_cuda_graph_disabled_model=False,
        is_multimodal=False,
        is_multimodal_piecewise_cuda_graph_supported=False,
        is_multimodal_breakable_cuda_graph_supported=False,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def _neuter_hardware_rules():
    """Neutralize the platform rules so CPU-only CI reaches the DCP rule.

    tc_piecewise's rules consult the platform context via
    ``get_platform()`` plus ``is_cpu``/``is_mps``/``current_platform``.
    All are scoped overrides: with an otherwise-clean config only the DCP
    rule can fire.
    """
    import sglang.srt.arg_groups.cuda_graph_hook as hook
    from sglang.srt.runtime_context import override_platform

    return (
        patch.object(hook, "is_cpu", return_value=False),
        patch.object(hook, "is_mps", return_value=False),
        override_platform(is_hip=False, is_npu=False, is_xpu=False, is_musa=False),
        patch.object(hook.current_platform, "is_out_of_tree", return_value=False),
    )


def _run_breakable_rule(dcp_size, *, locked=False, through_cascade=False):
    args = ServerArgs(model_path="dummy")
    args.dcp_size = dcp_size
    args._model_config = _make_model_config()
    args.cuda_graph_config = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.DISABLED),
        prefill=PhaseConfig(backend=Backend.BREAKABLE),
    )
    args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")} if locked else set()
    cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
    with cpu_patch, mps_patch, platform_override, oot_patch:
        if through_cascade:
            # The lock escape lives in the cascade dispatcher, not the rule.
            apply_cuda_graph_compatibility(args)
        else:
            disable_breakable_cudagraph_if_incompatible(args)
    return resolution_result(args, "cuda_graph_config").prefill.backend


def _run_tc_piecewise_rule(dcp_size):
    args = ServerArgs(model_path="dummy")
    args.dcp_size = dcp_size
    args._model_config = _make_model_config()
    args.cuda_graph_config = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.DISABLED),
        prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
    )
    args._cuda_graph_config_locked = set()
    cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
    with cpu_patch, mps_patch, platform_override, oot_patch:
        disable_tc_piecewise_cudagraph_if_incompatible(args)
    return resolution_result(args, "cuda_graph_config").prefill.backend


class TestDCPDisablesUnlockedPrefillCudaGraph(CustomTestCase):
    """dcp_size>1 + no lock → prefill CG disabled, with today's logs."""

    def test_breakable_auto_disabled_under_dcp(self):
        for dcp_size in (2, 4):
            with self.subTest(dcp_size=dcp_size):
                self.assertEqual(_run_breakable_rule(dcp_size), Backend.DISABLED)

    def test_tc_piecewise_auto_disabled_under_dcp(self):
        for dcp_size in (2, 4):
            with self.subTest(dcp_size=dcp_size):
                self.assertEqual(_run_tc_piecewise_rule(dcp_size), Backend.DISABLED)

    def test_dcp_size_one_keeps_both_backends(self):
        # Control: without DCP the same clean config keeps capture enabled.
        self.assertEqual(_run_breakable_rule(1), Backend.BREAKABLE)
        self.assertEqual(_run_tc_piecewise_rule(1), Backend.TC_PIECEWISE)

    def test_breakable_emits_existing_log_message(self):
        args = ServerArgs(model_path="dummy")
        args.dcp_size = 2
        args._model_config = _make_model_config()
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.BREAKABLE),
        )
        args._cuda_graph_config_locked = set()
        cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
        with cpu_patch, mps_patch, platform_override, oot_patch:
            with self.assertLogs(
                "sglang.srt.arg_groups.cuda_graph_hook", level=logging.WARNING
            ) as captured:
                disable_breakable_cudagraph_if_incompatible(args)
        joined = "\n".join(captured.output)
        self.assertIn("Breakable CUDA graph is incompatible with", joined)
        self.assertIn("decode context parallel (dcp_size > 1)", joined)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.DISABLED,
        )

    def test_tc_piecewise_rule_deletes_only_via_disabled_declare(self):
        # The tc_piecewise cascade declares DISABLED without a per-rule
        # message; pin that its outcome (not its silence) is the contract.
        backend = _run_tc_piecewise_rule(2)
        self.assertEqual(backend, Backend.DISABLED)


class TestLockedPrefillBackendEscapesTheCascade(CustomTestCase):
    """The _cuda_graph_config_locked escape: an explicit prefill backend
    skips the auto-disable cascade, whichever value the user chose."""

    def test_locked_breakable_survives_dcp(self):
        # The escape is checked by the cascade dispatcher before any rule runs.
        self.assertEqual(
            _run_breakable_rule(2, locked=True, through_cascade=True),
            Backend.BREAKABLE,
        )

    def test_locked_disabled_survives_dcp(self):
        # Even a user's explicit 'disabled' is respected (no duplicated declare).
        args = ServerArgs(model_path="dummy")
        args._model_config = _make_model_config()
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        )
        args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}
        apply_cuda_graph_compatibility(args)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.DISABLED,
        )

    def test_handle_cuda_graph_config_locked_breakable_kept_under_dcp(self):
        # End-to-end through the resolution entry: an explicit
        # --cuda-graph-backend-prefill lock survives the cascade under DCP.
        args = ServerArgs(
            model_path="dummy",
            dcp_size=2,
            cuda_graph_backend_prefill=Backend.BREAKABLE,
        )
        args._model_config = _make_model_config()
        with patch("sglang.srt.utils.is_cuda", return_value=True):
            handle_cuda_graph_config(args)
        self.assertIn((Phase.PREFILL, "backend"), args._cuda_graph_config_locked)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )


class TestUnlockedResolutionPathUnderDCP(CustomTestCase):
    """Through the full config entry (parse → compat cascade) the default
    prefill backend auto-disables under DCP without any explicit flags."""

    def _resolved(self, **overrides):
        args = ServerArgs(model_path="dummy", **overrides)
        args._model_config = _make_model_config()
        with patch("sglang.srt.utils.is_cuda", return_value=True):
            handle_cuda_graph_config(args)
        return args

    def test_default_resolution_disables_prefill_graph_under_dcp(self):
        args = self._resolved(dcp_size=2)
        prefill = resolution_result(args, "cuda_graph_config").prefill
        self.assertEqual(prefill.backend, Backend.DISABLED)
        # No explicit lock was involved: this is the auto-disable path.
        self.assertNotIn((Phase.PREFILL, "backend"), args._cuda_graph_config_locked)

    def test_default_resolution_keeps_prefill_graph_without_dcp(self):
        args = self._resolved(dcp_size=1)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )


if __name__ == "__main__":
    unittest.main()
