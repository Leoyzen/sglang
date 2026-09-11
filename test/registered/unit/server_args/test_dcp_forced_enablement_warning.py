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
"""Unit tests for the forced-enablement warning under DCP (openspec
``enable-dcp-bcg-prefill-cudagraph`` task 4.2 / design D3 / spec "Forced
enablement warns").

Pinned contracts:
  * User-locked prefill backend + dcp_size > 1 + backend NOT on the
    allowlist → startup WARNING naming DCP × prefill CUDA graph as
    unvalidated for that backend, and capture proceeds (backend survives).
  * Allowlisted backend + lock → NO warning (quiet, validated path).
  * dcp_size == 1 → NO warning regardless of allowlist membership.
  * The lock escape semantics themselves are untouched (backend kept).

    python -m pytest test/registered/unit/server_args/test_dcp_forced_enablement_warning.py -x -q
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cuda_graph_hook import (
    apply_cuda_graph_compatibility,
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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HOOK_LOG = "sglang.srt.arg_groups.cuda_graph_hook"


def _make_model_config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        is_piecewise_cuda_graph_disabled_model=False,
        is_multimodal=False,
        is_multimodal_piecewise_cuda_graph_supported=False,
        is_multimodal_breakable_cuda_graph_supported=False,
    )


def _neuter_hardware_rules():
    import sglang.srt.arg_groups.cuda_graph_hook as hook
    from sglang.srt.runtime_context import override_platform

    return (
        patch.object(hook, "is_cpu", return_value=False),
        patch.object(hook, "is_mps", return_value=False),
        override_platform(is_hip=False, is_npu=False, is_xpu=False, is_musa=False),
        patch.object(hook.current_platform, "is_out_of_tree", return_value=False),
    )


def _locked_args(*, backend: Backend, dcp_size: int, attention_backend):
    args = ServerArgs(model_path="dummy")
    args.dcp_size = dcp_size
    if attention_backend is not None:
        args.attention_backend = attention_backend
    args._model_config = _make_model_config()
    args.cuda_graph_config = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.DISABLED),
        prefill=PhaseConfig(backend=backend),
    )
    args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}
    return args


class TestForcedEnablementWarning(CustomTestCase):
    def test_unvalidated_locked_backend_warns_and_proceeds(self):
        args = _locked_args(
            backend=Backend.BREAKABLE, dcp_size=2, attention_backend="fa3"
        )
        cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
        with cpu_patch, mps_patch, platform_override, oot_patch:
            with self.assertLogs(_HOOK_LOG, level=logging.WARNING) as captured:
                apply_cuda_graph_compatibility(args)
        joined = "\n".join(captured.output)
        # Warning names the combination as unvalidated...
        self.assertIn("unvalidated", joined)
        self.assertIn("decode context parallelism", joined)
        self.assertIn("fa3", joined)
        # ...and the locked backend proceeds (no disable).
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

    def test_unvalidated_locked_tc_piecewise_warns_too(self):
        args = _locked_args(
            backend=Backend.TC_PIECEWISE, dcp_size=2, attention_backend="flashinfer"
        )
        cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
        with cpu_patch, mps_patch, platform_override, oot_patch:
            with self.assertLogs(_HOOK_LOG, level=logging.WARNING) as captured:
                apply_cuda_graph_compatibility(args)
        self.assertIn("unvalidated", "\n".join(captured.output))
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.TC_PIECEWISE,
        )

    def test_allowlisted_locked_backend_is_quiet(self):
        import sglang.srt.arg_groups.cuda_graph_hook as hook

        for name in sorted(hook.DCP_PREFILL_CG_ATTENTION_BACKEND_ALLOWLIST):
            with self.subTest(attention_backend=name):
                args = _locked_args(
                    backend=Backend.BREAKABLE, dcp_size=2, attention_backend=name
                )
                cpu_patch, mps_patch, platform_override, oot = _neuter_hardware_rules()
                with cpu_patch, mps_patch, platform_override, oot:
                    with patch.object(
                        hook.logger,
                        "warning",
                        side_effect=AssertionError("allowlisted+locked must not warn"),
                    ):
                        apply_cuda_graph_compatibility(args)
                self.assertEqual(
                    resolution_result(args, "cuda_graph_config").prefill.backend,
                    Backend.BREAKABLE,
                )

    def test_dcp_off_locked_backend_is_quiet(self):
        args = _locked_args(
            backend=Backend.BREAKABLE, dcp_size=1, attention_backend="fa3"
        )
        cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
        with cpu_patch, mps_patch, platform_override, oot_patch:
            with patch.object(
                __import__(
                    "sglang.srt.arg_groups.cuda_graph_hook", fromlist=["logger"]
                ).logger,
                "warning",
                side_effect=AssertionError("dcp_size=1 must not warn"),
            ):
                apply_cuda_graph_compatibility(args)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

    def test_unvalidated_locked_disabled_backend_does_not_warn(self):
        # Locking 'disabled' is not forcing capture on; no warning.
        args = _locked_args(
            backend=Backend.DISABLED, dcp_size=2, attention_backend="fa3"
        )
        cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
        with cpu_patch, mps_patch, platform_override, oot_patch:
            with patch.object(
                __import__(
                    "sglang.srt.arg_groups.cuda_graph_hook", fromlist=["logger"]
                ).logger,
                "warning",
                side_effect=AssertionError("disabled lock must not warn"),
            ):
                apply_cuda_graph_compatibility(args)

    def test_unvalidated_unlocked_backend_stays_on_disable_path(self):
        # No lock → the cascade handles it (auto-disable), NOT the warning
        # path: outcome is DISABLED, and no forced-enablement warning fires.
        args = ServerArgs(model_path="dummy")
        args.dcp_size = 2
        args.attention_backend = "fa3"
        args._model_config = _make_model_config()
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.BREAKABLE),
        )
        args._cuda_graph_config_locked = set()
        cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
        with cpu_patch, mps_patch, platform_override, oot_patch:
            with self.assertLogs(_HOOK_LOG, level=logging.WARNING) as captured:
                apply_cuda_graph_compatibility(args)
        joined = "\n".join(captured.output)
        self.assertNotIn("forced on by an explicit prefill-backend lock", joined)
        self.assertIn("Breakable CUDA graph is incompatible with", joined)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.DISABLED,
        )


if __name__ == "__main__":
    unittest.main()
