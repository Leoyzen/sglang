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
"""Unit tests for the backend-conditional DCP prefill CUDA graph gate
(openspec ``enable-dcp-bcg-prefill-cudagraph`` task 4.1 / design D3 /
spec "Allowlisted backend enables capture").

The blanket ``dcp_size > 1`` auto-disable is replaced by:
  * dcp_size > 1 + backend NOT on ``DCP_PREFILL_CG_ATTENTION_BACKEND_ALLOWLIST``
    → today's auto-disable with the EXACT existing log message (unvalidated
    backends keep today's behavior);
  * dcp_size > 1 + allowlisted backend → the rule passes through, the
    backend stays enabled (capture proceeds through the Section 2/3
    metadata wiring);
  * dcp_size == 1 → unaffected in all cases.

Backend identity is resolved at hook time via ``attention_backends_of``
(the same access the hook's other backend-aware rules use; split fields
fall back to the base backend). An unresolved/auto backend is treated as
NOT allowlisted — conservative, today's behavior.

    python -m pytest test/registered/unit/server_args/test_dcp_backend_allowlist_gate.py -x -q
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cuda_graph_hook import (
    DCP_PREFILL_CG_ATTENTION_BACKEND_ALLOWLIST,
    _dcp_prefill_backend_allowlisted,
    apply_cuda_graph_compatibility,
    disable_breakable_cudagraph_if_incompatible,
    disable_tc_piecewise_cudagraph_if_incompatible,
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

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _make_model_config():
    """Llama-shaped stand-in: the DCP rule is the only rule that can fire."""
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


def _run_rule(backend: Backend, dcp_size: int, attention_backend=None):
    args = ServerArgs(model_path="dummy")
    args.dcp_size = dcp_size
    if attention_backend is not None:
        args.attention_backend = attention_backend
    args._model_config = _make_model_config()
    args.cuda_graph_config = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.DISABLED),
        prefill=PhaseConfig(backend=backend),
    )
    args._cuda_graph_config_locked = set()
    cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
    with cpu_patch, mps_patch, platform_override, oot_patch:
        if backend == Backend.BREAKABLE:
            disable_breakable_cudagraph_if_incompatible(args)
        else:
            disable_tc_piecewise_cudagraph_if_incompatible(args)
    return resolution_result(args, "cuda_graph_config").prefill.backend


class TestDcpBackendAllowlistGate(CustomTestCase):
    def test_unvalidated_backend_still_auto_disables(self):
        # An allowlist-unaware (unvalidated) backend keeps today's behavior:
        # disabled for both prefill CG backends, exact existing log message.
        self.assertEqual(
            _run_rule(Backend.BREAKABLE, 2, attention_backend="fa3"),
            Backend.DISABLED,
        )
        self.assertEqual(
            _run_rule(Backend.TC_PIECEWISE, 2, attention_backend="fa3"),
            Backend.DISABLED,
        )

    def test_unvalidated_backend_breakable_log_message_unchanged(self):
        args = ServerArgs(model_path="dummy")
        args.dcp_size = 2
        args.attention_backend = "flashinfer"  # not on the allowlist
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

    def test_allowlisted_backend_passes_the_rule(self):
        # A backend on the allowlist is NOT blocked by the DCP rule; other
        # rules would still apply, but with a clean config capture proceeds.
        for backend in (Backend.BREAKABLE, Backend.TC_PIECEWISE):
            for name in sorted(DCP_PREFILL_CG_ATTENTION_BACKEND_ALLOWLIST):
                with self.subTest(backend=backend, attention_backend=name):
                    self.assertEqual(
                        _run_rule(backend, 2, attention_backend=name), backend
                    )

    def test_fake_allowlisted_name_via_patched_lookup(self):
        # Allowlist mechanism test: a FAKE backend name injected into the
        # allowlist lets an otherwise-identical config pass the DCP rule.
        import sglang.srt.arg_groups.cuda_graph_hook as hook

        fake = frozenset({"fake_allowlisted"})
        with patch.object(hook, "DCP_PREFILL_CG_ATTENTION_BACKEND_ALLOWLIST", fake):
            self.assertEqual(
                _run_rule(Backend.BREAKABLE, 2, attention_backend="fake_allowlisted"),
                Backend.BREAKABLE,
            )
            self.assertEqual(
                _run_rule(
                    Backend.TC_PIECEWISE, 2, attention_backend="fake_allowlisted"
                ),
                Backend.TC_PIECEWISE,
            )
            # And a non-matching name still disables.
            self.assertEqual(
                _run_rule(Backend.BREAKABLE, 2, attention_backend="other"),
                Backend.DISABLED,
            )

    def test_dcp_size_one_unaffected_by_allowlist(self):
        self.assertEqual(
            _run_rule(Backend.BREAKABLE, 1, attention_backend="fa3"), Backend.BREAKABLE
        )
        self.assertEqual(
            _run_rule(Backend.TC_PIECEWISE, 1, attention_backend="fa3"),
            Backend.TC_PIECEWISE,
        )

    def test_split_prefill_field_falls_back_to_base_backend(self):
        # attention_backends_of's contract: an unset prefill_attention_backend
        # falls back to attention_backend — pin that the allowlist lookup
        # honors it (trtllm_mla on the BASE field still counts).
        args = ServerArgs(model_path="dummy")
        args.attention_backend = "trtllm_mla"
        args.prefill_attention_backend = None
        self.assertTrue(_dcp_prefill_backend_allowlisted(args))

    def test_prefill_split_field_takes_precedence(self):
        # A split prefill field overrides the base for the allowlist check.
        args = ServerArgs(model_path="dummy")
        args.attention_backend = "flashinfer"
        args.prefill_attention_backend = "trtllm_mla"
        self.assertTrue(_dcp_prefill_backend_allowlisted(args))

        args.prefill_attention_backend = "fa3"
        self.assertFalse(_dcp_prefill_backend_allowlisted(args))

    def test_unresolved_backend_is_conservatively_not_allowlisted(self):
        # attention_backend unset (auto): the helper must treat it as NOT
        # allowlisted so today's disable behavior holds.
        args = ServerArgs(model_path="dummy")
        self.assertFalse(_dcp_prefill_backend_allowlisted(args))


class TestCascadeEndToEndWithAllowlist(CustomTestCase):
    def test_cascade_locked_backend_still_escapes_entirely(self):
        # The locked-backend escape happens in the cascade dispatcher before
        # any rule runs and is untouched by the allowlist change.
        args = ServerArgs(model_path="dummy")
        args.dcp_size = 2
        args.attention_backend = "fa3"
        args._model_config = _make_model_config()
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.BREAKABLE),
        )
        args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")}
        cpu_patch, mps_patch, platform_override, oot_patch = _neuter_hardware_rules()
        with cpu_patch, mps_patch, platform_override, oot_patch:
            apply_cuda_graph_compatibility(args)
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )


if __name__ == "__main__":
    unittest.main()
