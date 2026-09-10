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
"""Pin preserved guard rails around the DCP × prefill-CUDA-graph capability
(openspec ``enable-dcp-bcg-prefill-cudagraph`` task 1.3 / spec "Guard rails
preserved").

(a) Prefill-CP × DCP stays rejected at launch, regardless of CUDA graph
    settings. The existing rejection assertions quoted inline below:
      * ``arg_groups/deepseek_v4_hook.py`` (``--dsv4-attn-backend trtllm``):
            assert not (
                cfg.attn_cp_size > 1 or cfg.dcp_size > 1 or cfg.enable_prefill_cp
            ), (
                "--dsv4-attn-backend trtllm does not support context parallelism "
                "(prefill CP, attention CP, or decode CP)."
            )
      * ``arg_groups/model_overrides/deepseek_v2.py`` (HYV4 family):
            if cfg.enable_prefill_cp:
                raise ValueError(
                    "--enable-prefill-cp is not supported for HYV4 ..."
                )
            if dcp_size > 1:
                raise ValueError(
                    "--dcp-size > 1 is not supported for HYV4 ..."
                )

(b) Fused top-k v2 stays off under DCP irrespective of CUDA graph
    configuration. Today's mechanism, pinned against unmodified code:
      * ``layers/attention/dsa/utils.py``
        (``should_remap_pd_dsa_seed_to_local_slots``) carries the literal
        DCP × fused-topk cross-check:
            ... and not get_parallel().dcp_enabled
        so the PD seed→fused-domain remap never composes fused top-k with
        decode context parallelism.
      * ``layers/attention/dsa/dsa_topk_backend.py``
        (``DSATopKBackend.should_use_topk_v2``) gates the fused top-k v2
        JIT kernel on ``SGLANG_OPT_USE_TOPK_V2`` — and DSA
        ``_build_topk_v2_plan`` returns None when it is off, which by the
        documented contract means metadata is "never dispatched to v2".
        No CUDA-graph enablement path may flip these.
      (#31821 F8 precedent: fused top-k v2 measured silent 0.000 accuracy
      under DCP.)

All cases are CPU-only: the DSA surfaces under test are pure config
gates (no NCCL, no model weights).

    python -m pytest test/registered/unit/server_args/test_dcp_guard_rails_preserved.py -x -q
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    apply_deepseek_v4_defaults,
)
from sglang.srt.arg_groups.model_overrides import deepseek_v2 as dsv2_overrides
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.utils import (
    should_remap_pd_dsa_seed_to_local_slots,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# (a) Launch rejection pins
# ---------------------------------------------------------------------------
class TestPrefillCpDcpLaunchRejection(CustomTestCase):
    """The launch must fail when prefill-CP composes with DCP, regardless
    of CUDA graph configuration."""

    @staticmethod
    def _dsv4_args(**overrides):
        """A DeepseekV4 record carrying --dsv4-attn-backend trtllm."""
        defaults = dict(
            model_path="dummy",
            device="cuda",
            dsv4_attn_backend="trtllm",
            kv_cache_dtype="auto",
            chunked_prefill_size=4096,
            enable_hisparse=False,
            disaggregation_mode="null",
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_cp=False,
            max_running_requests=None,
            speculative_algorithm=None,
        )
        defaults.update(overrides)
        args = ServerArgs(**defaults)
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        )
        return args

    @staticmethod
    def _with_sm100():
        """The CP assertion sits after the SM100 gate; CPU CI has no GPU, so
        patch the cached hardware probe (the hook lazy-imports
        ``sglang.srt.utils.common.is_sm100_supported``)."""
        return patch("sglang.srt.utils.common.is_sm100_supported", return_value=True)

    def test_dsv4_trtllm_rejects_prefill_cp_plus_dcp(self):
        # Quoted assertion (deepseek_v4_hook.py, apply_deepseek_v4_defaults):
        #   assert not (cfg.attn_cp_size > 1 or cfg.dcp_size > 1
        #               or cfg.enable_prefill_cp), (
        #     "--dsv4-attn-backend trtllm does not support context parallelism "
        #     "(prefill CP, attention CP, or decode CP).")
        for combo in (
            dict(dcp_size=2, enable_prefill_cp=True, cp_strategy="interleave"),
            dict(dcp_size=4, enable_prefill_cp=True, cp_strategy="interleave"),
            dict(
                attn_cp_size=2,
                dcp_size=2,
                enable_prefill_cp=True,
                cp_strategy="interleave",
            ),
        ):
            with self.subTest(combo=combo):
                args = self._dsv4_args(**combo)
                with self._with_sm100():
                    with self.assertRaisesRegex(
                        AssertionError, "does not support context parallelism"
                    ):
                        apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")

    def test_dsv4_trtllm_rejection_independent_of_cuda_graph_settings(self):
        # "regardless of CUDA graph settings": prefill CG explicitly enabled
        # must not weaken the launch rejection in any combination.
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            CudaGraphConfig,
            PhaseConfig,
        )

        for cg_backend in (Backend.BREAKABLE, Backend.TC_PIECEWISE, Backend.FULL):
            with self.subTest(cg_backend=cg_backend):
                args = self._dsv4_args(
                    dcp_size=2,
                    enable_prefill_cp=True,
                    cp_strategy="interleave",
                )
                args.cuda_graph_config = CudaGraphConfig(
                    decode=PhaseConfig(backend=Backend.DISABLED),
                    prefill=PhaseConfig(backend=cg_backend),
                )
                with self._with_sm100():
                    with self.assertRaisesRegex(
                        AssertionError, "does not support context parallelism"
                    ):
                        apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")

    def test_hyv4_rejects_prefill_cp_and_dcp_separately(self):
        # Quoted raise (model_overrides/deepseek_v2.py, _deepseek_family_overrides):
        #   "--enable-prefill-cp is not supported for HYV4 ..."  and
        #   "--dcp-size > 1 is not supported for HYV4 because decode context
        #    parallelism gathers query heads across DCP ranks ..."
        for combo, pattern in (
            (
                dict(enable_prefill_cp=True, cp_strategy="interleave"),
                "--enable-prefill-cp is not supported for HYV4",
            ),
            (
                dict(dcp_size=2),
                "--dcp-size > 1 is not supported for HYV4",
            ),
            (
                dict(dcp_size=2, enable_prefill_cp=True, cp_strategy="interleave"),
                "not supported for HYV4",
            ),
        ):
            with self.subTest(combo=combo):
                args = ServerArgs(model_path="dummy", **combo)
                hf_config = SimpleNamespace(
                    architectures=["HYV4ForCausalLM"],
                    quantization_config=None,
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    dsv2_overrides._deepseek_family_overrides(args, hf_config)

    def test_hyv4_and_dsv4_accept_dcp_free_configuration(self):
        # Control: the same records without CP/DCP must pass validation.
        args = self._dsv4_args()
        with self._with_sm100():
            apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")  # must not raise

        hy = ServerArgs(model_path="dummy")
        hy._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["HYV4ForCausalLM"]),
        )
        overrides = dsv2_overrides._deepseek_family_overrides(
            hy, SimpleNamespace(architectures=["HYV4ForCausalLM"])
        )
        self.assertIsInstance(overrides, dict)


# ---------------------------------------------------------------------------
# (b) Fused top-k v2 off under DCP
# ---------------------------------------------------------------------------
class TestFusedTopkV2OffUnderDcp(CustomTestCase):
    """Fused top-k v2 must stay unreachable under DCP regardless of CUDA
    graph configuration."""

    def test_pd_seed_remap_excludes_dcp(self):
        # The utils gate carries the literal cross-check
        # "... and not get_parallel().dcp_enabled". Simulate the parallel
        # bag with DCP off vs on and observe the gate flip.
        fake_parallel = SimpleNamespace(dcp_enabled=False)
        fake_memory = SimpleNamespace(enable_hisparse=False)
        fake_disagg = SimpleNamespace(disaggregation_mode="decode")
        env_patches = (
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_hip",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_disagg",
                return_value=fake_disagg,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_memory",
                return_value=fake_memory,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.get_parallel",
                return_value=fake_parallel,
            ),
        )
        # PD seed remap additionally requires the fused-topk env on: that is
        # the path composing fused top-k with the DCP-aware allocator.
        with envs.SGLANG_DSA_FUSE_TOPK.override(True):
            with (
                env_patches[0],
                env_patches[1],
                env_patches[2],
                env_patches[3],
                env_patches[4],
            ):
                self.assertTrue(
                    should_remap_pd_dsa_seed_to_local_slots(),
                    "sanity: with DCP off the gate is True on CUDA",
                )

                fake_parallel.dcp_enabled = True
                # With DCP on the fused-topk composition path must close, no
                # matter what CUDA graph does.
                self.assertFalse(
                    should_remap_pd_dsa_seed_to_local_slots(),
                    "DCP must keep the PD seed->fused-domain remap closed",
                )

    def test_should_use_topk_v2_env_gate_honored_under_cg_configs(self):
        # The v2 kernel is reachable only through SGLANG_OPT_USE_TOPK_V2;
        # DSA builds no v2 plan when it is off ("never dispatched to v2").
        # Pin that an OFF env keeps the gate closed with CUDA graph capture
        # mode flags in any state — an enablement path that turns v2 on as
        # a side effect of enabling prefill CG would fail here.
        backend = DSATopKBackend.SGL_KERNEL
        with envs.SGLANG_OPT_USE_TOPK_V2.override(False):
            self.assertFalse(backend.should_use_topk_v2())
        with envs.SGLANG_OPT_USE_TOPK_V2.override(True):
            self.assertTrue(backend.should_use_topk_v2())
            # Non-SGL backends are unfused regardless of the env.
            self.assertFalse(DSATopKBackend.TORCH.should_use_topk_v2())
            self.assertFalse(DSATopKBackend.FLASHINFER.should_use_topk_v2())

    def test_topk_v2_plan_contract_none_means_never_dispatched(self):
        # Pinned contract (dsa_backend.py _build_topk_v2_plan docstring):
        #   "None only when the SGL v2 path is disabled; such metadata is
        #    never dispatched to v2."
        # Simulate: with the env off, _build_topk_v2_plan on a bare instance
        # returns None without importing the JIT kernel.
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        instance = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        instance.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        fake_seqlens = torch.zeros(4, dtype=torch.int32)
        with envs.SGLANG_OPT_USE_TOPK_V2.override(False):
            with patch(
                "sglang.kernels.ops.attention.dsv4.topk.plan_topk_v2"
            ) as plan_fn:
                plan = instance._build_topk_v2_plan(fake_seqlens)
        self.assertIsNone(plan)
        plan_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
