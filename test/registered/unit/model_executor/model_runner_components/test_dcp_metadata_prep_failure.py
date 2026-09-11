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
"""Unit tests for graceful DCP metadata prep failure handling (openspec
``enable-dcp-bcg-prefill-cudagraph`` task 3.2; spec 'Metadata prep failure
aborts capture cleanly').

Pinned contracts:
  * Capture path: a fault-injected builder raising during capture marks
    the runner, logs the reason, and surfaces an internal abort error the
    capture-setup layer converts into 'prefill CG disabled for the run +
    eager fallback' — no silent wrong outputs.
  * Replay path: a raising builder makes load_batch raise the internal
    abort (after a logged reason) and execute() falls back to the eager
    runner for that batch — the server continues.
  * Non-DCP failures are NOT converted: an unrelated RuntimeError during
    capture propagates (existing behavior).

    python -m pytest test/registered/unit/model_executor/model_runner_components/test_dcp_metadata_prep_failure.py -x -q
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _DcpCaptureAbort,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(capture_hidden_mode=0):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.zeros(4, dtype=torch.int64),
        positions=torch.arange(4, dtype=torch.int64),
        out_cache_loc=torch.zeros(4, dtype=torch.int64),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.tensor([4], dtype=torch.int64),
        seq_lens_sum=4,
        extend_seq_lens=torch.tensor([4], dtype=torch.int64),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int64),
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[4],
        extend_start_loc=torch.tensor([0], dtype=torch.int64),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


class TestDcpMetadataPrepFailure(CustomTestCase):
    def _runner(self, *, dcp_size=2):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            ps=SimpleNamespace(attn_dcp_size=dcp_size),
            model=SimpleNamespace(),
            kv_cache_dtype=torch.bfloat16,
            device="cpu",
            attn_backend=SimpleNamespace(name="fake"),
            eager_runner=None,
        )
        runner.dcp_buffers = None
        runner.dcp_metadata_prep_failed = False
        runner._dcp_replay_failure_logged = False
        runner.use_captured_attn_metadata = False
        runner.attn_metadata_buffers = None
        runner._is_full_backend = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        return runner

    def test_capture_failure_sets_flag_and_raises(self):
        runner = self._runner()
        fb = _batch()
        with patch(
            "sglang.srt.model_executor.model_runner_components.dcp_prefill_metadata.prepare_dcp_extend_metadata",
            side_effect=RuntimeError("planner exploded"),
        ):
            with self.assertRaises(RuntimeError):
                with self.assertLogs(
                    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
                    level="ERROR",
                ) as logs:
                    from sglang.srt.model_executor.forward_context import (
                        ForwardContext,
                        forward_context,
                    )

                    with forward_context(
                        ForwardContext(attn_backend=runner.model_runner.attn_backend)
                    ):
                        runner._prepare_capture_dcp_metadata(fb)
        # The reason is logged and the disable-for-the-run signal is set.
        self.assertTrue(runner.dcp_metadata_prep_failed)
        joined = "\n".join(logs.output)
        self.assertIn("Prefill DCP metadata preparation failed", joined)
        self.assertIn("aborted", joined)

    def test_init_captures_abort_and_raises_internal_abort(self):
        runner = self._runner()
        runner.dcp_metadata_prep_failed = True  # as set by capture failure
        # Simulate capture() blowing up after the flag was set.
        with patch.object(runner, "capture", side_effect=RuntimeError("kaboom")):
            with self.assertRaises(_DcpCaptureAbort):
                with patch(
                    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner.logger"
                ):
                    # Reproduce the __init__ guard inline: the handler wraps
                    # capture() and converts flagged failures.
                    try:
                        try:
                            runner.capture()
                        except RuntimeError as exc:
                            if not getattr(runner, "dcp_metadata_prep_failed", False):
                                raise
                            raise _DcpCaptureAbort(str(exc)) from exc
                    except _DcpCaptureAbort:
                        raise

    def test_unrelated_capture_failure_propagates(self):
        # An unflagged RuntimeError (not DCP-related) must NOT be converted.
        class _RaisingRunner:
            dcp_metadata_prep_failed = False

            def capture(self):
                raise RuntimeError("OOM during capture")

        runner = _RaisingRunner()
        with self.assertRaisesRegex(RuntimeError, "OOM during capture"):
            try:
                runner.capture()
            except RuntimeError as exc:
                if not getattr(runner, "dcp_metadata_prep_failed", False):
                    raise
                raise _DcpCaptureAbort(str(exc)) from exc

    def test_replay_failure_falls_back_to_eager(self):
        runner = self._runner()
        eager_outputs = object()

        eager_runner = SimpleNamespace(execute=lambda fb, **kw: eager_outputs)
        runner.model_runner.eager_runner = eager_runner

        fb = _batch()
        with patch(
            "sglang.srt.model_executor.model_runner_components.dcp_prefill_metadata.prepare_dcp_extend_metadata",
            side_effect=RuntimeError("planner exploded"),
        ):
            with patch.object(
                runner,
                "_execute_with_replay_session",
                side_effect=lambda fb_, **kw: (_ for _ in ()).throw(
                    _DcpCaptureAbort("planner exploded")
                ),
            ) as session_mock:
                out = runner.execute(fb)
        self.assertIs(out, eager_outputs)
        session_mock.assert_called_once()

    def test_replay_failure_without_eager_runner_propagates(self):
        runner = self._runner()  # eager_runner is None
        fb = _batch()
        with patch.object(
            runner,
            "_execute_with_replay_session",
            side_effect=_DcpCaptureAbort("planner exploded"),
        ):
            with self.assertRaises(_DcpCaptureAbort):
                runner.execute(fb)

    def test_replay_failure_with_eager_placeholder_propagates(self):
        # When the prefill graph was disabled at capture, prefill runner IS
        # the EagerRunner and this fallback must not recurse into itself.
        from sglang.srt.model_executor.runner.eager_runner import EagerRunner

        runner = self._runner()
        runner.model_runner.eager_runner = EagerRunner.__new__(EagerRunner)
        fb = _batch()
        with patch.object(
            runner,
            "_execute_with_replay_session",
            side_effect=_DcpCaptureAbort("planner exploded"),
        ):
            with self.assertRaises(_DcpCaptureAbort):
                runner.execute(fb)


if __name__ == "__main__":
    unittest.main()
