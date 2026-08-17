"""GPU numerical parity test for DSV4 DCP (Decode Context Parallel).

Launches a DeepSeek-V4-Flash server with ``dcp_size=1`` (baseline) and
``dcp_size=2`` (sharded), sends identical prompts, and compares the
returned top-logprobs to ensure DCP sharding does not change decode
outputs beyond fp8 tolerance.

Registration mirrors ``test_kimi_linear_dcp4.py``: ``register_cuda_ci``
with a multi-GPU runner config.
"""

import os
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="2-gpu-large")

# Env-overridable model path.  Default is the small DSV4-Flash HF model.
DSV4_MODEL = os.environ.get(
    "SGLANG_DSV4_DCP_TEST_MODEL", "deepseek-ai/DeepSeek-V4-Flash"
)

# Tolerance for fp8 logprob comparison.
ATOL = 1e-2
RTOL = 1e-2

PROMPTS = [
    # Short prompt
    "What is 2+2?",
    # Longer context prompt
    "Explain the concept of context parallelism in large language model "
    "inference. Cover KV cache sharding, communication patterns, and the "
    "trade-off between memory savings and communication overhead. "
    "Conclude with a one-sentence summary.",
]


def _has_two_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _launch(model: str, base_url: str, dcp_size: int, port: int):
    return popen_launch_server(
        model,
        base_url=f"http://127.0.0.1:{port}",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
        other_args=[
            "--tp-size",
            "2",
            "--dcp-size",
            str(dcp_size),
            "--attention-backend",
            "deepseek_v4",
            "--trust-remote-code",
            "--random-seed",
            "0",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mem-fraction-static",
            "0.80",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dsv4-prefill-backend",
            "flashmla_sparse_q8",
            "--moe-runner-backend",
            "flashinfer_mxfp4",
        ],
    )


def _get_logprobs(base_url: str, prompt: str) -> list[float]:
    """Send a prompt and return the top-1 logprob for each generated token."""
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 1,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    logprobs = []
    for choice in data.get("choices", []):
        lp = choice.get("logprobs", {})
        for token_lp in lp.get("content", []):
            if token_lp and token_lp.get("logprob") is not None:
                logprobs.append(token_lp["logprob"])
    return logprobs


@unittest.skipUnless(_has_two_gpus(), "DSV4 DCP parity test requires ≥2 GPUs")
class TestDsv4DcpDecodeParity(CustomTestCase):
    """Compare decode logprobs between dcp_size=1 and dcp_size=2."""

    @classmethod
    def setUpClass(cls):
        cls.port_baseline = 27117
        cls.port_dcp2 = 27118
        cls.url_baseline = f"http://127.0.0.1:{cls.port_baseline}"
        cls.url_dcp2 = f"http://127.0.0.1:{cls.port_dcp2}"

        cls.proc_baseline = _launch(
            DSV4_MODEL, cls.url_baseline, dcp_size=1, port=cls.port_baseline
        )
        cls.proc_dcp2 = _launch(
            DSV4_MODEL, cls.url_dcp2, dcp_size=2, port=cls.port_dcp2
        )

    @classmethod
    def tearDownClass(cls):
        for proc in [
            getattr(cls, "proc_baseline", None),
            getattr(cls, "proc_dcp2", None),
        ]:
            if proc:
                kill_process_tree(proc.pid, wait_timeout=60)

    def test_short_prompt_logprob_parity(self):
        """Short prompt: logprobs from dcp_size=2 must match baseline."""
        lp_base = _get_logprobs(self.url_baseline, PROMPTS[0])
        lp_dcp2 = _get_logprobs(self.url_dcp2, PROMPTS[0])

        self.assertTrue(lp_base, "baseline produced no logprobs")
        self.assertTrue(lp_dcp2, "dcp_size=2 produced no logprobs")
        self.assertEqual(len(lp_base), len(lp_dcp2))

        base_t = torch.tensor(lp_base)
        dcp2_t = torch.tensor(lp_dcp2)
        max_abs_diff = (base_t - dcp2_t).abs().max().item()
        self.assertLess(max_abs_diff, ATOL, f"short prompt max abs diff={max_abs_diff}")

    def test_long_prompt_logprob_parity(self):
        """Longer context prompt: logprobs from dcp_size=2 must match baseline."""
        lp_base = _get_logprobs(self.url_baseline, PROMPTS[1])
        lp_dcp2 = _get_logprobs(self.url_dcp2, PROMPTS[1])

        self.assertTrue(lp_base, "baseline produced no logprobs")
        self.assertTrue(lp_dcp2, "dcp_size=2 produced no logprobs")
        self.assertEqual(len(lp_base), len(lp_dcp2))

        base_t = torch.tensor(lp_base)
        dcp2_t = torch.tensor(lp_dcp2)
        max_abs_diff = (base_t - dcp2_t).abs().max().item()
        self.assertLess(max_abs_diff, ATOL, f"long prompt max abs diff={max_abs_diff}")

    def test_dcp2_smoke_response(self):
        """Smoke test: dcp_size=2 server must respond coherently to a factual prompt."""
        resp = requests.post(
            self.url_dcp2 + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Reply with just the number.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 8,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        self.assertTrue(text, "dcp_size=2 returned empty response")
        self.assertIn(
            "4", text, f"dcp_size=2 smoke response did not contain '4': {text!r}"
        )


@unittest.skipUnless(_has_two_gpus(), "DSV4 DCP smoke test requires ≥2 GPUs")
class TestDsv4DcpDecodeSmoke(CustomTestCase):
    """Standalone DCP decode smoke test.

    Launches a single DSV4 server with ``--tp 2 --dcp-size 2`` and verifies
    a factual prompt returns a coherent answer.  Unlike the parity test,
    this does not compare against a baseline — it only checks that the DCP
    server produces a correct response.
    """

    @classmethod
    def setUpClass(cls):
        cls.port = 27119
        cls.url = f"http://127.0.0.1:{cls.port}"

        args = [
            "--tp-size",
            "2",
            "--dcp-size",
            "2",
            "--attention-backend",
            "deepseek_v4",
            "--trust-remote-code",
            "--random-seed",
            "0",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mem-fraction-static",
            "0.75",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dsv4-prefill-backend",
            "flashmla_sparse_q8",
            "--moe-runner-backend",
            "flashinfer_mxfp4",
        ]

        cls.proc = popen_launch_server(
            DSV4_MODEL,
            base_url=cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
        )

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "proc", None)
        if proc:
            kill_process_tree(proc.pid, wait_timeout=60)

    def test_paris_in_response(self):
        """Send 'What is the capital of France?' and assert 'Paris' in output."""
        resp = requests.post(
            self.url + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France?",
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        self.assertTrue(text, "DCP decode smoke returned empty response")
        self.assertIn(
            "Paris",
            text,
            f"DCP decode smoke response did not contain 'Paris': {text!r}",
        )


if __name__ == "__main__":
    unittest.main()
