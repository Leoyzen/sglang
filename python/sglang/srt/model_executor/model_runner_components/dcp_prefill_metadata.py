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

"""Shared prefill DCP metadata builder.

Openspec change ``enable-dcp-bcg-prefill-cudagraph`` task 2.1 (design D1/D4).

The eager extend path (``EagerRunner._execute_extend``) was the only call
site that built ``forward_batch.attn_dcp_metadata`` through the model's
``prepare_context_parallel_metadata_for_dcp``. Capture-time dummy extends
(BCG / tc_piecewise prefill CUDA graphs) must go through the *same* path so
captured segments bind real DCP metadata instead of ``None`` (design D1),
and so the capture dummy satisfies the real-batch DCP invariants by
construction (design D4). This module is the single entry point for both.

The builder is deliberately thin: it reproduces the eager call site's
guards and argument wiring exactly and delegates to the model's own
``prepare_context_parallel_metadata_for_dcp`` (DeepSeek V2/V3, Kimi Linear
/ K3 / K25 today), which routes to
``sglang.srt.layers.dcp.planner.prepare_decode_context_parallel_metadata``.
Keeping the model wrapper in the chain preserves model-specific delegation
(e.g. VLM wrappers handing the call to the inner language model, the PR
#31514 fix) without this module knowing the model zoo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


def prepare_dcp_extend_metadata(
    model_runner: ModelRunner,
    forward_batch: ForwardBatch,
) -> Optional[DecodeContextParallelMetadata]:
    """Build prefill DCP extend metadata for ``forward_batch``.

    Mirrors the eager call site exactly: no-op (returns ``None`` and leaves
    ``forward_batch.attn_dcp_metadata`` untouched) unless decode context
    parallelism is active (``attn_dcp_size > 1``) and the model implements
    ``prepare_context_parallel_metadata_for_dcp``. On success the metadata
    is assigned to ``forward_batch.attn_dcp_metadata`` and also returned so
    capture paths can bind it into persistent buffers.

    Requires an active forward context (``get_req_to_token_pool`` /
    ``get_token_to_kv_pool``), same as the eager extend path.
    """
    if model_runner.ps.attn_dcp_size <= 1:
        return None
    model = model_runner.model
    prepare_fn = getattr(model, "prepare_context_parallel_metadata_for_dcp", None)
    if prepare_fn is None:
        return None
    assert callable(prepare_fn), (
        "prepare_context_parallel_metadata_for_dcp must be callable"
    )

    from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
        create_chunked_prefix_cache_kv_indices,
    )
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )

    # prepare kv cache buffer for dcp to gather kv cache
    forward_batch.attn_dcp_metadata = prepare_fn(
        forward_batch.seq_lens,
        forward_batch.extend_prefix_lens,
        forward_batch.extend_prefix_lens_cpu,
        forward_batch.extend_seq_lens,
        forward_batch.req_pool_indices,
        get_req_to_token_pool().req_to_token,
        forward_batch.seq_lens_sum,
        get_token_to_kv_pool().get_kv_buffer_shape()[0],
        model_runner.kv_cache_dtype,
        model_runner.device,
        create_chunked_prefix_cache_kv_indices,
    )
    return forward_batch.attn_dcp_metadata
