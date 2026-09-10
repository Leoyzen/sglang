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

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


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


class PrefillDcpBuffers:
    """Persistent static buffers for the DCP intermediates captured prefill
    segments read (opsx 2.3, design D1).

    The captured segments address the ``DecodeContextParallelMetadata``
    tensors by data pointer, so the storage must be allocated ONCE, before
    capture, and refreshed in place per replay — never reallocated (spec:
    'No in-graph allocation of DCP intermediates'). This mirrors how decode
    graph replay keeps ``dcp_kv_mask`` live in its static input buffers
    (#18167).

    Which tensors the captured segments read, identified from the DCP
    consumers active during a prefill (extend) forward:

    - ``dcp_kv_indptr`` / ``dcp_kv_indices``: consumed by paged prefill
      attention planning (flashinfer_mla_backend
      ``forward_extend`` reads ``attn_dcp_metadata.dcp_kv_indptr`` /
      ``.dcp_kv_indices`` for the wrapper plan).
    - ``dcp_kv_buffer``: the cross-rank gathered KV store read as ``k_buf``
      by paged prefill attention (flashinfer_mla_backend) and written by
      ``all_gather_kv_cache_for_mla_extend`` / ``for_mha_extend`` in the
      model's attention prepare (may sit inside pre-attention captured
      segments).
    - ``dcp_local_prefix_kv_indices``: the rank-local ownership indices fed
      to ``get_mla_kv_buffer`` by the same gather helpers — i.e. the KV
      ownership read of pre-attention segments (#33253 class).
    - ``dcp_extend_prefix_lens_sum`` (host int): bounds the prefix portion
      of ``dcp_kv_buffer``.

    ``dcp_kv_mask`` (HIP decode write mask) is carried by the batch and
    already refreshed per replay by the scheduler path; it is not
    prefill-graph-owned here.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        max_bs: int,
        max_num_tokens: int,
        kv_cache_dim: int,
        kv_cache_dtype: torch.dtype,
    ) -> None:
        self.device = torch.device(device)
        self.max_bs = max_bs
        self.max_num_tokens = max_num_tokens
        self.kv_cache_dim = kv_cache_dim
        self.kv_cache_dtype = kv_cache_dtype
        with torch.device(self.device):
            # indptr: bs+1 entries; indices: flat seq_lens_sum entries
            # (bounded by max_num_tokens); prefix indices: also bounded by
            # max_num_tokens (prefix ⊆ seq_len ⊆ max_num_tokens per bucket;
            # an aggregate bucket's seq_lens_sum == its token count).
            self.dcp_kv_indptr = torch.zeros((max_bs + 1,), dtype=torch.int32)
            self.dcp_kv_indices = torch.zeros((max_num_tokens,), dtype=torch.int32)
            self.dcp_local_prefix_kv_indices = torch.zeros(
                (max_num_tokens,), dtype=torch.int32
            )
            self.dcp_kv_buffer = torch.zeros(
                (max_num_tokens, 1, kv_cache_dim), dtype=kv_cache_dtype
            )
        # Host-side scalar refreshed with the buffers.
        self.dcp_extend_prefix_lens_sum = 0

    def bind(self) -> DecodeContextParallelMetadata:
        """Build a metadata object whose tensors ARE views of this storage.

        The views keep stable data pointers across capture and replay; the
        planner/refresh path writes into the slices, never reassigns them.
        """
        return DecodeContextParallelMetadata(
            dcp_kv_indptr=self.dcp_kv_indptr,
            dcp_kv_buffer=self.dcp_kv_buffer,
            dcp_kv_indices=self.dcp_kv_indices,
            dcp_local_prefix_kv_indices=self.dcp_local_prefix_kv_indices,
            dcp_extend_prefix_lens_sum=self.dcp_extend_prefix_lens_sum,
        )

    def refresh_from(
        self, metadata: DecodeContextParallelMetadata, *, bs: int
    ) -> DecodeContextParallelMetadata:
        """Copy freshly computed values into the static views, in place.

        ``metadata`` is the planner output for the current batch; every
        field is written with copy_/assignment into the persistent slice —
        no reallocation. Returns a metadata object exposing the refreshed
        static views (the object the captured segments already address).
        """
        self.dcp_kv_indptr.zero_()
        assert metadata.dcp_kv_indptr is not None
        width = min(int(metadata.dcp_kv_indptr.shape[0]), self.max_bs + 1)
        self.dcp_kv_indptr[:width].copy_(metadata.dcp_kv_indptr[:width])

        self.dcp_kv_indices.zero_()
        assert metadata.dcp_kv_indices is not None
        width = min(int(metadata.dcp_kv_indices.shape[0]), self.max_num_tokens)
        self.dcp_kv_indices[:width].copy_(metadata.dcp_kv_indices[:width])

        self.dcp_local_prefix_kv_indices.zero_()
        if metadata.dcp_local_prefix_kv_indices is not None:
            width = min(
                int(metadata.dcp_local_prefix_kv_indices.shape[0]),
                self.max_num_tokens,
            )
            self.dcp_local_prefix_kv_indices[:width].copy_(
                metadata.dcp_local_prefix_kv_indices[:width]
            )

        self.dcp_kv_buffer.zero_()
        assert metadata.dcp_kv_buffer is not None
        rows = min(int(metadata.dcp_kv_buffer.shape[0]), self.max_num_tokens)
        self.dcp_kv_buffer[:rows].copy_(metadata.dcp_kv_buffer[:rows])

        self.dcp_extend_prefix_lens_sum = int(metadata.dcp_extend_prefix_lens_sum or 0)
        return self.bind()
