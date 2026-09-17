from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )


class CandidateMetadata:
    """Base of an implementation's published state on
    ``DSV4Metadata.candidate_metadata``."""


@dataclass(frozen=True)
class HopperSparseBlockTable(CandidateMetadata):
    blocks: torch.Tensor
    valid_lens: torch.Tensor
    topk_metadata: torch.Tensor


@dataclass(frozen=True)
class IndexerInputs:
    """One index-source layer's operands on the paged fp4 decode path (one query
    row per request, or per draft token under verify)."""

    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    metadata: PagedIndexerMetadata  # this ratio's lengths, page table and plans
    # [rows] int, one request id per query row, the rows of one request
    # consecutive (verify: its draft tokens); None = every row its own request
    request_ids: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]


def make_candidate_indexer(
    topk_blocks: int, block_size: int
) -> Optional[DeepGemmCandidateIndexer]:
    """The paged fp4 decode path's two-level indexer; None on Hopper, whose decode
    indexer selects through masks inline."""
    if topk_blocks <= 0 or get_platform().device_sm < 100:
        return None
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        raise RuntimeError(
            "the candidate indexer needs DeepGEMM's paged sparse MQA logits "
            "(sgl-deep-gemm >= 0.2.0 with SGLANG_ENABLE_JIT_DEEPGEMM on)"
        )
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )

    return DeepGemmCandidateIndexer(topk_blocks, block_size)


# TODO(candidate): Hopper decode and prefill still select through these masks
# inline in the backend; move them behind the protocol as publish/select_prefill.
@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = (
        None  # prefill: [rows_b, lc_b] each (bool)
    )
    request_blocks: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = (
        None  # prefill compact: each (blocks[rows_b, topk_blocks] int32, valid[rows_b, topk_blocks] bool)
    )


def published_masks(candidate) -> CandidateMasks:
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


def select_candidate_block_indices(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected block indices and their reachability, including the newest block."""
    if (
        logits.is_cuda
        and torch.version.cuda is not None
        and logits.ndim == 2
        and logits.stride(1) == 1
        and torch.is_tensor(compress_lens)
        and compress_lens.device == logits.device
        and compress_lens.dtype in (torch.int32, torch.int64)
        and compress_lens.numel() == logits.shape[0]
        and logits.numel() > 0
        and 0 < block_size <= 1024
    ):
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_indices,
        )

        return candidate_block_indices(
            logits,
            compress_lens.reshape(-1).contiguous(),
            topk_blocks=topk_blocks,
            block_size=block_size,
        )

    width = logits.size(-1)
    padding = -width % block_size
    scores = F.pad(logits, (0, padding), value=-torch.inf) if padding else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    return top.indices, top.values > -torch.inf


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: a bool mask over positions keeping the
    topk_blocks best-scoring blocks per query. Unreachable positions are already -inf
    in logits, so an all -inf block means not reachable yet; the block holding the
    query's newest position is always kept."""
    indices, valid = select_candidate_block_indices(
        logits, compress_lens, topk_blocks, block_size
    )
    width = logits.size(-1)
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (*logits.shape[:-1], num_blocks), dtype=torch.bool, device=logits.device
    ).scatter_(
        -1,
        indices,
        valid,
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
