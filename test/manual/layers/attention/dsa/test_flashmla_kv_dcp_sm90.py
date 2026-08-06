"""Four-GPU FlashMLA-KV DCP correctness check for SM90.

Run with:
    torchrun --master-addr 127.0.0.1 --master-port 29591 --nproc-per-node 4 \
      test/manual/layers/attention/dsa/test_flashmla_kv_dcp_sm90.py

This intentionally avoids model weights. It exercises the production DCP owner
rule, FP8 KV layout, FlashMLA's -1 sparse-index holes, natural-log LSE
normalization, and the cross-rank online-softmax merge.
"""

import os

import torch
import torch.distributed as dist
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache

LOG2_E = 1.4426950408889634
SEQ_LEN = 4096
TOPK = 2048
NUM_HEADS = 64
HEAD_DIM = 576
VALUE_DIM = 512
PAGE_SIZE = 64


def _flashmla_sparse_decode(
    q: torch.Tensor,
    kv_bf16: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = kv_bf16.shape[0]
    padded_tokens = ((num_tokens + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    kv_padded = torch.zeros(
        padded_tokens, HEAD_DIM, dtype=torch.bfloat16, device=q.device
    )
    kv_padded[:num_tokens] = kv_bf16
    kv_fp8 = quantize_k_cache(kv_padded.view(-1, PAGE_SIZE, 1, HEAD_DIM))

    cache_seqlens = torch.tensor([TOPK], dtype=torch.int32, device=q.device)
    scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=NUM_HEADS,
        num_heads_k=1,
        num_heads_q=NUM_HEADS,
        is_fp8_kvcache=True,
        topk=TOPK,
    )
    out, natural_lse = flash_mla_with_kvcache(
        q=q.view(1, 1, NUM_HEADS, HEAD_DIM),
        k_cache=kv_fp8,
        block_table=torch.empty((1, 0), dtype=torch.int32, device=q.device),
        cache_seqlens=cache_seqlens,
        head_dim_v=VALUE_DIM,
        tile_scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=HEAD_DIM**-0.5,
        is_fp8_kvcache=True,
        indices=indices.view(1, 1, TOPK),
    )
    return out.view(NUM_HEADS, VALUE_DIM), natural_lse.view(NUM_HEADS)


@torch.inference_mode()
def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(f"This test requires four ranks, got {world_size}.")

    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise ValueError("flashmla_kv DCP integration test requires SM90.")

    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(20260731)
        q = torch.randn(
            NUM_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-1, 1)
        global_kv = (
            torch.randn(
                SEQ_LEN,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            / 10
        ).clamp_(-1, 1)
        topk_indices = torch.randperm(
            SEQ_LEN, dtype=torch.int32, device=device, generator=generator
        )[:TOPK]
    else:
        q = torch.empty(NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        global_kv = torch.empty(SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device=device)
        topk_indices = torch.empty(TOPK, dtype=torch.int32, device=device)

    dist.broadcast(q, src=0)
    dist.broadcast(global_kv, src=0)
    dist.broadcast(topk_indices, src=0)

    # Production owner rule: global slot g belongs to rank g % DCP, at local
    # row g // DCP. Unowned TopK entries remain -1 holes for FlashMLA.
    local_kv = global_kv[rank::world_size].contiguous()
    owned = topk_indices.remainder(world_size) == rank
    local_indices = torch.where(
        owned, topk_indices // world_size, torch.full_like(topk_indices, -1)
    )
    local_out, local_natural_lse = _flashmla_sparse_decode(q, local_kv, local_indices)

    # Match the SGLang patch: normalize FlashMLA's natural-log LSE to base-2,
    # then perform the same online-softmax correction across DCP ranks.
    local_lse = local_natural_lse.float() * LOG2_E
    gathered_lse = [torch.empty_like(local_lse) for _ in range(world_size)]
    dist.all_gather(gathered_lse, local_lse)
    lse_stack = torch.stack(gathered_lse)
    ln_2 = torch.log(torch.tensor(2.0, device=device))
    global_lse = torch.logsumexp(lse_stack * ln_2, dim=0) * LOG2_E
    local_weight = torch.exp2(local_lse - global_lse).unsqueeze(-1)
    merged_out = local_out.float() * local_weight
    dist.all_reduce(merged_out)

    if rank == 0:
        reference_out, reference_natural_lse = _flashmla_sparse_decode(
            q, global_kv, topk_indices
        )
        torch.testing.assert_close(
            merged_out,
            reference_out.float(),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            global_lse,
            reference_natural_lse.float() * LOG2_E,
            atol=2e-3,
            rtol=2e-3,
        )
        max_out_error = (merged_out - reference_out.float()).abs().max().item()
        max_lse_error = (
            (global_lse - reference_natural_lse.float() * LOG2_E).abs().max().item()
        )
        print(
            "FlashMLA KV DCP SM90 passed: "
            f"max_out_error={max_out_error:.6f}, "
            f"max_lse_error={max_lse_error:.6f}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
