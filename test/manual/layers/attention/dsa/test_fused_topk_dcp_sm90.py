"""Compare fused-v2 and unfused DSA top-k slot transforms under DCP.

Run on one SM90 GPU; DCP ranks are emulated after the common global-slot
transform because the indexer and its page table are replicated group-wide.

    CUDA_VISIBLE_DEVICES=0 python3 test/manual/layers/attention/dsa/test_fused_topk_dcp_sm90.py
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_512_v2,
)

PAGE_SIZE = 64
TOPK = 2048


def _make_dcp_page_tables(
    batch_size: int, seq_len: int, dcp_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the layout produced by allocator pages of PAGE_SIZE * dcp_size."""
    superpage_size = PAGE_SIZE * dcp_size
    num_superpages = (seq_len + superpage_size - 1) // superpage_size
    rows = []
    for batch_id in range(batch_size):
        generator = torch.Generator(device=device).manual_seed(
            1009 * dcp_size + batch_id
        )
        # Give each request a disjoint, permuted global-slot region.
        superpages = (
            torch.randperm(num_superpages, generator=generator, device=device)
            + batch_id * num_superpages
        )
        offsets = torch.arange(superpage_size, device=device)
        slots = (superpages[:, None] * superpage_size + offsets[None, :]).flatten()
        rows.append(slots[:seq_len])

    page_table_1 = torch.stack(rows).to(torch.int32)
    real_page_table = page_table_1[:, ::PAGE_SIZE] // PAGE_SIZE
    return page_table_1, real_page_table


@torch.inference_mode()
def main() -> None:
    assert torch.cuda.get_device_capability()[0] == 9
    device = "cuda"
    torch.manual_seed(20260731)

    for dcp_size in (2, 4, 8):
        for batch_size, seq_len in ((4, 4096), (4, 16385), (2, 65537)):
            width = (seq_len + 3) & ~3
            scores = torch.randn(batch_size, width, dtype=torch.float32, device=device)[
                :, :seq_len
            ]
            seq_lens = torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device=device
            )
            page_table_1, real_page_table = _make_dcp_page_tables(
                batch_size, seq_len, dcp_size, device
            )

            raw = torch.topk(scores, TOPK, dim=-1, sorted=False).indices
            unfused_global = torch.gather(page_table_1, 1, raw).to(torch.int32)

            fused_global = torch.full(
                (batch_size, TOPK), -1, dtype=torch.int32, device=device
            )
            plan = plan_topk_v2(seq_lens)
            topk_transform_512_v2(
                scores,
                seq_lens,
                real_page_table,
                fused_global,
                PAGE_SIZE,
                plan,
            )
            torch.cuda.synchronize()

            for dcp_rank in range(dcp_size):
                fused_local = torch.where(
                    fused_global % dcp_size == dcp_rank,
                    fused_global // dcp_size,
                    -1,
                )
                unfused_local = torch.where(
                    unfused_global % dcp_size == dcp_rank,
                    unfused_global // dcp_size,
                    -1,
                )

                # Order is irrelevant to sparse attention. Random fp32 scores
                # make boundary ties negligibly likely, so compare slot sets.
                for row in range(batch_size):
                    fused_set = set(fused_local[row][fused_local[row] >= 0].tolist())
                    unfused_set = set(
                        unfused_local[row][unfused_local[row] >= 0].tolist()
                    )
                    assert fused_set == unfused_set, (
                        f"mismatch: {dcp_size=} {dcp_rank=} {batch_size=} "
                        f"{seq_len=} row={row} "
                        f"only_fused={list(fused_set - unfused_set)[:8]} "
                        f"only_unfused={list(unfused_set - fused_set)[:8]}"
                    )

            print(f"passed: dcp={dcp_size} batch={batch_size} seq={seq_len}")


if __name__ == "__main__":
    main()
