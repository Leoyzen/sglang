"""Convert MXFP8 weights to block-fp8 [128,128] for AMD gfx942 (CDNA3 / MI300).

gfx942 has no hardware MX-scaled matmul: Triton's ``tl.dot_scaled`` fails to
lower and the gfx950 ``mfma_scale`` intrinsics are unavailable. So MXFP8
checkpoints (e4m3fn weights + 1x32 UE8M0 scales) are converted at load time to
block-wise FP8 [128,128] (e4m3fn + fp32 scales), which runs through SGLang's
native DeepSeek-V3 block-fp8 kernels (aiter / triton). The conversion is:

    bf16 = e4m3.to(f32) * exp2(ue8m0_scale.to(f32) - 127.0)   # dequant 1x32
    block-fp8 = per-128x128-block quantize(bf16)              # requant 128x128
"""

from __future__ import annotations

from typing import Tuple

import torch

MXFP8_BLOCK_SIZE = 32


def _ue8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 uint8 (biased exponent, bias 127) -> fp32 multiplier 2^(v-127)."""
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def dequant_mxfp8_2d_to_bf16(
    weight: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:
    """Dequant a 2D MXFP8 tensor (e4m3fn + 1x32 UE8M0 scales) to bf16.

    weight: [N, K] float8_e4m3fn; scale_u8: [N, K//32] uint8.
    """
    n, k = weight.shape
    descale = _ue8m0_to_fp32(scale_u8).unsqueeze(-1)  # [N, K//32, 1]
    deq = weight.to(torch.float32).view(n, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (deq * descale).view(n, k).to(torch.bfloat16)


def bf16_to_block_fp8_128(
    weight: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D bf16/fp32 weight to block-wise FP8 (e4m3fn) + fp32 scales.

    Returns (qweight [N,K] e4m3fn, scale [ceil(N/block), ceil(K/block)] fp32).
    Mirrors the DeepSeek-V3 block-fp8 contract (divide by e4m3fn max 448).
    The downstream gfx942 path normalizes e4m3fn -> e4m3fnuz separately.
    """
    n, k = weight.shape
    pn = ((n + block - 1) // block) * block
    pk = ((k + block - 1) // block) * block
    xp = torch.zeros((pn, pk), dtype=torch.float32, device=weight.device)
    xp[:n, :k] = weight.to(torch.float32)
    xv = xp.view(pn // block, block, pk // block, block)
    amax = xv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    sf = amax / 448.0
    xq = (xv / sf).to(torch.float8_e4m3fn)
    qweight = xq.view(pn, pk)[:n, :k].contiguous()
    scale = sf.view(pn // block, pk // block).contiguous()
    return qweight, scale


def convert_mxfp8_weight_to_block_fp8(
    weight: torch.Tensor, scale_u8: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 (e4m3fn + 1x32 UE8M0) -> block-fp8 [block,block] (e4m3fn + fp32).

    Used on gfx942 to run MXFP8 checkpoints through the fast native block-fp8
    kernels.
    """
    bf16 = dequant_mxfp8_2d_to_bf16(weight, scale_u8)
    return bf16_to_block_fp8_128(bf16, block=block)


def dequant_block_fp8_2d_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor, block: int = 32
) -> torch.Tensor:
    """Dequant a DeepSeek-style 2D block-fp8 tensor to bf16 exactly.

    weight: [N, K] float8_e4m3fn; scale: [N//block, K//block] holding the
    per-block multipliers as fp32 (values may be ue8m0 bit patterns viewed as
    fp32 or plain fp32 block scales).
    """
    n, k = weight.shape
    descale = (
        scale.to(torch.float32)
        .repeat_interleave(block, dim=0)
        .repeat_interleave(block, dim=1)[:n, :k]
    )
    return (weight.to(torch.float32) * descale).to(torch.bfloat16)


def requant_block_fp8_32_to_128(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Requantize a [32,32]-block FP8 weight to [128,128] blocks at load time.

    Dequantization is exact (fp32 multiply), so the only new error is the
    single re-round to e4m3fn under 4x-coarser scales -- measured identical in
    distribution to a native [128,128] checkpoint (mean_rel and p99 match).

    weight: [N, K] float8_e4m3fn; scale: [N//32, K//32] fp32 (or ue8m0 viewed
    as fp32). Returns (qweight [N,K] e4m3fn, scale [N//128, K//128] fp32).
    Caller must guarantee N % 128 == 0 and K % 128 == 0.
    """
    bf16 = dequant_block_fp8_2d_to_bf16(weight, scale, block=32)
    return bf16_to_block_fp8_128(bf16, block=128)
