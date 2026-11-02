# SPDX-License-Identifier: Apache-2.0
"""PixelUMM split QK normalization and MRoPE with externally computed means."""

import torch
import triton
import triton.language as tl


@triton.jit
def _nomas_qk_norm_mrope(
    X,
    MT,
    MH,
    WT,
    WH,
    COS,
    SIN,
    OUT,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    XB: tl.constexpr,
    XT: tl.constexpr,
    XH: tl.constexpr,
    OB: tl.constexpr,
    OUT_TOKEN_STRIDE: tl.constexpr,
    OH: tl.constexpr,
    CT: tl.constexpr,
    ST: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    j = tl.arange(0, 128)
    batch = rows // (TOKENS * HEADS)
    token = (rows // HEADS) % TOKENS
    head = rows % HEADS
    offset = batch * XB + token * XT + head * XH
    x = tl.load(X + offset[:, None] + j[None, :], rows[:, None] < ROWS, 0).to(
        tl.float32
    )
    mt = tl.load(MT + rows, rows < ROWS, 0.0)
    mh = tl.load(MH + rows, rows < ROWS, 0.0)
    ms = tl.where(j[None, :] < 64, mt[:, None], mh[:, None])
    wt = tl.load(WT + j % 64).to(tl.float32)
    wh = tl.load(WH + j % 64).to(tl.float32)
    weight = tl.where(j < 64, wt, wh)
    normalized = (x * tl.rsqrt(ms + EPS)) * weight[None, :]
    # Rotate independently within the T/H/W sections (64, 32, 32).
    half = tl.where(j < 64, 32, 16)
    base = tl.where(j < 64, 0, tl.where(j < 96, 64, 96))
    local = j - base
    partner = base + (local + half) % (half * 2)
    rotated = tl.gather(normalized, tl.broadcast_to(partner[None, :], (R, 128)), 1)
    rotated = tl.where(local[None, :] < half[None, :], -rotated, rotated)
    cos = tl.load(COS + token[:, None] * CT + j[None, :], rows[:, None] < ROWS, 0.0)
    sin = tl.load(SIN + token[:, None] * ST + j[None, :], rows[:, None] < ROWS, 0.0)
    offset = batch * OB + token * OUT_TOKEN_STRIDE + head * OH
    tl.store(
        OUT + offset[:, None] + j[None, :],
        normalized * cos + rotated * sin,
        rows[:, None] < ROWS,
    )


def nomas_qk_norm_mrope(
    x, mean_t, mean_hw, weight_t, weight_hw, cos, sin, eps, out=None
):
    """Normalize BF16 ``[B,T,H,128]`` Q/K and apply FP32 (64,32,32) MRoPE.

    The caller supplies contiguous FP32 ``[B,T,H,1]`` ATen mean squares to
    preserve its reduction order. All tensors must share a CUDA device. RoPE
    tables have shape ``[1,T,1,128]`` and are shared across batches and heads.
    Input/output batch, token and head strides may differ, allowing bounded
    token chunks to write directly into the final output without a slice copy.
    Separate FP32 multiply/add operations and the final BF16 cast are retained.
    """
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("nomas_qk_norm_mrope requires CUDA")
    if (
        x.ndim != 4
        or x.shape[-1] != 128
        or x.dtype != torch.bfloat16
        or x.stride(-1) != 1
    ):
        raise ValueError("expected BF16 [B,T,H,128] with contiguous head dimension")
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if out.shape != x.shape or out.dtype != x.dtype or out.stride(-1) != 1:
        raise ValueError("out must match x with contiguous head dimension")
    for mean in (mean_t, mean_hw):
        if (
            mean.shape != (*x.shape[:3], 1)
            or mean.dtype != torch.float32
            or not mean.is_contiguous()
        ):
            raise ValueError("means must be contiguous FP32 [B,T,H,1]")
    for weight in (weight_t, weight_hw):
        if (
            weight.shape != (64,)
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            raise ValueError("weights must be contiguous BF16 [64]")
    for table in (cos, sin):
        if (
            table.shape != (1, x.shape[1], 1, 128)
            or table.dtype != torch.float32
            or table.stride(-1) != 1
        ):
            raise ValueError(
                "RoPE tables must be FP32 [1,T,1,128] with contiguous last dimension"
            )
    if any(
        t.device != x.device
        for t in (out, mean_t, mean_hw, weight_t, weight_hw, cos, sin)
    ):
        raise ValueError("all tensors must share a CUDA device")
    rows = x.numel() // 128
    if rows:
        _nomas_qk_norm_mrope[(triton.cdiv(rows, 4),)](
            x,
            mean_t,
            mean_hw,
            weight_t,
            weight_hw,
            cos,
            sin,
            out,
            rows,
            x.shape[1],
            x.shape[2],
            *x.stride()[:3],
            *out.stride()[:3],
            cos.stride(1),
            sin.stride(1),
            eps,
            4,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
