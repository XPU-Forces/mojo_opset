# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Triton: batched int8 matmul + scales + sum over batch -> (M,N) bf16.
#
# Avoids tl.dot (ILU compiler segfault). Uses rank-1 K updates. One launch per batch;
# each launch uses 2D views so the kernel has no batch-stride arithmetic.

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_one_batch_accum_kernel(
    x_ptr,
    w_ptr,
    s1_ptr,
    s2_ptr,
    out_acc_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_s1m,
    stride_s2,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """x: (M,K), w: (K,N), s1: (M,), s2: (N,), accumulate scaled (x@w) into out_acc."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in tl.range(0, K):
        a_col = tl.load(
            x_ptr + offs_m * stride_xm + k_idx * stride_xk,
            mask=mask_m,
            other=0,
        ).to(tl.float32)
        b_row = tl.load(
            w_ptr + k_idx * stride_wk + offs_n * stride_wn,
            mask=mask_n,
            other=0,
        ).to(tl.float32)
        tile += a_col[:, None] * b_row[None, :]

    s1 = tl.load(s1_ptr + offs_m * stride_s1m, mask=mask_m, other=0.0).to(tl.float32)
    s2 = tl.load(s2_ptr + offs_n * stride_s2, mask=mask_n, other=0.0).to(tl.float32)
    contrib = tile * s1[:, None] * s2[None, :]

    c_ptrs = out_acc_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = mask_m[:, None] & mask_n[None, :]
    prev = tl.load(c_ptrs, mask=mask, other=0.0)
    tl.store(c_ptrs, prev + contrib, mask=mask)


def quant_group_linear_reduce_sum_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
) -> torch.Tensor:
    """
    x: (B, M, K) int8, contiguous.
    w: (B, K, N) int8, contiguous.
    x1_scale: (B, M) float32
    x2_scale: (N,) bfloat16 (or convertible)
    Returns: (M, N) bfloat16
    """
    if x.dtype != torch.int8 or w.dtype != torch.int8:
        raise TypeError("quant_group_linear_reduce_sum_impl expects int8 x and w")
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    x1_scale = x1_scale.contiguous()
    if x2_scale.dtype != torch.bfloat16:
        x2_scale = x2_scale.to(torch.bfloat16).contiguous()
    else:
        x2_scale = x2_scale.contiguous()

    b, m, k = x.shape
    b2, k2, n = w.shape
    if b != b2 or k != k2:
        raise ValueError("x and w batch/K dimensions must match")

    # Match MojoQuantGroupLinearReduceSum reference: per-batch fp32 tile, cast to bf16,
    # then in-place bf16 accumulation (not fp32 sum then one cast).
    reduced_out = torch.zeros((m, n), device=x.device, dtype=torch.bfloat16)
    batch_contrib = torch.zeros((m, n), device=x.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)

    for batch_idx in range(b):
        batch_contrib.zero_()
        x_b = x[batch_idx]
        w_b = w[batch_idx]
        s1_b = x1_scale[batch_idx]
        _quant_one_batch_accum_kernel[grid](
            x_b,
            w_b,
            s1_b,
            x2_scale,
            batch_contrib,
            m,
            n,
            k,
            x_b.stride(0),
            x_b.stride(1),
            w_b.stride(0),
            w_b.stride(1),
            s1_b.stride(0),
            x2_scale.stride(0),
            batch_contrib.stride(0),
            batch_contrib.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        reduced_out += batch_contrib.to(torch.bfloat16)

    return reduced_out
