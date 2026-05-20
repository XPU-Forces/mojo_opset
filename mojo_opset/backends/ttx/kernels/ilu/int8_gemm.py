# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Triton INT8 GEMM + fused dequantization kernel for Iluvatar GPUs.
#
# Computes:
#     output = (A_int8 @ B_int8) * input_scale * weight_scale [+ bias]
#
# with INT32 accumulation and fused epilogue (scale, bias, dtype cast).
# Avoids tl.dot for int8 (ILU compiler SharedToDotOperand layout conversion
# generates invalid bitcasts between mismatched widths, e.g. <2xf32>-><4xi8>,
# causing LLVM verifier errors then segfault in make_llir).
# Uses rank-1 K-loop accumulation in int32 instead.

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_gemm_dequant_kernel(
    a_ptr, bt_ptr, c_ptr,
    input_scale_ptr, weight_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btn, stride_btk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_idx in tl.range(0, K):
        a_col = tl.load(
            a_ptr + offs_m * stride_am + k_idx * stride_ak,
            mask=mask_m, other=0,
        ).to(tl.int32)
        b_row = tl.load(
            bt_ptr + offs_n * stride_btn + k_idx * stride_btk,
            mask=mask_n, other=0,
        ).to(tl.int32)
        acc += a_col[:, None] * b_row[None, :]

    # Fused epilogue: int32 -> fp32 -> scale -> [+bias] -> output_dtype
    c_fp32 = acc.to(tl.float32)
    in_scale = tl.load(input_scale_ptr + offs_m, mask=mask_m, other=0.0)
    wt_scale = tl.load(weight_scale_ptr + offs_n, mask=mask_n, other=0.0)
    c_fp32 = c_fp32 * in_scale[:, None] * wt_scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        c_fp32 = c_fp32 + bias[None, :]

    c = c_fp32.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def prepare_b_impl(b: torch.Tensor) -> torch.Tensor:
    """Transpose B to (N, K) row-major layout.

    For inference: weight B is fixed, call once and reuse.

    Args:
        b: (K, N) int8

    Returns:
        bt: (N, K) int8, contiguous
    """
    return b.T.contiguous()


_BLOCK_M = 32
_BLOCK_N = 32


def int8_gemm_dequant_impl(
    a: torch.Tensor,
    b_transposed: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    M_orig: int,
    N_orig: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused Triton INT8 GEMM + dequantization on Iluvatar GPU.

    Computes: output = (a @ b) * input_scale * weight_scale [+ bias]

    Args:
        a: (M, K) int8, contiguous
        b_transposed: (N, K) int8, from prepare_b_impl
        input_scale: (M,) float32, per-token scale
        weight_scale: (N,) float32, per-channel scale
        bias: (N,) output_dtype or None
        M_orig: original M (rows to keep in output)
        N_orig: original N (cols to keep in output)
        output_dtype: torch.float16 / torch.bfloat16 / torch.float32

    Returns:
        c: (M_orig, N_orig) in output_dtype
    """
    if not a.is_contiguous():
        a = a.contiguous()

    M, K = a.shape
    N, K_bt = b_transposed.shape

    has_bias = bias is not None
    bias_ptr = bias if has_bias else weight_scale

    c = torch.empty(M, N, device=a.device, dtype=output_dtype)

    grid = (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)

    _int8_gemm_dequant_kernel[grid](
        a, b_transposed, c,
        input_scale, weight_scale, bias_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b_transposed.stride(0), b_transposed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HAS_BIAS=has_bias,
    )

    return c[:M_orig, :N_orig]
