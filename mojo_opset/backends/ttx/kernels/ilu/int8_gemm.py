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
# Uses BLOCK_K-tiled rank-1 accumulation in int32 instead.

import torch
import triton
import triton.language as tl

from .utils import smart_triton_autotune


@smart_triton_autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=16, num_stages=1),
    ],
    selected_idx=0,
    key=["M", "N", "K"],
)
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def _int8_gemm_dequant_kernel(
    A,
    B,
    C,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bs0,
    stride_bs1,
    stride_cm,
    stride_cn,
    TRANS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
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

    for kk in tl.range(0, tl.cdiv(K, BLOCK_K)):
        rk = kk * BLOCK_K + tl.arange(0, BLOCK_K)

        A_offset = A + offs_m[:, None] * stride_am + rk[None, :] * stride_ak
        if TRANS_WEIGHT:
            B_offset = B + offs_n[None, :] * stride_bs0 + rk[:, None] * stride_bs1
        else:
            B_offset = B + rk[:, None] * stride_bs0 + offs_n[None, :] * stride_bs1

        if EVEN_K:
            a_tile = tl.load(A_offset)
            b_tile = tl.load(B_offset)

            acc += tl.dot(a_tile, b_tile, out_dtype=tl.int32, input_precision="ieee")
        else:
            mask_k = rk < K
            a_tile = tl.load(
                A_offset,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0,
            )
            b_tile = tl.load(
                B_offset,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )

            acc += tl.dot(a_tile, b_tile, out_dtype=tl.int32, input_precision="ieee")

    c_fp32 = acc.to(tl.float32)
    in_scale = tl.load(input_scale_ptr + offs_m, mask=mask_m, other=0.0)
    wt_scale = tl.load(weight_scale_ptr + offs_n, mask=mask_n, other=0.0)
    c_fp32 = c_fp32 * in_scale[:, None] * wt_scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        c_fp32 = c_fp32 + bias[None, :]

    c = c_fp32.to(C.dtype.element_ty)
    Cs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(Cs, c, mask=mask_m[:, None] & mask_n[None, :])


def prepare_b_impl(b: torch.Tensor) -> torch.Tensor:
    """Transpose B to (N, K) row-major layout.

    For inference: weight B is fixed, call once and reuse.

    Args:
        b: (K, N) int8

    Returns:
        bt: (N, K) int8, contiguous
    """
    return b.T.contiguous()


def int8_gemm_dequant_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    M_orig: int,
    N_orig: int,
    output_dtype: torch.dtype,
    trans_weight: bool = False,
) -> torch.Tensor:
    """Fused Triton INT8 GEMM + dequantization on Iluvatar GPU.

    Computes: output = (a @ b) * input_scale * weight_scale [+ bias]

    Padding is not supported: `M` must equal `M_orig` and `N` must equal `N_orig`.

    Args:
        a: (M, K) int8
        b: int8 weight tensor. Layout depends on ``trans_weight``:
            - ``trans_weight=True``: ``(N, K)`` (already transposed; kernel-native layout)
            - ``trans_weight=False``: ``(K, N)`` (will be transposed to ``(N, K)`` internally)
        input_scale: (M,) float32, per-token scale
        weight_scale: (N,) float32, per-channel scale
        bias: (N,) output_dtype or None
        M_orig: must equal M
        N_orig: must equal N
        output_dtype: torch.float16 / torch.bfloat16 / torch.float32
        trans_weight: layout flag for ``b`` (see above). Defaults to False.

    Returns:
        c: (M, N) in output_dtype
    """
    if not a.is_contiguous():
        raise ValueError("int8_gemm_dequant_impl: `a` must be contiguous")
    if not b.is_contiguous():
        raise ValueError("int8_gemm_dequant_impl: `b` must be contiguous")

    M, K = a.shape
    if trans_weight:
        N, K_b = b.shape
    else:
        K_b, N = b.shape

    if K != K_b:
        raise ValueError(
            f"int8_gemm_dequant_impl: K dim mismatch, a has K={K} but b has K={K_b} "
            f"(trans_weight={trans_weight})"
        )

    if M != M_orig:
        raise ValueError(
            f"int8_gemm_dequant_impl: padding is not supported, M ({M}) must equal M_orig ({M_orig})"
        )
    if N != N_orig:
        raise ValueError(
            f"int8_gemm_dequant_impl: padding is not supported, N ({N}) must equal N_orig ({N_orig})"
        )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else weight_scale

    c = torch.empty(M, N, device=a.device, dtype=output_dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _int8_gemm_dequant_kernel[grid](
        a, b, c,
        input_scale, weight_scale, bias_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        TRANS_WEIGHT=trans_weight,
        HAS_BIAS=has_bias,
    )

    return c
