"""ILU Triton: INT8 GEMM + fused dequantization.

Computes:
    output = (A_int8 @ B_int8) * input_scale * weight_scale [+ bias]

with INT32 accumulation and fused epilogue (scale, optional bias, dtype cast).
Avoids tl.dot for int8 on ILU; that path can segfault in the compiler.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import libentry

_DEFAULT_BM = 32
_DEFAULT_BN = 32


@libentry()
@triton.jit
def _int8_gemm_dequant_kernel(
    a_ptr,
    bt_ptr,
    c_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_btn,
    stride_btk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    OUT_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
            mask=mask_m,
            other=0,
        ).to(tl.int32)
        b_row = tl.load(
            bt_ptr + offs_n * stride_btn + k_idx * stride_btk,
            mask=mask_n,
            other=0,
        ).to(tl.int32)
        acc += a_col[:, None] * b_row[None, :]

    c_fp32 = acc.to(tl.float32)
    in_scale = tl.load(input_scale_ptr + offs_m, mask=mask_m, other=0.0)
    wt_scale = tl.load(weight_scale_ptr + offs_n, mask=mask_n, other=0.0)
    c_fp32 = c_fp32 * in_scale[:, None] * wt_scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        c_fp32 = c_fp32 + bias[None, :]

    c = c_fp32.to(OUT_T)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


_OUTPUT_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def prepare_b_impl(b: torch.Tensor) -> torch.Tensor:
    """Transpose weight B from (K, N) to (N, K) row-major.

    For inference the weight is fixed; call once and reuse.
    """
    return b.T.contiguous()


def int8_gemm_dequant_impl(
    a: torch.Tensor,
    b_transposed: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    M: int,
    N: int,
    output_dtype: torch.dtype,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused INT8 GEMM + dequantization for ILU.

    Computes: output = (a_i8 @ b_i8) * input_scale * weight_scale [+ bias]

    Args:
        a: (M, K) int8, contiguous.
        b_transposed: (N, K) int8, from prepare_b_impl.
        input_scale: (M,) float32, per-token scale.
        weight_scale: (N,) float32, per-channel scale.
        bias: (N,) or None.
        M: number of rows (original, before any external padding).
        N: number of columns (original).
        output_dtype: torch.float16 / torch.bfloat16 / torch.float32.
        out: optional pre-allocated (M, N) output buffer.
    """
    K = a.shape[1]

    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=output_dtype)

    OUT_T = _OUTPUT_DTYPE_MAP[output_dtype]
    has_bias = bias is not None
    bm, bn = _DEFAULT_BM, _DEFAULT_BN
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

    _int8_gemm_dequant_kernel[grid](
        a,
        b_transposed,
        out,
        input_scale,
        weight_scale,
        bias if has_bias else input_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_transposed.stride(0),
        b_transposed.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=has_bias,
        OUT_T=OUT_T,
        BLOCK_M=bm,
        BLOCK_N=bn,
    )
    return out
