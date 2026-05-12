"""
ILU Triton: 2D GEMM kernel  y = x @ weight.T (+ bias).
weight is (N, K), x is (M, K), output is (M, N).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.utils import input_guard
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype

from .utils import libentry

_DEFAULT_BM = 64
_DEFAULT_BN = 64
_DEFAULT_BK = 32


@libentry()
@triton.jit
def _linear_fwd_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    OUT_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        wtile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        wtrans = tl.trans(wtile)
        acc = tl.dot(a, wtrans, acc=acc)

    if HAS_BIAS:
        bvec = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bvec[None, :]

    c = acc.to(OUT_T)
    c_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


@input_guard(make_contiguous=True, auto_to_device=True)
def linear_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    M: Optional[int] = None,
    N: Optional[int] = None,
    K: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """2D GEMM: out(M,N) = x(M,K) @ weight(N,K).T [+ bias(N,)].

    x must be 2D. Callers handle any batch-dim reshape.

    Args:
        M, N, K: Pre-computed dimensions. When provided, skips Host-side
            int() conversions on tensor shapes.
        out: Pre-allocated (M, N) output buffer. When provided, skips
            internal torch.empty allocation.
    """
    if M is not None:
        m, n_out, n_in = M, N, K
    else:
        if weight.dim() != 2 or x.dim() != 2:
            raise ValueError("x and weight must both be 2D")
        n_out, n_in = int(weight.shape[0]), int(weight.shape[1])
        m = int(x.shape[0])

    if x.dtype not in torch_to_triton_dtype or weight.dtype != x.dtype:
        raise TypeError("linear_fwd_impl expects matching float16, bfloat16, or float32 tensors")
    if bias is not None and bias.dtype != x.dtype:
        raise TypeError("bias dtype must match input")

    if m == 0:
        if out is not None:
            return out
        return x.new_empty(0, n_out, dtype=x.dtype, device=x.device)

    if out is None:
        out = torch.empty(m, n_out, device=x.device, dtype=x.dtype)
    OUT_T = torch_to_triton_dtype[x.dtype]
    has_bias = bias is not None
    bm, bn, bk = _DEFAULT_BM, _DEFAULT_BN, _DEFAULT_BK
    grid = (triton.cdiv(m, bm) * triton.cdiv(n_out, bn),)
    _linear_fwd_kernel[grid](
        x,
        weight,
        bias if has_bias else x,
        out,
        m,
        n_out,
        n_in,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=has_bias,
        OUT_T=OUT_T,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
    )
    return out
