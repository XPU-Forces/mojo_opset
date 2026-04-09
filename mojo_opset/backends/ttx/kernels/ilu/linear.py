"""
ILU Triton: F.linear(x, weight, bias) with weight (out_features, in_features), x (*, in_features).
Computes y = x @ weight.T (+ bias), same as torch.nn.functional.linear.
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
) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError("weight must be 2D (out_features, in_features)")
    n_out, n_in = int(weight.shape[0]), int(weight.shape[1])
    if x.dtype not in torch_to_triton_dtype or weight.dtype != x.dtype:
        raise TypeError("linear_fwd_impl expects matching float16, bfloat16, or float32 tensors")
    if bias is not None and bias.dtype != x.dtype:
        raise TypeError("bias dtype must match input")

    *batch, k_in = x.shape
    if k_in != n_in:
        raise ValueError(f"input last dim {k_in} != weight in_features {n_in}")
    x2 = x.reshape(-1, n_in)
    m = int(x2.shape[0])
    if m == 0:
        return x.new_empty(*batch, n_out, dtype=x.dtype, device=x.device)

    out = torch.empty(m, n_out, device=x.device, dtype=x.dtype)
    OUT_T = torch_to_triton_dtype[x.dtype]
    has_bias = bias is not None
    bm, bn, bk = _DEFAULT_BM, _DEFAULT_BN, _DEFAULT_BK
    grid = (triton.cdiv(m, bm) * triton.cdiv(n_out, bn),)
    _linear_fwd_kernel[grid](
        x2,
        weight,
        bias if has_bias else x2,
        out,
        m,
        n_out,
        n_in,
        x2.stride(0),
        x2.stride(1),
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
    return out.reshape(*batch, n_out)
