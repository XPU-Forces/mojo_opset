"""
ILU Triton: T5-style relative position embedding gather (weight[bucket[lq, lk], :] -> [1, H, Lq, Lk]).
Bucket indices are computed in PyTorch to match MojoRelativeEmbedding._relative_position_bucket exactly.
"""

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.utils import input_guard
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype

from .utils import libentry


@libentry()
@triton.jit
def _relative_embedding_gather_kernel(
    bucket_ptr,
    w_ptr,
    out_ptr,
    LQ,
    LK,
    H,
    stride_b_lq,
    stride_b_lk,
    stride_w_nb,
    stride_w_h,
    H_PAD: tl.constexpr,
    OUT_T: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lq = pid // LK
    lk = pid % LK
    b = tl.load(bucket_ptr + lq * stride_b_lq + lk * stride_b_lk).to(tl.int32)
    offs_h = tl.arange(0, H_PAD)
    mask_h = offs_h < H
    w_ptrs = w_ptr + b * stride_w_nb + offs_h * stride_w_h
    vals = tl.load(w_ptrs, mask=mask_h, other=0.0)
    off_o = offs_h * (LQ * LK) + lq * LK + lk
    tl.store(out_ptr + off_o, vals.to(OUT_T), mask=mask_h)


@input_guard(make_contiguous=True, auto_to_device=True)
def relative_embedding_fwd_impl(bucket: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if bucket.dim() != 2:
        raise ValueError("bucket must be 2D [Lq, Lk]")
    if weight.dim() != 2:
        raise ValueError("weight must be 2D [num_buckets, num_heads]")
    Lq, Lk = int(bucket.shape[0]), int(bucket.shape[1])
    H = int(weight.shape[1])
    bucket = bucket.to(torch.int32)

    out = torch.empty(1, H, Lq, Lk, device=weight.device, dtype=weight.dtype)
    if Lq == 0 or Lk == 0:
        return out

    H_PAD = triton.next_power_of_2(H)
    OUT_T = torch_to_triton_dtype[weight.dtype]
    grid = (Lq * Lk,)
    _relative_embedding_gather_kernel[grid](
        bucket,
        weight,
        out,
        LQ=Lq,
        LK=Lk,
        H=H,
        stride_b_lq=bucket.stride(0),
        stride_b_lk=bucket.stride(1),
        stride_w_nb=weight.stride(0),
        stride_w_h=weight.stride(1),
        H_PAD=H_PAD,
        OUT_T=OUT_T,
    )
    return out
