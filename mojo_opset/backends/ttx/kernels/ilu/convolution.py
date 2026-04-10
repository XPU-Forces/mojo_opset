"""
ILU Triton: causal conv1d state update (same math as MojoCausalConv1dUpdateState / F.conv1d on cat(state, x)).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.utils import input_guard

from .utils import libentry


@libentry()
@triton.jit
def _causal_conv1d_update_ilu_kernel(
    x_ptr,
    cs_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_x_b,
    stride_x_d,
    stride_x_t,
    stride_cs_b,
    stride_cs_d,
    stride_cs_s,
    stride_w_d,
    stride_w_w,
    stride_bias,
    stride_o_b,
    stride_o_d,
    stride_o_t,
    B: tl.constexpr,
    D: tl.constexpr,
    ST: tl.constexpr,
    T: tl.constexpr,
    W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    OUT_T: tl.constexpr,
):
    """Virtual concat [cs | x]; output time t uses window starting at j_start = ST - W + 1 + t."""
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)
    total = B * D * T

    for flat in tl.range(pid, total, pnum):
        b = flat // (D * T)
        rem = flat % (D * T)
        d = rem // T
        t = rem % T

        j_start = ST - W + 1 + t
        acc = tl.full((), 0.0, tl.float32)

        for kw in range(W):
            gi = j_start + kw
            off_cs = b * stride_cs_b + d * stride_cs_d + gi * stride_cs_s
            off_x = b * stride_x_b + d * stride_x_d + (gi - ST) * stride_x_t
            mask_cs = gi < ST
            mask_x = gi >= ST
            v_cs = tl.load(cs_ptr + off_cs, mask=mask_cs, other=0.0).to(tl.float32)
            v_x = tl.load(x_ptr + off_x, mask=mask_x, other=0.0).to(tl.float32)
            v = v_cs + v_x
            wi = tl.load(w_ptr + d * stride_w_d + kw * stride_w_w).to(tl.float32)
            acc = acc + v * wi

        if HAS_BIAS:
            acc = acc + tl.load(b_ptr + d * stride_bias).to(tl.float32)

        if SILU:
            acc = acc * tl.sigmoid(acc)

        off_o = b * stride_o_b + d * stride_o_d + t * stride_o_t
        tl.store(out_ptr + off_o, acc.to(OUT_T))


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_update_bdt_impl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
):
    del conv_state_indices

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    _, width = weight.shape
    out = torch.empty_like(x)

    total_tasks = batch * dim * seqlen
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_tasks, BLOCK_SIZE),)

    bias_arg = bias if bias is not None else torch.zeros(1, device=x.device, dtype=torch.float32)
    stride_bias = bias.stride(0) if bias is not None else 0

    if x.dtype == torch.float16:
        out_t = tl.float16
    elif x.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    _causal_conv1d_update_ilu_kernel[grid](
        x,
        conv_state,
        weight,
        bias_arg,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        weight.stride(1),
        stride_bias,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B=batch,
        D=dim,
        ST=state_len,
        T=seqlen,
        W=width,
        HAS_BIAS=bias is not None,
        SILU=activation in ["silu", "swish"],
        OUT_T=out_t,
    )

    hidden = torch.cat([conv_state, x], dim=-1)
    conv_state.copy_(hidden[:, :, -state_len:])

    if unsqueeze:
        out = out.squeeze(-1)
    return out

