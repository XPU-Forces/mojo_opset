"""
ILU Triton SDPA: masked softmax attention (GQA via repeat_interleave on host).
Backward path uses torch SDPA for correct GQA gradients.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .utils import libentry


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


@libentry()
@triton.jit
def _sdpa_masked_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    lse_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_v_d,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    stride_o_d,
    stride_m0,
    stride_m1,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    sm_scale,
    WRITE_LSE: tl.constexpr,
    OUT_T: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)
    total = B * H * S

    offs_d = tl.arange(0, D)

    for flat in tl.range(pid, total, pnum):
        b = flat // (H * S)
        rem = flat % (H * S)
        h = rem // S
        qi = rem % S

        q_base = b * stride_q_b + h * stride_q_h + qi * stride_q_s
        q_vec = tl.load(q_ptr + q_base + offs_d * stride_q_d, mask=offs_d < D, other=0.0).to(tl.float32)

        m_max = tl.full((), -float("inf"), tl.float32)
        for j in range(S):
            m_off = qi * stride_m0 + j * stride_m1
            allowed = tl.load(mask_ptr + m_off)
            k_base = b * stride_k_b + h * stride_k_h + j * stride_k_s
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=offs_d < D, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            m_max = tl.maximum(m_max, s)

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((D,), dtype=tl.float32)
        for j in range(S):
            m_off = qi * stride_m0 + j * stride_m1
            allowed = tl.load(mask_ptr + m_off)
            k_base = b * stride_k_b + h * stride_k_h + j * stride_k_s
            v_base = b * stride_v_b + h * stride_v_h + j * stride_v_s
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=offs_d < D, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + v_base + offs_d * stride_v_d, mask=offs_d < D, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            p = tl.exp(s - m_max)
            denom = denom + p
            acc = acc + p * v_vec

        out_vec = acc / denom
        o_base = b * stride_o_b + h * stride_o_h + qi * stride_o_s
        tl.store(out_ptr + o_base + offs_d * stride_o_d, out_vec.to(OUT_T), mask=offs_d < D)

        if WRITE_LSE:
            lse_val = m_max + tl.log(denom)
            lse_off = b * stride_lse_b + h * stride_lse_h + qi * stride_lse_s
            tl.store(lse_ptr + lse_off, lse_val)


def _launch_sdpa_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    lse: Optional[torch.Tensor],
) -> None:
    b, h, s, d = q.shape
    assert k.shape == (b, h, s, d) and v.shape == (b, h, s, d)
    mask = mask.contiguous()

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    write_lse = lse is not None
    if write_lse:
        assert lse.shape == (b, h, s) and lse.dtype == torch.float32
    else:
        lse = torch.empty(1, device=q.device, dtype=torch.float32)

    total_tasks = b * h * s
    block = 256
    grid = (triton.cdiv(total_tasks, block),)

    _sdpa_masked_fwd_kernel[grid](
        q,
        k,
        v,
        mask,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        mask.stride(0),
        mask.stride(1),
        lse.stride(0) if write_lse else 0,
        lse.stride(1) if write_lse else 0,
        lse.stride(2) if write_lse else 0,
        B=b,
        H=h,
        S=s,
        D=d,
        sm_scale=float(sm_scale),
        WRITE_LSE=write_lse,
        OUT_T=out_t,
    )


def sdpa_infer_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    assert q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1]
    assert q.shape[-2] == k.shape[-2] and k.shape[-2] == v.shape[-2]
    seq_length = q.shape[-2]
    assert mask is not None
    assert mask.shape == (seq_length, seq_length) and mask.dtype == torch.bool

    if not enable_gqa:
        assert q.shape[1] == k.shape[1] == v.shape[1]
    else:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0

    head_dim = q.shape[-1]
    sm_scale = scale if scale is not None else head_dim**-0.5

    n_rep = q.shape[1] // k.shape[1]
    k_eff = _repeat_kv(k, n_rep) if enable_gqa else k
    v_eff = _repeat_kv(v, n_rep) if enable_gqa else v

    q_c = q.contiguous()
    k_c = k_eff.contiguous()
    v_c = v_eff.contiguous()
    out = torch.empty_like(q_c)
    _launch_sdpa_masked(q_c, k_c, v_c, mask, out, sm_scale, lse=None)
    return out.to(q.dtype)


def sdpa_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
):
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]

    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] == v.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] == v.shape[3]

    head_dim = q.shape[-1]
    sm_scale = scale if scale is not None else head_dim**-0.5

    n_rep = q.shape[1] // k.shape[1]
    k_eff = _repeat_kv(k, n_rep) if gqa_enabled else k
    v_eff = _repeat_kv(v, n_rep) if gqa_enabled else v

    q_c = q.contiguous()
    k_c = k_eff.contiguous()
    v_c = v_eff.contiguous()
    out = torch.empty_like(q_c)
    lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _launch_sdpa_masked(q_c, k_c, v_c, mask, out, sm_scale, lse)
    return out, lse


def sdpa_bwd_impl(
    o: torch.Tensor,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
):
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4 and len(lse.shape) == 3
    assert q.shape == o.shape == do.shape
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]
    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] == v.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] == v.shape[3]
    assert q.shape[0] == lse.shape[0] and q.shape[1] == lse.shape[1] and q.shape[2] == lse.shape[2]

    q_ = q.detach().requires_grad_(True)
    k_ = k.detach().requires_grad_(True)
    v_ = v.detach().requires_grad_(True)
    kwargs = {"dropout_p": 0.0, "is_causal": False, "enable_gqa": gqa_enabled}
    if scale is not None:
        kwargs["scale"] = scale
    o_fwd = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, **kwargs)
    dq, dk, dv = torch.autograd.grad(o_fwd, (q_, k_, v_), do, retain_graph=False)
    return dq, dk, dv
