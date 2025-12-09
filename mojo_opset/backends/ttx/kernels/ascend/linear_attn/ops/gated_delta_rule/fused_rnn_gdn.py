from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    T,
    cu_seqlens,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    bs: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_tasks = NV * HV * bs

    for task_id in range(pid, num_tasks, grid_size):
        i_n = task_id // (HV * NV)
        i_hv = (task_id // NV) % HV
        i_v = task_id % NV

        i_h = i_hv * (H // HV)
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            T_local = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
        else:
            bos = i_n * T
            T_local = T

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        p_q = q + (bos * H + i_h) * K + o_k

        p_k = k + (bos * HV + i_hv) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        p_beta = beta + bos * HV + i_hv
        p_g = g + bos * HV + i_hv
        p_o = o + (bos * H + i_h) * V + o_v
        p_ht = ht + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        if USE_INITIAL_STATE:
            p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
            b_h = tl.load(p_h0, mask=mask_h, other=0)
        else:
            b_h = tl.zeros([BK, BV], dtype=p_ht.dtype.element_ty)

        for _ in range(T_local):
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)
            b_h = b_h.to(tl.float32)

            if USE_QK_L2NORM_IN_KERNEL:
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            p_o_per_head = p_o
            p_q_per_head = p_q
            for _ in range(H // HV):
                b_q = tl.load(p_q_per_head, mask=mask_k, other=0).to(tl.float32)

                if USE_QK_L2NORM_IN_KERNEL:
                    b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
                b_o = tl.sum(b_h * b_q[:, None], 0) * scale

                tl.store(p_o_per_head, b_o.to(p_o.dtype.element_ty), mask=mask_v)
                p_o_per_head += V
                p_q_per_head += K

            p_q += H * K
            p_o += H * V
            p_k += HV * K
            p_v += HV * V
            p_g += HV
            p_beta += HV
            b_h = b_h.to(p_ht.dtype.element_ty)

        tl.store(p_ht, b_h, mask=mask_h)


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]

    if scale is None:
        scale = K**-0.5

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 16)
    NV = triton.cdiv(V, BV)

    bs = B if cu_seqlens is None else cu_seqlens.numel() - 1

    o = q.new_empty(B, T, H, V)
    final_state = q.new_empty(bs, HV, K, V, dtype=torch.float32)
    if initial_state is not None:
        initial_state = initial_state.contiguous()

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        o=o,
        cu_seqlens=cu_seqlens,
        scale=scale,
        h0=initial_state,
        ht=final_state,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NV=NV,
        bs=bs,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
    )
    return o, final_state
