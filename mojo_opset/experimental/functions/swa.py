import torch
from mojo_opset.core.function import MojoFunction
from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_chunk_indices
from typing import Optional, Tuple

def generate_swa_mask(
    q_seq_len: int,
    kv_seq_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
) -> torch.Tensor:
    kv_cache_len = kv_seq_len - q_seq_len
    causal_mask = (torch.arange(0, q_seq_len)[:, None] + kv_cache_len) >= torch.arange(0, kv_seq_len)[None, :]
    if local_window_size is not None:
        local_window_mask = (
            torch.arange(0, q_seq_len)[:, None] + kv_cache_len
            <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
        ) & causal_mask
    else:
        local_window_mask = causal_mask
    if global_window_size is not None:
        global_window_mask = (torch.arange(0, kv_seq_len) < global_window_size)[
            None, :
        ] & causal_mask
    else:
        global_window_mask = causal_mask

    s_mask = local_window_mask | global_window_mask
    return s_mask

def swa_torch_forward(
    q: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
    k: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
    v: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor, # [bsz + 1]
    cu_seqlens_kv: torch.Tensor, # [bsz + 1]
    max_seqlen_q: int,
    max_seqlen_kv: int,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    sm_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_q_len, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty_like(q)
    softmax_lse = torch.empty((n_q_heads, total_q_len), dtype=torch.float32, device=q.device)
    bsz = cu_seqlens_q.shape[0] - 1
    for i in range(bsz):
        q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        q_seq_len = q_i.shape[0]
        q_i = q_i.permute(1, 0, 2) # -> [n_q_heads, q_seq_len, head_dim]

        k_i = k[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]]
        kv_seq_len = k_i.shape[0]
        k_i_T = k_i.permute(1, 2, 0)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i_T = k_i_T.repeat_interleave(n_q_heads // n_kv_heads, dim=0) # -> [n_q_heads, head_dim, kv_seq_len]
        s_i = torch.bmm(q_i, k_i_T).float() * sm_scale # -> [n_q_heads, q_seq_len, kv_seq_len]

        if is_causal:
            s_mask = generate_swa_mask(
                q_seq_len,
                kv_seq_len,
                local_window_size,
                global_window_size,
            ).to(s_i.device)
            s_i = torch.where(s_mask, s_i, float("-inf"))
        m_i = torch.max(s_i, dim=-1, keepdim=True).values # -> [n_q_heads, q_seq_len, 1]
        s_i = s_i - m_i # -> [n_q_heads, q_seq_len, kv_seq_len]
        p_i = torch.exp(s_i)
        l_i = torch.sum(p_i, dim=-1, keepdim=True) # -> [n_q_heads, q_seq_len, 1]
        p_i = p_i.to(v.dtype)

        v_i = v[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]].permute(1, 0, 2) # -> [n_kv_heads, kv_seq_len, head_dim]
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0) # -> [n_q_heads, kv_seq_len, head_dim]
        o_i = torch.bmm(p_i, v_i).float() # -> [n_q_heads, q_seq_len, head_dim]
        o_i = o_i / l_i # -> [n_q_heads, q_seq_len, head_dim]
        o_i = o_i.permute(1, 0, 2) # -> [q_seq_len, n_q_heads, head_dim]
        o[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = o_i.to(o.dtype)
        lse_i = m_i + torch.log(l_i) # -> [n_q_heads, q_seq_len, 1]
        assert lse_i.dtype == torch.float32
        softmax_lse[:, cu_seqlens_q[i]:cu_seqlens_q[i+1]] = lse_i.squeeze(-1) # -> [q_seq_len, n_q_heads]

    return o, softmax_lse

def swa_torch_backward(
    do: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
    q: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
    k: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
    v: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
    o: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
    softmax_lse: torch.Tensor, # [n_q_heads, total_q_len]
    cu_seqlens_q: torch.Tensor, # [bsz + 1]
    cu_seqlens_kv: torch.Tensor, # [bsz + 1]
    max_seqlen_q: int,
    max_seqlen_kv: int,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    sm_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.sum(o.float() * do.float(), dim=-1) # -> [total_q_len, n_q_heads]
    bsz = cu_seqlens_q.shape[0] - 1
    for i in range(bsz):

        # Step 1: recompute p_i
        q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        q_seq_len = q_i.shape[0]
        q_i = q_i.permute(1, 0, 2) # -> [n_q_heads, q_seq_len, head_dim]

        k_i = k[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]]
        kv_seq_len = k_i.shape[0]
        k_i = k_i.permute(1, 0, 2) # -> [n_kv_heads, kv_seq_len, head_dim]
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i = k_i.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i = k_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0) # -> [n_q_heads, head_dim, kv_seq_len]

        s_i = torch.bmm(q_i, k_i.mT).float() * sm_scale # -> [n_q_heads, q_seq_len, kv_seq_len]

        if is_causal:
            s_mask = generate_swa_mask(
                q_seq_len,
                kv_seq_len,
                local_window_size,
                global_window_size,
            ).to(s_i.device)
            s_i = torch.where(s_mask, s_i, float("-inf"))

        lse_i = softmax_lse[:, cu_seqlens_q[i]:cu_seqlens_q[i+1]] # -> [n_q_heads, q_seq_len]
        p_i = torch.exp(s_i - lse_i.unsqueeze(-1)) # -> [n_q_heads, q_seq_len, kv_seq_len]

        # Step 2: compute dv_i
        assert p_i.dtype == torch.float32
        v_i = v[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]]
        v_i = v_i.permute(1, 0, 2) # -> [n_kv_heads, kv_seq_len, head_dim]
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0) # -> [n_q_heads, kv_seq_len, head_dim]

        do_i = do[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        do_i = do_i.permute(1, 0, 2) # -> [n_q_heads, q_seq_len, head_dim]

        dp_i = torch.bmm(do_i, v_i.mT).float() # -> [n_q_heads, q_seq_len, kv_seq_len]
        assert dp_i.dtype == torch.float32
        # Note: rowsum(P * dP) => rowsum(P * (dO @ V_T)) => rowsum(dO * (P @ V)) => rowsum(dO * O)
        delta_i = delta[cu_seqlens_q[i]:cu_seqlens_q[i+1]].permute(1, 0).unsqueeze(-1) # -> [n_q_heads, q_seq_len]
        # print(f"{p_i.shape=} {do_i.shape=}")
        # delta_i_ref = torch.sum(p_i * dp_i, dim=-1, keepdim=True)
        # torch.testing.assert_close(delta_i, delta_i_ref)
        assert delta_i.dtype == torch.float32
        ds_i = p_i * (dp_i - delta_i)
        ds_i = ds_i * sm_scale
        assert ds_i.dtype == torch.float32
        ds_i = ds_i.to(do_i.dtype)

        p_i = p_i.to(do_i.dtype)

        dq_i = torch.bmm(ds_i, k_i) # -> [n_q_heads, q_seq_len, head_dim]
        dq[cu_seqlens_q[i]:cu_seqlens_q[i+1]] = dq_i.permute(1, 0, 2) # -> [q_seq_len, n_q_heads, head_dim]

        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                ds_i = ds_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(1, 0, 2, 3) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                q_i = q_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(1, 0, 2, 3) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]
            else:
                ds_i = ds_i.unflatten(0, (n_kv_heads, n_q_heads // n_kv_heads)) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                q_i = q_i.unflatten(0, (n_kv_heads, n_q_heads // n_kv_heads)) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]

            ds_i = ds_i.flatten(1, 2) # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, kv_seq_len]
            q_i = q_i.flatten(1, 2) # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, head_dim]
            
        dk_i = torch.bmm(ds_i.mT, q_i) # -> [n_kv_heads, kv_seq_len, head_dim]
        dk[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]] = dk_i.permute(1, 0, 2) # -> [kv_seq_len, n_kv_heads, head_dim]
        
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                p_i = p_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(1, 0, 2, 3) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                do_i = do_i.unflatten(0, (n_q_heads // n_kv_heads, n_kv_heads)).permute(1, 0, 2, 3) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]
            else:
                p_i = p_i.unflatten(0, (n_kv_heads, n_q_heads // n_kv_heads)) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, kv_seq_len]
                do_i = do_i.unflatten(0, (n_kv_heads, n_q_heads // n_kv_heads)) # -> [n_kv_heads, n_q_heads // n_kv_heads, q_seq_len, head_dim]

            p_i = p_i.flatten(1, 2) # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, kv_seq_len]
            do_i = do_i.flatten(1, 2) # -> [n_kv_heads, n_q_heads // n_kv_heads * q_seq_len, head_dim]

        dv_i = torch.bmm(p_i.mT, do_i) # -> [n_q_heads, kv_seq_len, head_dim]
        dv[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]] = dv_i.permute(1, 0, 2)# -> [kv_seq_len, n_kv_heads, head_dim]
    return dq, dk, dv


class MojoSWAFunction(MojoFunction):

    @staticmethod
    def forward(
        ctx, 
        q: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor, # [bsz + 1]
        cu_seqlens_kv: torch.Tensor, # [bsz + 1]
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        o, softmax_lse = swa_torch_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, is_causal, local_window_size, global_window_size, sm_scale, gqa_interleave
        )

        ctx.save_for_backward(o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.is_causal = is_causal
        ctx.local_window_size = local_window_size
        ctx.global_window_size = global_window_size
        ctx.gqa_interleave = gqa_interleave
        return o


    def backward(
        ctx, 
        do: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               None, None, None, None, None, None, None, None, None]:
        o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_kv = ctx.max_seqlen_kv
        is_causal = ctx.is_causal
        local_window_size = ctx.local_window_size
        global_window_size = ctx.global_window_size
        gqa_interleave = ctx.gqa_interleave

        dq, dk, dv = swa_torch_backward(
            do, q, k, v, o, softmax_lse, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, is_causal, local_window_size, global_window_size, sm_scale, gqa_interleave
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None

AUX_MASK_SIZE = 256
AUX_MASK = torch.cat([
    torch.cat([
        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).triu().bool(),
        torch.ones(AUX_MASK_SIZE,AUX_MASK_SIZE).bool(),
        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).tril().bool(),
        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool()
    ], dim=1),
    torch.cat([
        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
    ], dim=1),
], dim=0).npu()

import triton
import triton.language as tl

@triton.jit
def gen_mask_n_right_bound(mask_10_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, n_start, right):
  # tl.arange(n_start, n_start + N_BLOCK)[None, :] < right
  offset = min(max(n_start - right, -mask_size), 0)
  mask = tl.load(mask_10_ptr + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n)
  return mask.to(tl.int1)

@triton.jit
def gen_mask_n_left_bound(mask_01_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, n_start, left):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= left
    offset = min(max(n_start - left, -mask_size), 0)
    mask = tl.load(mask_01_ptr + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n)
    return mask.to(tl.int1)

@triton.jit
def gen_mask_m_right_bound(mask_10t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, right):
  # tl.arange(m_start, m_start + M_BLOCK)[:, None] < right
  offset = min(max(m_start - right, -mask_size), 0)
  mask = tl.load(mask_10t_ptr + (offset + tl.arange(0, M_BLOCK)[:, None]) * mask_stride_m + tl.arange(0, N_BLOCK)[None, :] * mask_stride_n)
  return mask.to(tl.int1)

@triton.jit
def gen_mask_m_left_bound(mask_01t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, left):
    # tl.arange(m_start, m_start + M_BLOCK)[:, None] >= left
    offset = min(max(m_start - left, -mask_size), 0)
    mask = tl.load(mask_01t_ptr + (offset + tl.arange(0, M_BLOCK)[:, None]) * mask_stride_m + tl.arange(0, N_BLOCK)[None, :] * mask_stride_n)
    return mask.to(tl.int1)

@triton.jit
def gen_mask_tril(mask_ptr_tril, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] <= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(mask_ptr_tril + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n)
    return mask.to(tl.int1)

@triton.jit
def gen_mask_triu(mask_ptr_triu, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    len_offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(mask_ptr_triu + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m + (len_offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n)
    return mask.to(tl.int1)

@triton.jit
def _sdpa_single_block_fwd(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask,
    qk_scale,
    offs_m,
    offs_n,
    seq_m,
    seq_n,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return acc_ptr, l_i, m_i
    # -- Compute qk ----

    # Load (transposed) K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T)
    # tl.compile_hint(qk, "tile_cube_loop")

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, -1e6)  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k_T.dtype)
    
    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, 1)  
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)
    # tl.compile_hint(acc_ptr, "tile_cube_loop")

    # Update current block max
    m_i = m_ij  

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i

@triton.jit
def _sdpa_fwd_kernel(
    o_ptr,
    lse_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    scale,
    q_chunk_indices_ptr,
    num_q_chunks,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lse_h,
    stride_lse_t,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    aux_mask_ptr,
    aux_mask_size,
    stride_mask_m,
    stride_mask_n,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE == BLOCK_M, "Currently only support CHUNK_SIZE == BLOCK_SIZE_M")
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    # Hint(chenyifan): 
    #   the prepared aux_mask is [[empty, triu, full, tril, empty],
    #                             [full, empty, empty, full, empty]]
    #   every mask [BLOCK_M, BLOCK_N] can be sliced from the aux_mask and further combined
    aux_mask_ptr_01 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_10 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_triu = aux_mask_ptr + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_tril = aux_mask_ptr + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_01t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m
    aux_mask_ptr_10t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 2 * stride_mask_n

    num_tasks = num_q_chunks * NUM_Q_HEADS
    for task_id in range(pid, num_tasks, n_programs):
        chunk_id = task_id // NUM_Q_HEADS
        q_head_id = task_id % NUM_Q_HEADS
        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        b_id = tl.load(q_chunk_indices_ptr + chunk_id * 2)
        q_block_id = tl.load(q_chunk_indices_ptr + chunk_id * 2 + 1)

        q_start = tl.load(cu_seqlens_q_ptr + b_id)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1)
        kv_start = tl.load(cu_seqlens_kv_ptr + b_id)
        kv_end = tl.load(cu_seqlens_kv_ptr + b_id + 1)

        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        block_len = min(BLOCK_M, q_seq_len - q_block_id * BLOCK_M)
        q_block_ptr = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
            shape=(q_seq_len, HEAD_DIM),
            strides=(stride_qt, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1,0),
        )
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
            shape=(q_seq_len, HEAD_DIM),
            strides=(stride_ot, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1,0),
        )
        lse_i_ptr = lse_ptr + q_head_id * stride_lse_h + q_start * stride_lse_t
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
            shape=(kv_seq_len, HEAD_DIM),
            strides=(stride_kt, stride_kd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1,0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
            shape=(kv_seq_len, HEAD_DIM),
            strides=(stride_vt, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1,0),
        )
        if block_len > 0:
            q_mask = gen_mask_m_right_bound(
                aux_mask_ptr_10t, aux_mask_size, stride_mask_m, stride_mask_n,
                BLOCK_M, BLOCK_N,
                q_block_id * BLOCK_M, q_seq_len,
            )
            cur_q_block_ptr = tl.advance(q_block_ptr, ((q_block_id * BLOCK_M).to(tl.int32), 0))
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            q_block_start = q_block_id * BLOCK_M + kv_computed_len

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            if IS_CAUSAL:
                num_total_kv_blocks = tl.cdiv(min((q_block_id + 1) * BLOCK_M + kv_computed_len, kv_seq_len), BLOCK_N)
                if GLOBAL_WINDOW is not None:
                    num_global_window_blocks = min(tl.cdiv(GLOBAL_WINDOW, BLOCK_N), num_total_kv_blocks)
                    for kv_block_id in range(0, num_global_window_blocks):
                        kv_block_start = kv_block_id * BLOCK_N
                        kv_block_len = min(BLOCK_N, kv_seq_len - kv_block_start)
                        mask = gen_mask_n_right_bound(
                            aux_mask_ptr_10, aux_mask_size, stride_mask_m, stride_mask_n,
                            BLOCK_M, BLOCK_N,
                            kv_block_start, GLOBAL_WINDOW,
                        )
                        if LOCAL_WINDOW is not None:
                            mask_sw = gen_mask_triu(
                                aux_mask_ptr_triu, aux_mask_size, stride_mask_m, stride_mask_n,
                                BLOCK_M, BLOCK_N,
                                q_block_start, kv_block_start + LOCAL_WINDOW,
                            )
                            mask = mask | mask_sw
                        mask_causal = gen_mask_tril(
                            aux_mask_ptr_tril, aux_mask_size, stride_mask_m, stride_mask_n,
                            BLOCK_M, BLOCK_N,
                            q_block_start, kv_block_start, 
                        )
                        mask = mask & mask_causal
                        mask = mask & q_mask
                        cur_k_block_ptr = tl.advance(k_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                        cur_v_block_ptr = tl.advance(v_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                        acc, l_i, m_i = _sdpa_single_block_fwd(
                            acc, l_i, m_i, cur_q_block, cur_k_block_ptr, cur_v_block_ptr, mask, scale,
                            q_block_start, kv_block_start, block_len, kv_block_len,
                            HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 
                            v_ptr.dtype.element_ty == tl.float8e5,
                        )
                else:
                    num_global_window_blocks = 0
            else:
                num_total_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
                num_global_window_blocks = 0

            if IS_CAUSAL:
                if LOCAL_WINDOW is not None:
                    sw_start_block = max(q_block_start - LOCAL_WINDOW, 0) //  BLOCK_N
                    start_block = max(sw_start_block, num_global_window_blocks)
                elif GLOBAL_WINDOW is None:
                    start_block = 0
                else:
                    # Global window has been computed, but local window is None, so no more kvblocks
                    start_block = num_total_kv_blocks
            else:
                start_block = 0
            for kv_block_id in range(start_block, num_total_kv_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_len = min(BLOCK_N, kv_seq_len - kv_block_start)
                if IS_CAUSAL:
                    mask = gen_mask_tril(
                        aux_mask_ptr_tril, aux_mask_size, stride_mask_m, stride_mask_n,
                        BLOCK_M, BLOCK_N,
                        q_block_start, kv_block_start,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu, aux_mask_size, stride_mask_m, stride_mask_n,
                            BLOCK_M, BLOCK_N,
                            q_block_start, kv_block_start + LOCAL_WINDOW,
                        )
                        mask = mask & mask_sw
                else:
                    mask = gen_mask_n_right_bound(
                        aux_mask_ptr_10, aux_mask_size, stride_mask_m, stride_mask_n,
                        BLOCK_M, BLOCK_N,
                        kv_block_start, kv_seq_len,
                    )
                mask = mask & q_mask
                cur_k_block_ptr = tl.advance(k_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                cur_v_block_ptr = tl.advance(v_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                acc, l_i, m_i = _sdpa_single_block_fwd(
                    acc, l_i, m_i, cur_q_block, cur_k_block_ptr, cur_v_block_ptr, mask, scale,
                    q_block_start, kv_block_start, block_len, kv_block_len,
                    HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            lse_i = m_i + tl.math.log(l_i)
            lse_i_offs = (q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
            lse_i_mask = lse_i_offs < q_seq_len
            tl.store(lse_i_ptr + lse_i_offs * stride_lse_t, lse_i, mask=lse_i_mask)
            cur_o_block_ptr = tl.advance(o_block_ptr, ((q_block_id * BLOCK_M).to(tl.int32), 0))
            accumulator = acc / l_i[:, None]
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0,1))

def _sdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor, # [bsz + 1]
    cu_seqlens_kv: torch.Tensor, # [bsz + 1]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    sm_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tot_q_toks, num_q_heads, head_dim = q.shape
    tot_kv_toks, num_kv_heads, _ = k.shape
    o = torch.zeros_like(q)
    softmax_lse = torch.zeros((num_q_heads, tot_q_toks), dtype=torch.float32, device=q.device) + float('-inf')
    if q.dtype == torch.float32:
        CHUNK_SIZE = 64
        BLOCK_M = CHUNK_SIZE
        BLOCK_N = 64
    else:
        CHUNK_SIZE = 64
        BLOCK_M = CHUNK_SIZE
        BLOCK_N = 64
    q_chunk_indices = prepare_chunk_indices(cu_seqlens_q, CHUNK_SIZE)
    print(f"{q_chunk_indices=}")
    
    BLOCK_D = head_dim
    cube_num = get_num_cores("cube")
    
    grid = (cube_num, )

    _sdpa_fwd_kernel[grid](
        o, softmax_lse, q, k, v, 
        cu_seqlens_q, cu_seqlens_kv, sm_scale,
        q_chunk_indices, q_chunk_indices.shape[0],
        o.stride(0), o.stride(1), o.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        AUX_MASK, AUX_MASK_SIZE, AUX_MASK.stride(0), AUX_MASK.stride(1),
        is_causal,global_window_size, local_window_size,
        num_q_heads, num_kv_heads, gqa_interleave, head_dim,
        CHUNK_SIZE, BLOCK_M, BLOCK_N, BLOCK_D,
    )
    return o, softmax_lse


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 32}),
        triton.Config(kwargs={"BLOCK_SIZE": 64}),
        triton.Config(kwargs={"BLOCK_SIZE": 128}),
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _sdpa_bwd_preprocess(
        d_ptr: torch.Tensor,
        o_ptr: torch.Tensor,
        do_ptr: torch.Tensor,
        num_tokens: int,
        d_stride_h: int, d_stride_t: int,
        o_stride_t: int, o_stride_h: int, o_stride_d: int,
        do_stride_t: int, do_stride_h: int, do_stride_d: int,
        NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    num_blocks = tl.cdiv(num_tokens, BLOCK_SIZE)
    num_tasks = num_blocks * NUM_HEADS
    tl.static_assert(d_ptr.type.element_ty == tl.float32)
    for task_id in range(pid, num_tasks, n_progs):
        block_id = task_id // NUM_HEADS
        head_id = task_id % NUM_HEADS
        t_offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        t_mask = t_offs < num_tokens
        o = tl.load(o_ptr + t_offs[:, None] * o_stride_t + head_id * o_stride_h + tl.arange(0, HEAD_DIM)[None, :] * o_stride_d, mask=t_mask[:, None], other=0.0)
        do = tl.load(do_ptr + t_offs[:, None] * do_stride_t + head_id * do_stride_h + tl.arange(0, HEAD_DIM)[None, :] * do_stride_d, mask=t_mask[:, None], other=0.0)
        delta = tl.sum(o.cast(tl.float32) * do.cast(tl.float32), axis=-1)
        tl.store(d_ptr + head_id * d_stride_h + t_offs * d_stride_t, delta, mask=t_mask)


@triton.jit
def _sdpa_single_block_bwd_dkdv(
    dk_ptr,
    dv_ptr,
    d,
    lse,
    Q_block_ptr,
    DO_block_ptr,
    k,
    v,
    mask,
    qk_scale,
    offs_m,
    offs_n,
    seq_m,
    seq_n,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return dk_ptr, dv_ptr
    # -- Compute qk ----

    # Load (transposed) K block
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_T = tl.trans(q)
    qkT = tl.dot(k, q_T) # [BLOCK_N, BLOCK_M]
    # tl.compile_hint(qk, "tile_cube_loop")
    qkT = qkT * qk_scale

    # -- Compute p ----
    # Softmax weights p = exp(qk - lse.unsqueeze(1))
    # a.k.a. pT = exp(qkT - lse.unsqueeze(0))
    # scale according to recorded logsumexp
    pT = tl.math.exp(qkT - lse[None, :])
    if mask is not None and mask is not True:
        pT = tl.where(mask, pT, 0.0)  # 32B # bool
    pT_cast = pT.to(q_T.dtype)

    # -- Compute dV ----
    # dv = pT @ do
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dv = tl.dot(pT_cast, do, dv_ptr)

    # -- Compute dS ----
    # dpT = v @ doT
    doT = tl.trans(do)
    dpT = tl.dot(v, doT)

    # dsT = pT * (dpT - dT)
    dsT = pT * (dpT - d[None, :]) * qk_scale
    dsT_cast = dsT.to(q_T.dtype)
    
    # -- Compute dK ----
    # dk = dsT @ q
    dk = tl.dot(dsT_cast, q, dk_ptr)

    return dk, dv


@triton.jit
def _sdpa_single_block_bwd_dq(
    dq_ptr,
    d,
    lse,
    q,
    do,
    K_block_ptr,
    V_block_ptr,
    mask,
    qk_scale,
    offs_m,
    offs_n,
    seq_m,
    seq_n,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return dq_ptr
    # -- Compute qk ----

    # Load (transposed) K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T) # [BLOCK_M, BLOCK_N]
    # tl.compile_hint(qk, "tile_cube_loop")
    qk = qk * qk_scale

    # -- Compute p ----
    # Softmax weights p = exp(qk - lse)
    # scale according to recorded logsumexp
    p = tl.math.exp(qk - lse[:, None])
    if mask is not None and mask is not True:
        p = tl.where(mask, p, 0.0)  # 32B # bool

    # -- Compute dS ----
    # dp = do @ v.T
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_T = tl.trans(v)
    dp = tl.dot(do, v_T)

    # ds = p * (dp - d)
    ds = p * (dp - d[:, None]) * qk_scale
    ds_cast = ds.to(q.dtype)
    
    # -- Compute dK ----
    # dq = ds @ k
    dq = tl.dot(ds_cast, k, dq_ptr)

    return dq


@triton.jit
def _sdpa_bwd_dkdv_kernel(
    dk_ptr,
    dv_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    scale,
    kv_chunk_indices_ptr,
    num_kv_chunks,
    stride_dkt,
    stride_dkh,
    stride_dkd,
    stride_dvt,
    stride_dvh,
    stride_dvd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_delta_h,
    stride_delta_t,
    stride_lse_h,
    stride_lse_t,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    aux_mask_ptr,
    aux_mask_size,
    stride_mask_m,
    stride_mask_n,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE == BLOCK_N, "Currently only support CHUNK_SIZE == BLOCK_N")
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    # Hint(chenyifan): 
    #   the prepared aux_mask is [[empty, triu, full, tril, empty],
    #                             [full, empty, empty, full, empty]]
    #   every mask [BLOCK_M, BLOCK_N] can be sliced from the aux_mask and further combined
    aux_mask_ptr_01 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_10 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_triu = aux_mask_ptr + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_tril = aux_mask_ptr + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_01t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m
    aux_mask_ptr_10t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 2 * stride_mask_n

    num_kv_tasks = num_kv_chunks * NUM_KV_HEADS
    for kv_task_id in range(pid, num_kv_tasks, n_programs):
        kv_chunk_id = kv_task_id // NUM_KV_HEADS
        kv_head_id = kv_task_id % NUM_KV_HEADS
        b_id = tl.load(kv_chunk_indices_ptr + kv_chunk_id * 2)

        kv_block_id = tl.load(kv_chunk_indices_ptr + kv_chunk_id * 2 + 1)

        kv_start = tl.load(cu_seqlens_kv_ptr + b_id)
        kv_end = tl.load(cu_seqlens_kv_ptr + b_id + 1)
        q_start = tl.load(cu_seqlens_q_ptr + b_id)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1)

        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        block_len = min(BLOCK_N, kv_seq_len - kv_block_id * BLOCK_N)

        if block_len > 0:
            dk_block_ptr = tl.make_block_ptr(
                base=dk_ptr + kv_start * stride_dkt + kv_head_id * stride_dkh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_dkt, stride_dkd),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1,0),
            )
            dv_block_ptr = tl.make_block_ptr(
                base=dv_ptr + kv_start * stride_dvt + kv_head_id * stride_dvh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_dvt, stride_dvd),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1,0),
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                shape=(kv_seq_len, HEAD_DIM),
                strides=(stride_kt, stride_kd),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1,0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                shape=(kv_seq_len, HEAD_DIM),
                strides=(stride_vt, stride_vd),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1,0),
            )

            cur_k_block_ptr = tl.advance(k_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
            cur_k_block = tl.load(cur_k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            cur_v_block_ptr = tl.advance(v_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
            cur_v_block = tl.load(cur_v_block_ptr, boundary_check=(0, 1), padding_option="zero")
   
            kv_block_start = kv_block_id * BLOCK_N

            dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
            dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

            # For GQA, iterate over all q_heads
            for q_head_rpt in tl.static_range(NUM_Q_HEADS // NUM_KV_HEADS):
                if GQA_INTERLEAVE:
                    q_head_id = NUM_KV_HEADS * q_head_rpt + kv_head_id
                else:
                    q_head_id = q_head_rpt + kv_head_id * (NUM_Q_HEADS // NUM_KV_HEADS)

                q_block_ptr = tl.make_block_ptr(
                    base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                    shape=(q_seq_len, HEAD_DIM),
                    strides=(stride_qt, stride_qd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1,0),
                )
                do_block_ptr = tl.make_block_ptr(
                    base=do_ptr + q_start * stride_dot + q_head_id * stride_doh,
                    shape=(q_seq_len, HEAD_DIM),
                    strides=(stride_dot, stride_dod),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1,0),
                )
                lse_i_ptr = lse_ptr + q_head_id * stride_lse_h + q_start * stride_lse_t
                delta_i_ptr = delta_ptr + q_head_id * stride_delta_h + q_start * stride_delta_t

                if IS_CAUSAL:
                    cur_q_start = max(kv_block_start - kv_computed_len, 0)
                    if GLOBAL_WINDOW is None and LOCAL_WINDOW is None:
                        # vanilla attention, iterate over all queries
                        cur_q_end = q_seq_len
                    elif GLOBAL_WINDOW is not None and kv_block_start < GLOBAL_WINDOW:
                        # sink token is attended by all succeeding tokens
                        cur_q_end = q_seq_len
                    elif LOCAL_WINDOW is not None:
                        # otherwise, it can only be attented as sliding window tokens
                        cur_q_end = min(max(kv_block_start + BLOCK_N + LOCAL_WINDOW - kv_computed_len, 0), q_seq_len)
                    else:
                        # Not attended
                        cur_q_end = cur_q_start
                else:
                    # full attention, iterate over all queries
                    cur_q_start = 0
                    cur_q_end = q_seq_len

                num_total_q_blocks = tl.cdiv(cur_q_end - cur_q_start, BLOCK_M)
            
                for q_block_id in range(0, num_total_q_blocks):
                    q_block_start = cur_q_start + q_block_id * BLOCK_M
                    q_block_len = min(BLOCK_M, cur_q_end - q_block_start)
                    q_mask = gen_mask_n_right_bound(
                        aux_mask_ptr_10, aux_mask_size, stride_mask_m, stride_mask_n,
                        BLOCK_N, BLOCK_M,
                        q_block_start, q_seq_len,
                    )
                    mask = gen_mask_m_right_bound(
                        aux_mask_ptr_10t, aux_mask_size, stride_mask_m, stride_mask_n,
                        BLOCK_N, BLOCK_M,
                        kv_block_start, GLOBAL_WINDOW,
                    )
                    
                    k_mask = gen_mask_m_right_bound(
                        aux_mask_ptr_10t, aux_mask_size, stride_mask_m, stride_mask_n,
                        BLOCK_N, BLOCK_M,
                        kv_block_start, kv_seq_len,
                    )
                    if IS_CAUSAL:
                        if LOCAL_WINDOW is not None:
                            mask_sw = gen_mask_tril(
                                aux_mask_ptr_tril, aux_mask_size, stride_mask_m, stride_mask_n,
                                BLOCK_N, BLOCK_M,
                                kv_block_start + LOCAL_WINDOW, q_block_start + kv_computed_len,
                            )

                            mask = mask | mask_sw
                        mask_causal = gen_mask_triu(
                            aux_mask_ptr_triu, aux_mask_size, stride_mask_m, stride_mask_n,
                            BLOCK_N, BLOCK_M,
                            kv_block_start, q_block_start + kv_computed_len
                        )
                        mask = mask & mask_causal
                    mask = mask & q_mask & k_mask

                    cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
                    cur_do_block_ptr = tl.advance(do_block_ptr, (q_block_start.to(tl.int32), 0))
                    q_offs = q_block_start + tl.arange(0, BLOCK_M)

                    cur_delta = tl.load(delta_i_ptr + q_offs * stride_delta_t, mask=q_offs < q_seq_len, other=0.0)
                    tl.static_assert(cur_delta.dtype == tl.float32)
                    cur_lse = tl.load(lse_i_ptr + q_offs * stride_lse_t, mask=q_offs < q_seq_len, other=-float("inf"))
                    tl.static_assert(cur_lse.dtype == tl.float32)
                    dk, dv = _sdpa_single_block_bwd_dkdv(
                        dk, dv, 
                        cur_delta, cur_lse,
                        cur_q_block_ptr, cur_do_block_ptr,
                        cur_k_block, cur_v_block, mask, scale,
                        q_block_start + kv_computed_len, kv_block_start,
                        q_block_len, block_len,
                        HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 
                        v_ptr.dtype.element_ty == tl.float8e5,
                    )

            cur_dk_block_ptr = tl.advance(dk_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
            tl.store(cur_dk_block_ptr, dk.to(dk_ptr.type.element_ty), boundary_check=(0,1))
            cur_dv_block_ptr = tl.advance(dv_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
            tl.store(cur_dv_block_ptr, dv.to(dv_ptr.type.element_ty), boundary_check=(0,1))
           


@triton.jit
def _sdpa_bwd_dq_kernel(
    dq_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    scale,
    q_chunk_indices_ptr,
    num_q_chunks,
    stride_dqt,
    stride_dqh,
    stride_dqd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_delta_h,
    stride_delta_t,
    stride_lse_h,
    stride_lse_t,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    aux_mask_ptr,
    aux_mask_size,
    stride_mask_m,
    stride_mask_n,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE == BLOCK_M, "Currently only support CHUNK_SIZE == BLOCK_SIZE_M")
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    # Hint(chenyifan): 
    #   the prepared aux_mask is [[empty, triu, full, tril, empty],
    #                             [full, empty, empty, full, empty]]
    #   every mask [BLOCK_M, BLOCK_N] can be sliced from the aux_mask and further combined
    aux_mask_ptr_01 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_10 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_triu = aux_mask_ptr + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_tril = aux_mask_ptr + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_01t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m
    aux_mask_ptr_10t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 2 * stride_mask_n

    num_tasks = num_q_chunks * NUM_Q_HEADS
    for task_id in range(pid, num_tasks, n_programs):
        chunk_id = task_id // NUM_Q_HEADS
        q_head_id = task_id % NUM_Q_HEADS
        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        b_id = tl.load(q_chunk_indices_ptr + chunk_id * 2)
        q_block_id = tl.load(q_chunk_indices_ptr + chunk_id * 2 + 1)

        q_start = tl.load(cu_seqlens_q_ptr + b_id)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1)
        kv_start = tl.load(cu_seqlens_kv_ptr + b_id)
        kv_end = tl.load(cu_seqlens_kv_ptr + b_id + 1)

        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        block_len = min(BLOCK_M, q_seq_len - q_block_id * BLOCK_M)
        q_block_ptr = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
            shape=(q_seq_len, HEAD_DIM),
            strides=(stride_qt, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1,0),
        )
        do_block_ptr = tl.make_block_ptr(
            base=do_ptr + q_start * stride_dot + q_head_id * stride_doh,
            shape=(q_seq_len, HEAD_DIM),
            strides=(stride_dot, stride_dod),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1,0),
        )
        dq_block_ptr = tl.make_block_ptr(
            base=dq_ptr + q_start * stride_dqt + q_head_id * stride_dqh,
            shape=(q_seq_len, HEAD_DIM),
            strides=(stride_dqt, stride_dqd),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1,0),
        )
        delta_i_ptr = delta_ptr + q_start * stride_delta_t + q_head_id * stride_delta_h
        lse_i_ptr = lse_ptr + q_head_id * stride_lse_h + q_start * stride_lse_t
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
            shape=(kv_seq_len, HEAD_DIM),
            strides=(stride_kt, stride_kd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1,0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
            shape=(kv_seq_len, HEAD_DIM),
            strides=(stride_vt, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1,0),
        )
        if block_len > 0:
            q_mask = gen_mask_m_right_bound(
                aux_mask_ptr_10t, aux_mask_size, stride_mask_m, stride_mask_n,
                BLOCK_M, BLOCK_N,
                q_block_id * BLOCK_M, q_seq_len,
            )
            cur_q_block_ptr = tl.advance(q_block_ptr, ((q_block_id * BLOCK_M).to(tl.int32), 0))
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            cur_do_block_ptr = tl.advance(do_block_ptr, ((q_block_id * BLOCK_M).to(tl.int32), 0))
            cur_do_block = tl.load(cur_do_block_ptr, boundary_check=(0, 1), padding_option="zero")

            q_offs = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
            cur_delta = tl.load(delta_i_ptr + q_offs, q_offs < q_seq_len, other=0.0)
            cur_lse  = tl.load(lse_i_ptr + q_offs, q_offs < q_seq_len, other=-float("inf"))

            dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            q_block_start = q_block_id * BLOCK_M + kv_computed_len

            if IS_CAUSAL:
                num_total_kv_blocks = tl.cdiv(min((q_block_id + 1) * BLOCK_M + kv_computed_len, kv_seq_len), BLOCK_N)
                if GLOBAL_WINDOW is not None:
                    num_global_window_blocks = min(tl.cdiv(GLOBAL_WINDOW, BLOCK_N), num_total_kv_blocks)
                    for kv_block_id in range(0, num_global_window_blocks):
                        kv_block_start = kv_block_id * BLOCK_N
                        kv_block_len = min(BLOCK_N, kv_seq_len - kv_block_start)
                        mask = gen_mask_n_right_bound(
                            aux_mask_ptr_10, aux_mask_size, stride_mask_m, stride_mask_n,
                            BLOCK_M, BLOCK_N,
                            kv_block_start, GLOBAL_WINDOW,
                        )
                        if LOCAL_WINDOW is not None:
                            mask_sw = gen_mask_triu(
                                aux_mask_ptr_triu, aux_mask_size, stride_mask_m, stride_mask_n,
                                BLOCK_M, BLOCK_N,
                                q_block_start, kv_block_start + LOCAL_WINDOW,
                            )
                            mask = mask | mask_sw
                        mask_causal = gen_mask_tril(
                            aux_mask_ptr_tril, aux_mask_size, stride_mask_m, stride_mask_n,
                            BLOCK_M, BLOCK_N,
                            q_block_start, kv_block_start, 
                        )
                        mask = mask & mask_causal
                        mask = mask & q_mask
                        cur_k_block_ptr = tl.advance(k_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                        cur_v_block_ptr = tl.advance(v_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                        dq = _sdpa_single_block_bwd_dq(
                            dq, cur_delta, cur_lse, cur_q_block, cur_do_block, cur_k_block_ptr, cur_v_block_ptr, mask, scale,
                            q_block_start, kv_block_start, block_len, kv_block_len,
                            HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 
                            v_ptr.dtype.element_ty == tl.float8e5,
                        )
                else:
                    num_global_window_blocks = 0
            else:
                num_total_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
                num_global_window_blocks = 0
            if IS_CAUSAL:
                if LOCAL_WINDOW is not None:
                    sw_start_block = max(q_block_start - LOCAL_WINDOW, 0) //  BLOCK_N
                    start_block = max(sw_start_block, num_global_window_blocks)
                elif GLOBAL_WINDOW is None:
                    start_block = 0
                else:
                    # Global window has been computed, but local window is None, so no more kvblocks
                    start_block = num_total_kv_blocks
            else:
                start_block = 0
            for kv_block_id in range(start_block, num_total_kv_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_len = min(BLOCK_N, kv_seq_len - kv_block_start)
                if IS_CAUSAL:
                    mask = gen_mask_tril(
                        aux_mask_ptr_tril, aux_mask_size, stride_mask_m, stride_mask_n,
                        BLOCK_M, BLOCK_N,
                        q_block_start, kv_block_start,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu, aux_mask_size, stride_mask_m, stride_mask_n,
                            BLOCK_M, BLOCK_N,
                            q_block_start, kv_block_start + LOCAL_WINDOW,
                        )
                        mask = mask & mask_sw
                else:
                    mask = gen_mask_n_right_bound(
                        aux_mask_ptr_10, aux_mask_size, stride_mask_m, stride_mask_n,
                        BLOCK_M, BLOCK_N,
                        kv_block_start, kv_seq_len,
                    )
                mask = mask & q_mask
                cur_k_block_ptr = tl.advance(k_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                cur_v_block_ptr = tl.advance(v_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                dq = _sdpa_single_block_bwd_dq(
                    dq, cur_delta, cur_lse, cur_q_block, cur_do_block, cur_k_block_ptr, cur_v_block_ptr, mask, scale,
                    q_block_start, kv_block_start, block_len, kv_block_len,
                    HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 
                    v_ptr.dtype.element_ty == tl.float8e5,
                )
            cur_dq_block_ptr = tl.advance(dq_block_ptr, ((q_block_id * BLOCK_M).to(tl.int32), 0))
            tl.store(cur_dq_block_ptr, dq.to(dq_block_ptr.type.element_ty), boundary_check=(0,1))


def _sdpa_bwd(
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        softmax_lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_causal: bool,
        local_window_size: Optional[int],
        global_window_size: Optional[int],
        sm_scale: Optional[float],
        gqa_interleave: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tot_q_toks, num_q_heads, head_dim = q.shape

    delta = torch.zeros((num_q_heads, tot_q_toks), dtype=torch.float32, device=q.device)
    o = o.contiguous()
    do = do.contiguous()

    num_vecs = get_num_cores('vector')
    _sdpa_bwd_preprocess[(num_vecs,)](
        delta, o, do, tot_q_toks,
        delta.stride(0), delta.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        num_q_heads, head_dim,
    )
    

    _, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    dk = torch.zeros_like(k).contiguous()
    dv = torch.zeros_like(v).contiguous()

    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 64
        BLOCK_N = 64
    CHUNK_SIZE = BLOCK_N
    kv_chunk_indices = prepare_chunk_indices(cu_seqlens_kv, CHUNK_SIZE)
    print(f"{kv_chunk_indices=}")
    
    BLOCK_D = head_dim
    cube_num = get_num_cores("cube")
    
    grid = (cube_num, )

    _sdpa_bwd_dkdv_kernel[grid](
        dk, dv, do, delta, softmax_lse, q, k, v, 
        cu_seqlens_q, cu_seqlens_kv, sm_scale,
        kv_chunk_indices, kv_chunk_indices.shape[0],
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1),
        softmax_lse.stride(0), softmax_lse.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        AUX_MASK, AUX_MASK_SIZE, AUX_MASK.stride(0), AUX_MASK.stride(1),
        is_causal,global_window_size, local_window_size,
        num_q_heads, num_kv_heads, gqa_interleave, head_dim,
        CHUNK_SIZE, BLOCK_M, BLOCK_N, BLOCK_D,
    )
    
    CHUNK_SIZE = BLOCK_M
    q_chunk_indices = prepare_chunk_indices(cu_seqlens_q, CHUNK_SIZE)
    print(f"{q_chunk_indices=}")
    dq = torch.zeros_like(q)
    _sdpa_bwd_dq_kernel[grid](
        dq, do, delta, softmax_lse, q, k, v, 
        cu_seqlens_q, cu_seqlens_kv, sm_scale,
        q_chunk_indices, q_chunk_indices.shape[0],
        dq.stride(0), dq.stride(1), dq.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1),
        softmax_lse.stride(0), softmax_lse.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        AUX_MASK, AUX_MASK_SIZE, AUX_MASK.stride(0), AUX_MASK.stride(1),
        is_causal,global_window_size, local_window_size,
        num_q_heads, num_kv_heads, gqa_interleave, head_dim,
        CHUNK_SIZE, BLOCK_M, BLOCK_N, BLOCK_D,
    )

    return dq, dk, dv

class TTXSWAFunction(MojoSWAFunction):
    @staticmethod
    def forward(
        ctx, 
        q: torch.Tensor, # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor, # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor, # [bsz + 1]
        cu_seqlens_kv: torch.Tensor, # [bsz + 1]
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        o, softmax_lse = _sdpa_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, is_causal, local_window_size, global_window_size, sm_scale, gqa_interleave
        )
        ctx.save_for_backward(o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.is_causal = is_causal
        ctx.local_window_size = local_window_size
        ctx.global_window_size = global_window_size
        ctx.gqa_interleave = gqa_interleave
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                     None, None, None, None, None, None, None, None, None]:
        o, softmax_lse, q, k, v, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_kv = ctx.max_seqlen_kv
        is_causal = ctx.is_causal
        local_window_size = ctx.local_window_size
        global_window_size = ctx.global_window_size
        gqa_interleave = ctx.gqa_interleave

        dq, dk, dv = _sdpa_bwd(
            do, q, k, v, o, softmax_lse, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, is_causal, local_window_size, global_window_size, sm_scale, gqa_interleave
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_sparse_torch(
    q,
    k,
    v,
    cu_seqlens,
    gqa_interleave: bool =False,
    softmax_scale=None,
    local_window_size=0,
    global_window_size=0,
):

    T, H, Dq = q.shape
    Hk = k.shape[-2]
    gqa_ratio = H // Hk
    o = torch.zeros_like(q)
    if softmax_scale == None:
        softmax_scale = Dq ** (-0.5)

    bz = len(cu_seqlens) - 1

    for b in range(bz):
        for h in range(H):
            seq_start = cu_seqlens[b]
            seq_end = cu_seqlens[b + 1]
            seq_len = seq_end - seq_start
            if gqa_interleave:
                hk = h % Hk
            else:
                hk = h // gqa_ratio

            b_q = q[seq_start:seq_end, h, :].cpu().double()
            b_k = k[seq_start:seq_end, hk, :].cpu().double()
            b_v = v[seq_start:seq_end, hk, :].cpu().double()
            b_s = b_q @ b_k.T

            casual_mask = (
                torch.arange(0, seq_len)[:, None] >= torch.arange(0, seq_len)[None, :]
            )
            if local_window_size is not None or global_window_size is not None:
                local_window_mask = (
                    torch.arange(0, seq_len)[:, None]
                    <= torch.arange(0, seq_len)[None, :] + local_window_size
                ) if local_window_size is not None else False
                global_window_mask = (torch.arange(0, seq_len) < global_window_size)[
                    None, :
                ] if global_window_size is not None else False
                b_s_mask = casual_mask & (local_window_mask | global_window_mask)
            else:
                b_s_mask = casual_mask

            b_s = torch.where(b_s_mask.to(device=b_s.device), b_s, -float("inf"))
            b_s = b_s * softmax_scale
            b_s = b_s.softmax(dim=-1)

            b_o = b_s @ b_v
            o[seq_start:seq_end, h, :] = b_o.to(o.dtype).to(o.device)

    return o

def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    max_q_len: int,
    max_kv_prefix_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = "npu",
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (bsz,), dtype=torch.int32, device=device)
    if max_kv_prefix_len > 0:
        kv_prefix_lens = torch.randint(max_kv_prefix_len // 2, max_kv_prefix_len, (bsz,), dtype=torch.int32, device=device)
    else:
        kv_prefix_lens = torch.zeros(bsz, dtype=torch.int32, device=device)
    kv_lens = kv_prefix_lens + q_lens
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0)])
    cu_seqlens_kv = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), kv_lens.cumsum(0)])

    query = torch.randn(cu_seqlens_q[-1].item(), q_head_num, head_dim, dtype=dtype, device=device)
    key = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype, device=device)
    value = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype, device=device)

    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, key, value, cu_seqlens_q, cu_seqlens_kv


def test_swa_function():
    test_configs = [
        (1, 1, 1, False, 128, 256, 0, torch.float32),
        # (4, 4, 2, True, 128, 512, 0, torch.float32),
        # (4, 4, 2, False, 128, 256, 0, torch.bfloat16),
        # (4, 16, 4, False, 128, 4096, 0, torch.bfloat16),
    ]

    local_window = 1023
    global_window = 0
    for bsz, q_head_num, kv_head_num, gqa_interleave, head_dim, max_q_len, max_kv_prefix_len, dtype in test_configs:
        print(bsz, q_head_num, kv_head_num, gqa_interleave, head_dim, max_q_len, max_kv_prefix_len, dtype)
        scale = 1.0 / head_dim ** 0.5
        for i in range(5):
            query, key, value, cu_seqlens_q, cu_seqlens_kv = generate_test_data(bsz, q_head_num, kv_head_num, head_dim, max_q_len, max_kv_prefix_len, dtype)
            print(i, cu_seqlens_q, cu_seqlens_kv)
            q_ref = query.clone().detach().requires_grad_(True)
            k_ref = key.clone().detach().requires_grad_(True)
            v_ref = value.clone().detach().requires_grad_(True)

            q_mojo = query.clone().detach().requires_grad_(True)
            k_mojo = key.clone().detach().requires_grad_(True)
            v_mojo = value.clone().detach().requires_grad_(True)

            o_ref = flash_attn_sparse_torch(
                q_ref, k_ref, v_ref, cu_seqlens_q, gqa_interleave, scale, local_window, global_window
            )

            o_mojo = MojoSWAFunction._registry.get("ttx").apply(
                q_mojo, k_mojo, v_mojo, cu_seqlens_q, cu_seqlens_kv, max_q_len, max_kv_prefix_len, True, local_window, global_window, scale, gqa_interleave
            )

            assert_close(o_ref, o_mojo)
            grad_out = torch.randn_like(o_ref)
            o_ref.backward(grad_out)
            o_mojo.backward(grad_out)
            assert_close(q_ref.grad, q_mojo.grad)
            assert_close(k_ref.grad, k_mojo.grad)
            assert_close(v_ref.grad, v_mojo.grad)



def assert_close(
    results,
    refs,
):
    """
    Asserts that the results are close to the reference tensors within specified tolerances.

    Args:
        results (Union[torch.Tensor, Tuple[Any, ...]]): The calculated result tensor(s).
        refs (Union[torch.Tensor, Tuple[Any, ...]]): The reference/golden tensor(s).

    Raises:
        AssertionError: If shapes, dtypes, or values do not match within tolerance.
    """
    assert type(results) is type(refs)
    if isinstance(results, torch.Tensor) and isinstance(refs, torch.Tensor):
        results = tuple([results])
        refs = tuple([refs])

    for result, ref in zip(results, refs):
        if isinstance(result, torch.Tensor) and isinstance(ref, torch.Tensor):
            assert result.shape == ref.shape
            assert result.dtype == ref.dtype
            dtype = result.dtype
            if dtype == torch.bfloat16:
                max_atol = 0.1
                max_rtol = 0.05
                mean_atol = 0.01
                mean_rtol = 0.01
            elif dtype == torch.float16:
                max_atol = 2e-2
                max_rtol = 2e-2
                mean_atol = 2e-2
                mean_rtol = 2e-2
            elif dtype == torch.float32:
                max_atol = 6e-3
                max_rtol = 6e-3
                mean_atol = 1e-4
                mean_rtol = 1e-4
            else:
                logger.warning(f"dtype {dtype} is not supported.")
                assert False

            torch.testing.assert_close(result.to(torch.float32), ref.to(torch.float32), atol=max_atol, rtol=max_rtol)
            assert (
                torch.mean(torch.abs(ref - result)) < max_atol
                or torch.mean(torch.abs((ref - result) / (ref + mean_atol))) < mean_rtol
            )
        else:
            assert result == ref

if __name__ == "__main__":
    test_swa_function()