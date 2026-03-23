import torch
from mojo_opset.core.operator import MojoOperator
from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from typing import Optional, Tuple


def generate_swa_mask(
    q_seq_len: int,
    kv_seq_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
) -> torch.Tensor:
    kv_computed_len = kv_seq_len - q_seq_len
    causal_mask = (torch.arange(0, q_seq_len)[:, None] + kv_computed_len) >= torch.arange(0, kv_seq_len)[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (
                torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
            )
            if local_window_size is not None
            else False
        )
        global_window_mask = (
            (torch.arange(0, kv_seq_len) < global_window_size)[None, :] if global_window_size is not None else False
        )
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask

    return mask


class MojoSWA(MojoOperator):

    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        total_q_len, n_q_heads, head_dim = q.shape
        n_kv_heads = k.shape[1]
        if sm_scale is None:
            sm_scale = 1.0 / (head_dim**0.5)

        o = torch.empty_like(q)
        bsz = cu_seqlens_q.shape[0] - 1
        for i in range(bsz):
            q_i = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
            q_seq_len = q_i.shape[0]
            q_i = q_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

            k_i = k[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]]
            kv_seq_len = k_i.shape[0]
            k_i_T = k_i.permute(1, 2, 0)
            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]
            s_i = torch.bmm(q_i, k_i_T).float() * sm_scale  # -> [n_q_heads, q_seq_len, kv_seq_len]

            if is_causal:
                s_mask = generate_swa_mask(
                    q_seq_len,
                    kv_seq_len,
                    local_window_size,
                    global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, q_seq_len, 1]
            s_i = s_i - m_i  # -> [n_q_heads, q_seq_len, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, q_seq_len, 1]
            p_i = p_i.to(v.dtype)

            v_i = v[cu_seqlens_kv[i] : cu_seqlens_kv[i + 1]].permute(1, 0, 2)  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            o_i = torch.bmm(p_i, v_i).float()  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i / l_i  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i.permute(1, 0, 2)  # -> [q_seq_len, n_q_heads, head_dim]
            o[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = o_i.to(o.dtype)
        return o


class MojoPagedPrefillSWA(MojoOperator):

    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        v_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        kv_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        total_q_len, n_q_heads, head_dim = q.shape
        _, n_kv_heads, page_size, _ = k_cache.shape
        if sm_scale is None:
            sm_scale = 1.0 / (head_dim**0.5)

        o = torch.empty_like(q)
        bsz = cu_seqlens_q.shape[0] - 1
        for i in range(bsz):
            q_i = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
            q_seq_len = q_i.shape[0]
            q_i = q_i.permute(1, 0, 2)  # -> [n_q_heads, q_seq_len, head_dim]

            kv_seq_len = kv_lens[i].item()
            kv_blocks = (kv_seq_len + page_size - 1) // page_size
            k_i = k_cache[block_table[i, :kv_blocks]]  # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_i = k_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]
            k_i_T = k_i.permute(0, 2, 1)  # -> [n_kv_heads, head_dim, kv_seq_len]
            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]
            s_i = torch.bmm(q_i, k_i_T).float() * sm_scale  # -> [n_q_heads, q_seq_len, kv_seq_len]

            if is_causal:
                s_mask = generate_swa_mask(
                    q_seq_len,
                    kv_seq_len,
                    local_window_size,
                    global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, q_seq_len, 1]
            s_i = s_i - m_i  # -> [n_q_heads, q_seq_len, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, q_seq_len, 1]
            p_i = p_i.to(q.dtype)

            v_i = v_cache[block_table[i, :kv_blocks]]
            v_i = v_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[
                :, :kv_seq_len
            ]  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            o_i = torch.bmm(p_i, v_i).float()  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i / l_i  # -> [n_q_heads, q_seq_len, head_dim]
            o_i = o_i.permute(1, 0, 2)  # -> [q_seq_len, n_q_heads, head_dim]
            o[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = o_i.to(o.dtype)
        return o


class MojoPagedDecodeSWA(MojoOperator):

    def forward(
        self,
        q: torch.Tensor,  # [bsz, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        v_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        kv_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        bsz, n_q_heads, head_dim = q.shape
        _, n_kv_heads, page_size, _ = k_cache.shape
        if sm_scale is None:
            sm_scale = 1.0 / (head_dim**0.5)

        o = torch.empty_like(q)
        for i in range(bsz):
            q_i = q[i].unsqueeze(1) # -> [n_q_heads, 1, head_dim]

            kv_seq_len = kv_lens[i].item()
            kv_blocks = (kv_seq_len + page_size - 1) // page_size
            k_i = k_cache[block_table[i, :kv_blocks]]  # [kv_blocks, n_kv_heads, page_size, head_dim]
            k_i = k_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[:, :kv_seq_len]
            k_i_T = k_i.permute(0, 2, 1)  # -> [n_kv_heads, head_dim, kv_seq_len]
            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    k_i_T = k_i_T.repeat_interleave(
                        n_q_heads // n_kv_heads, dim=0
                    )  # -> [n_q_heads, head_dim, kv_seq_len]
            s_i = torch.bmm(q_i, k_i_T).float() * sm_scale  # -> [n_q_heads, 1, kv_seq_len]

            if is_causal:
                s_mask = generate_swa_mask(
                    1,
                    kv_seq_len,
                    local_window_size,
                    global_window_size,
                ).to(s_i.device)
                s_i = torch.where(s_mask, s_i, float("-inf"))
            m_i = torch.max(s_i, dim=-1, keepdim=True).values  # -> [n_q_heads, 1, 1]
            s_i = s_i - m_i  # -> [n_q_heads, 1, kv_seq_len]
            p_i = torch.exp(s_i)
            l_i = torch.sum(p_i, dim=-1, keepdim=True)  # -> [n_q_heads, 1, 1]
            p_i = p_i.to(q.dtype)

            v_i = v_cache[block_table[i, :kv_blocks]]
            v_i = v_i.permute(1, 0, 2, 3).reshape(n_kv_heads, kv_blocks * page_size, head_dim)[
                :, :kv_seq_len
            ]  # -> [n_kv_heads, kv_seq_len, head_dim]
            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    v_i = v_i.repeat((n_q_heads // n_kv_heads, 1, 1))
                else:
                    v_i = v_i.repeat_interleave(n_q_heads // n_kv_heads, dim=0)  # -> [n_q_heads, kv_seq_len, head_dim]
            o_i = torch.bmm(p_i, v_i).float()  # -> [n_q_heads, 1, head_dim]
            o_i = o_i / l_i  # -> [n_q_heads, 1, head_dim]
            o_i = o_i.squeeze(1)  # -> [n_q_heads, head_dim]
            o[i] = o_i.to(o.dtype)
        return o


AUX_MASK_SIZE = 256
AUX_MASK = None


def get_aux_mask():
    global AUX_MASK
    global AUX_MASK_SIZE
    if AUX_MASK is None:
        AUX_MASK = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).triu().bool(),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).tril().bool(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        ).npu()
    return AUX_MASK_SIZE, AUX_MASK


import triton
import triton.language as tl


@triton.jit
def _swa_split_blocks(q_block_start_id, q_block_len, kv_seq_len, BLOCK_SIZE_N, IS_CAUSAL, GLOBAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE):
    if not IS_CAUSAL:
        return 0, 0, tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

    num_total_blocks = tl.cdiv(q_block_start_id + q_block_len, BLOCK_SIZE_N)
    if GLOBAL_WINDOW_SIZE is None and LOCAL_WINDOW_SIZE is None:
        return 0, 0, num_total_blocks
    
    if GLOBAL_WINDOW_SIZE is not None:
        num_global_window_blocks = min(tl.cdiv(GLOBAL_WINDOW_SIZE, BLOCK_SIZE_N), num_total_blocks)
    else:
        num_global_window_blocks = 0
    
    if LOCAL_WINDOW_SIZE is not None:
        local_window_start_id = max(q_block_start_id - LOCAL_WINDOW_SIZE, 0)
        local_window_start_block = local_window_start_id // BLOCK_SIZE_N
    else:
        local_window_start_block = num_total_blocks
    
    non_global_window_start_block = max(num_global_window_blocks, local_window_start_block)
    
    return num_global_window_blocks, non_global_window_start_block, num_total_blocks


@triton.jit
def gen_mask_n_right_bound(mask_10_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, n_start, right):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] < right
    offset = min(max(n_start - right, -mask_size), 0)
    mask = tl.load(
        mask_10_ptr
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_n_left_bound(mask_01_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, n_start, left):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= left
    offset = min(max(n_start - left, -mask_size), 0)
    mask = tl.load(
        mask_01_ptr
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_m_right_bound(mask_10t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, right):
    # tl.arange(m_start, m_start + M_BLOCK)[:, None] < right
    offset = min(max(m_start - right, -mask_size), 0)
    mask = tl.load(
        mask_10t_ptr
        + (offset + tl.arange(0, M_BLOCK)[:, None]) * mask_stride_m
        + tl.arange(0, N_BLOCK)[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_m_left_bound(mask_01t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, left):
    # tl.arange(m_start, m_start + M_BLOCK)[:, None] >= left
    offset = min(max(m_start - left, -mask_size), 0)
    mask = tl.load(
        mask_01t_ptr
        + (offset + tl.arange(0, M_BLOCK)[:, None]) * mask_stride_m
        + tl.arange(0, N_BLOCK)[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_tril(mask_ptr_tril, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] <= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_tril
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_triu(mask_ptr_triu, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    len_offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_triu
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (len_offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def _sdpa_acc_fwd_MxN(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask,
    qk_scale,
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
def _sdpa_infer_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    bsz,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
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

    cu_q_chunks = 0
    for b_id in range(bsz):
        q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
        kv_start = tl.load(cu_seqlens_kv_ptr + b_id).to(tl.int32)
        kv_end = tl.load(cu_seqlens_kv_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)

        prev_q_tasks = cu_q_chunks * NUM_Q_HEADS
        cu_q_chunks += num_q_chunks
        new_q_tasks = num_q_chunks * NUM_Q_HEADS
        for q_task_id in range((prev_q_tasks + pid) % n_programs, new_q_tasks, n_programs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            # q_block_ptr = tl.make_block_ptr(
            #     base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
            #     shape=(q_seq_len, HEAD_DIM),
            #     strides=(stride_qt, stride_qd),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_M, BLOCK_D),
            #     order=(1, 0),
            # )
            # o_block_ptr = tl.make_block_ptr(
            #     base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
            #     shape=(q_seq_len, HEAD_DIM),
            #     strides=(stride_ot, stride_od),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_M, BLOCK_D),
            #     order=(1, 0),
            # )
            # k_block_ptr = tl.make_block_ptr(
            #     base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
            #     shape=(kv_seq_len, HEAD_DIM),
            #     strides=(stride_kt, stride_kd),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_N, BLOCK_D),
            #     order=(1, 0),
            # )
            # v_block_ptr = tl.make_block_ptr(
            #     base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
            #     shape=(kv_seq_len, HEAD_DIM),
            #     strides=(stride_vt, stride_vd),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_N, BLOCK_D),
            #     order=(1, 0),
            # )
            q_mask = gen_mask_m_right_bound(
                aux_mask_ptr_10t,
                aux_mask_size,
                stride_mask_m,
                stride_mask_n,
                BLOCK_M,
                BLOCK_N,
                q_block_id * BLOCK_M,
                q_seq_len,
            )
            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start

            # cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            for kv_block_id in range(num_global_window_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_mask = gen_mask_n_right_bound(
                    aux_mask_ptr_10,
                    aux_mask_size,
                    stride_mask_m,
                    stride_mask_n,
                    BLOCK_M,
                    BLOCK_N,
                    kv_block_start,
                    kv_seq_len,
                )
                if IS_CAUSAL:
                    # actually, it must be true for global window blocks
                    mask_gw = gen_mask_n_right_bound(
                        aux_mask_ptr_10,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        kv_block_start,
                        GLOBAL_WINDOW,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu,
                            aux_mask_size,
                            stride_mask_m,
                            stride_mask_n,
                            BLOCK_M,
                            BLOCK_N,
                            q_block_start + kv_computed_len,
                            kv_block_start + LOCAL_WINDOW,
                        )
                        mask_gw = mask_gw | mask_sw
                    mask_causal = gen_mask_tril(
                        aux_mask_ptr_tril,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                    )
                    mask_causal = mask_gw & mask_causal
                    mask = mask_causal & q_mask & kv_mask
                else:
                    mask = q_mask & kv_mask
                # cur_k_block_ptr = tl.advance(k_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                # cur_v_block_ptr = tl.advance(v_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_acc_fwd_MxN(
                    acc,
                    l_i,
                    m_i,
                    cur_q_block,
                    cur_k_block_ptr,
                    cur_v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_mask = gen_mask_n_right_bound(
                    aux_mask_ptr_10,
                    aux_mask_size,
                    stride_mask_m,
                    stride_mask_n,
                    BLOCK_M,
                    BLOCK_N,
                    kv_block_start,
                    kv_seq_len,
                )
                if IS_CAUSAL:
                    mask_causal = gen_mask_tril(
                        aux_mask_ptr_tril,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu,
                            aux_mask_size,
                            stride_mask_m,
                            stride_mask_n,
                            BLOCK_M,
                            BLOCK_N,
                            q_block_start + kv_computed_len,
                            kv_block_start + LOCAL_WINDOW,
                        )
                        mask_causal = mask_causal & mask_sw

                    mask = mask_causal & q_mask & kv_mask
                else:
                    mask = q_mask & kv_mask
                # cur_k_block_ptr = tl.advance(k_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                # cur_v_block_ptr = tl.advance(v_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _sdpa_acc_fwd_MxN(
                    acc,
                    l_i,
                    m_i,
                    cur_q_block,
                    cur_k_block_ptr,
                    cur_v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            # cur_o_block_ptr = tl.advance(o_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            accumulator = acc / l_i[:, None]
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def swa_ttx_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,  # [bsz + 1]
    cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    sm_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_size, mask = get_aux_mask()
    bsz = cu_seqlens_q.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    tot_kv_toks, num_kv_heads, _ = k.shape
    o = torch.zeros_like(q)

    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 128
    BLOCK_D = head_dim

    cube_num = get_num_cores("cube")
    grid = (cube_num,)

    _sdpa_infer_kernel[grid](
        o,
        q,
        k,
        v,
        bsz,
        cu_seqlens_q,
        cu_seqlens_kv,
        sm_scale,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mask,
        mask_size,
        mask.stride(0),
        mask.stride(1),
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
    )
    return o


class TTXSWA(MojoSWA):
    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        cu_seqlens_kv: torch.Tensor,  # [bsz + 1]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:

        o = swa_ttx_infer(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            is_causal,
            local_window_size,
            global_window_size,
            sm_scale,
            gqa_interleave,
        )
        return o


@triton.jit
def _paged_prefill_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    bsz,
    cu_seqlens_q_ptr,
    kv_lens_ptr,
    block_table_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_block_table_b,
    stride_block_table_p,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    tl.static_assert(PAGE_SIZE == BLOCK_N, "Currently only support PAGE_SIZE == BLOCK_SIZE_N")
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

    cu_q_chunks = 0
    for b_id in range(bsz):
        q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
        kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)

        prev_q_tasks = cu_q_chunks * NUM_Q_HEADS
        cu_q_chunks += num_q_chunks
        new_q_tasks = num_q_chunks * NUM_Q_HEADS
        for q_task_id in range((prev_q_tasks + pid) % n_programs, new_q_tasks, n_programs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            # q_block_ptr = tl.make_block_ptr(
            #     base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
            #     shape=(q_seq_len, HEAD_DIM),
            #     strides=(stride_qt, stride_qd),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_M, BLOCK_D),
            #     order=(1, 0),
            # )
            # o_block_ptr = tl.make_block_ptr(
            #     base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
            #     shape=(q_seq_len, HEAD_DIM),
            #     strides=(stride_ot, stride_od),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_M, BLOCK_D),
            #     order=(1, 0),
            # )
            q_mask = gen_mask_m_right_bound(
                aux_mask_ptr_10t,
                aux_mask_size,
                stride_mask_m,
                stride_mask_n,
                BLOCK_M,
                BLOCK_N,
                q_block_id * BLOCK_M,
                q_seq_len,
            )
            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            # cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                PAGE_SIZE,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            for logical_page_id in range(num_global_window_blocks):
                kv_block_start = logical_page_id * PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )
                kv_mask = gen_mask_n_right_bound(
                    aux_mask_ptr_10,
                    aux_mask_size,
                    stride_mask_m,
                    stride_mask_n,
                    BLOCK_M,
                    BLOCK_N,
                    kv_block_start,
                    kv_seq_len,
                )
                if IS_CAUSAL:
                    # actually, it must be true for global window blocks
                    mask_gw = gen_mask_n_right_bound(
                        aux_mask_ptr_10,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        kv_block_start,
                        GLOBAL_WINDOW,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu,
                            aux_mask_size,
                            stride_mask_m,
                            stride_mask_n,
                            BLOCK_M,
                            BLOCK_N,
                            q_block_start + kv_computed_len,
                            kv_block_start + LOCAL_WINDOW,
                        )
                        mask_gw = mask_gw | mask_sw
                    mask_causal = gen_mask_tril(
                        aux_mask_ptr_tril,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                    )
                    mask_causal = mask_gw & mask_causal
                    mask = mask_causal & q_mask & kv_mask
                else:
                    mask = q_mask & kv_mask
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + physical_page_id * stride_kp + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + physical_page_id * stride_vp + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _sdpa_acc_fwd_MxN(
                    acc,
                    l_i,
                    m_i,
                    cur_q_block,
                    cur_k_block_ptr,
                    cur_v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            for logical_page_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = logical_page_id * PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )
                kv_mask = gen_mask_n_right_bound(
                    aux_mask_ptr_10,
                    aux_mask_size,
                    stride_mask_m,
                    stride_mask_n,
                    BLOCK_M,
                    BLOCK_N,
                    kv_block_start,
                    kv_seq_len,
                )
                if IS_CAUSAL:
                    mask_causal = gen_mask_tril(
                        aux_mask_ptr_tril,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu,
                            aux_mask_size,
                            stride_mask_m,
                            stride_mask_n,
                            BLOCK_M,
                            BLOCK_N,
                            q_block_start + kv_computed_len,
                            kv_block_start + LOCAL_WINDOW,
                        )
                        mask_causal = mask_causal & mask_sw
                    mask = mask_causal & q_mask & kv_mask
                else:
                    mask = q_mask & kv_mask

                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + physical_page_id * stride_kp + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + physical_page_id * stride_vp + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _sdpa_acc_fwd_MxN(
                    acc,
                    l_i,
                    m_i,
                    cur_q_block,
                    cur_k_block_ptr,
                    cur_v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            # cur_o_block_ptr = tl.advance(o_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            accumulator = acc / l_i[:, None]
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def swa_ttx_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,  # [bsz + 1]
    kvlens: torch.Tensor,  # [bsz + 1]
    block_table: torch.Tensor,  # [bsz, num_kv_blocks]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    sm_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_size, mask = get_aux_mask()
    bsz = cu_seqlens_q.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = k_cache.shape
    o = torch.zeros_like(q)
    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 64
        BLOCK_N = 128
    assert page_size == BLOCK_N

    BLOCK_D = head_dim
    cube_num = get_num_cores("cube")

    grid = (cube_num,)

    _paged_prefill_kernel[grid](
        o,
        q,
        k_cache,
        v_cache,
        bsz,
        cu_seqlens_q,
        kvlens,
        block_table,
        sm_scale,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        mask,
        mask_size,
        mask.stride(0),
        mask.stride(1),
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        page_size,
    )
    return o


class TTXPagedPrefillSWA(MojoPagedPrefillSWA):
    def forward(
        self,
        q: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        v_cache: torch.Tensor,  # [total_k_len, n_kv_heads, head_dim]
        cu_seqlens_q: torch.Tensor,  # [bsz + 1]
        kvlens: torch.Tensor,  # [bsz + 1]
        block_table: torch.Tensor,  # [bsz, num_kv_blocks]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:

        o = swa_ttx_paged_prefill(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            kvlens,
            block_table,
            is_causal,
            local_window_size,
            global_window_size,
            sm_scale,
            gqa_interleave,
        )
        return o

@triton.jit
def _sdpa_acc_fwd_1xN(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return acc_ptr, l_i, m_i
    # -- Compute qk ----
                    
    # Load K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 0))  # Scaled max
    qk = qk - m_ij  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, axis=0)
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha
    acc_ptr += tl.sum((p_cast[:, None] * v).to(tl.float32), axis=0)

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
    return acc_ptr, l_i, m_i



@triton.jit
def _paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_TOTAL_BLOCKS,
    MAX_NUM_BLOCKS_PER_SEQ,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    sm_scale,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be less than BLOCK_SIZE_D")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_Q_HEADS

    for q_task_id in range(pid, num_tasks, n_progs):
        q_head_id = q_task_id % NUM_Q_HEADS
        b_id = q_task_id // NUM_Q_HEADS
        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        kv_seq_len = tl.load(seqlens_ptr + b_id)


        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_id * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs, mask = offs_d < HEAD_DIM, other = 0.0)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

        num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
            kv_seq_len - 1,
            1,
            kv_seq_len,
            PAGE_SIZE,
            True,
            GLOBAL_WINDOW,
            LOCAL_WINDOW,
        )
        
        tl.static_assert(PAGE_SIZE == BLOCK_SIZE_N, "PAGE_SIZE should be equal to BLOCK_SIZE_N")

        for logical_page_id in range(num_global_window_blocks):
            kv_block_start = logical_page_id * PAGE_SIZE
            kv_block_end = min(kv_block_start + PAGE_SIZE, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            gw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N)) < GLOBAL_WINDOW
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N) + LOCAL_WINDOW) >= (kv_seq_len - 1)
                gw_mask = gw_mask | sw_mask
            kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len
            mask = gw_mask & kv_mask
            
            acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                acc, 
                l_i, 
                m_i, 
                q, 
                k_block_ptr, 
                v_block_ptr, 
                mask, 
                sm_scale, 
                HEAD_DIM, 
                BLOCK_SIZE_D, 
                BLOCK_SIZE_N, 
                BLOCK_SIZE_D, 
                v_cache_ptr.dtype.element_ty == tl.float8e5,
            )

        for logical_page_id in range(non_global_window_start_block, num_total_blocks):
            kv_block_start = logical_page_id * PAGE_SIZE
            kv_block_end = min(kv_block_start + PAGE_SIZE, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head,
                shape=(PAGE_SIZE, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            
            kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N) + LOCAL_WINDOW) >= (kv_seq_len - 1)
                mask = kv_mask & sw_mask
            else:
                mask = kv_mask
            
            acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                acc,
                l_i, 
                m_i, 
                q, 
                k_block_ptr, 
                v_block_ptr, 
                mask, 
                sm_scale, 
                HEAD_DIM,
                BLOCK_SIZE_D,
                BLOCK_SIZE_N, 
                BLOCK_SIZE_D,
                v_cache_ptr.dtype.element_ty == tl.float8e5,
            )

        acc = acc / l_i

        o_ptrs = o_ptr + b_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_d < HEAD_DIM)


def swa_ttx_paged_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    gqa_interleave: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, block_size, head_dim_cache = key_cache.shape

    assert block_size <= 128, f"temp: only support block_size <= 128, but got {block_size}"
    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    o = torch.empty_like(q)
    
    num_vectors = get_num_cores("vector")
    grid = (num_vectors, )
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    # Note(chenyifan): 
    #   under swa, the kv workload is rather evenly across diffrent queries,
    #   so we have low necessity to apply split-kv strategy             

    _paged_decode_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        block_tables,
        batch_size,
        num_total_blocks,
        max_num_blocks_per_seq,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        sm_scale,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        block_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=block_size,
        multibuffer=False,
    )
    return o



class TTXPagedDecodeSWA(MojoPagedDecodeSWA):

    def forward(
        self,
        q: torch.Tensor,  # [bsz, n_q_heads, head_dim]
        k_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        v_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        kv_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        is_causal: bool = True,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        # Note: if is_causal = False, local_window_size and global_window_size are not used.

        o = swa_ttx_paged_decode(
            q,
            k_cache,
            v_cache,
            kv_lens,
            block_table,
            local_window_size,
            global_window_size,
            gqa_interleave,
            sm_scale,
        )
        
        return o


def flash_attn_sparse_torch(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    gqa_interleave: bool = False,
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

    bz = len(cu_seqlens_q) - 1

    for b in range(bz):
        for h in range(H):
            q_seq_start = cu_seqlens_q[b].item()
            q_seq_end = cu_seqlens_q[b + 1].item()
            q_seq_len = q_seq_end - q_seq_start
            kv_seq_start = cu_seqlens_kv[b].item()
            kv_seq_end = cu_seqlens_kv[b + 1].item()
            kv_seq_len = kv_seq_end - kv_seq_start
            kv_computed_len = kv_seq_len - q_seq_len

            if gqa_interleave:
                hk = h % Hk
            else:
                hk = h // gqa_ratio

            b_q = q[q_seq_start:q_seq_end, h, :].cpu().double()
            b_k = k[kv_seq_start:kv_seq_end, hk, :].cpu().double()
            b_v = v[kv_seq_start:kv_seq_end, hk, :].cpu().double()
            b_s = b_q @ b_k.T

            casual_mask = (
                torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                >= torch.arange(0, kv_seq_len)[None, :]
            )
            if local_window_size is not None or global_window_size is not None:
                local_window_mask = (
                    (
                        torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                        <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
                    )
                    if local_window_size is not None
                    else False
                )
                global_window_mask = (
                    (torch.arange(0, kv_seq_len) < global_window_size)[None, :]
                    if global_window_size is not None
                    else False
                )
                b_s_mask = casual_mask & (local_window_mask | global_window_mask)
            else:
                b_s_mask = casual_mask

            b_s = torch.where(b_s_mask.to(device=b_s.device), b_s, -float("inf"))
            b_s = b_s * softmax_scale
            b_s = b_s.softmax(dim=-1)

            b_o = b_s @ b_v
            o[q_seq_start:q_seq_end, h, :] = b_o.to(o.dtype).to(o.device)

    return o


def paged_prefill_attn_sparse_torch(
    q,
    k_cache,
    v_cache,
    cu_seqlens,
    kvlens,
    block_table,
    gqa_interleave: bool = False,
    softmax_scale=None,
    local_window_size=0,
    global_window_size=0,
):

    T, H, Dq = q.shape
    _, Hk, P, _ = k_cache.shape
    gqa_ratio = H // Hk
    o = torch.zeros_like(q)
    if softmax_scale == None:
        softmax_scale = Dq ** (-0.5)

    bz = cu_seqlens.shape[0] - 1

    for b in range(bz):
        for h in range(H):
            seq_start = cu_seqlens[b].item()
            seq_end = cu_seqlens[b + 1].item()
            seq_len = seq_end - seq_start
            if gqa_interleave:
                hk = h % Hk
            else:
                hk = h // gqa_ratio

            b_q = q[seq_start:seq_end, h, :].cpu().double()
            kv_len = kvlens[b].item()
            b_pages = block_table[b, : (kv_len + P - 1) // P]
            b_k = k_cache[b_pages, hk].reshape(-1, Dq)[:kv_len].cpu().double()
            b_v = v_cache[b_pages, hk].reshape(-1, Dq)[:kv_len].cpu().double()
            b_s = b_q @ b_k.T

            casual_mask = torch.arange(kv_len - seq_len, kv_len)[:, None] >= torch.arange(0, kv_len)[None, :]
            if local_window_size is not None or global_window_size is not None:
                local_window_mask = (
                    (
                        torch.arange(kv_len - seq_len, kv_len)[:, None]
                        <= torch.arange(0, kv_len)[None, :] + local_window_size
                    )
                    if local_window_size is not None
                    else False
                )
                global_window_mask = (
                    (torch.arange(0, kv_len) < global_window_size)[None, :] if global_window_size is not None else False
                )
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
    gqa_interleave: bool,
    head_dim: int,
    max_q_len: int,
    max_kv_prefix_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = "npu",
    random_seed: Optional[int] = None,
):
    if random_seed is not None:
        set_seed(random_seed)
    q_lens = torch.randint(max(max_q_len // 2, 1), max_q_len + 1, (bsz,), dtype=torch.int32, device=device)
    if max_kv_prefix_len > 0:
        kv_prefix_lens = torch.randint(
            max_kv_prefix_len // 2, max_kv_prefix_len, (bsz,), dtype=torch.int32, device=device
        )
    else:
        kv_prefix_lens = torch.zeros(bsz, dtype=torch.int32, device=device)
    kv_lens = kv_prefix_lens + q_lens
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0)])
    cu_seqlens_kv = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), kv_lens.cumsum(0)])

    query = torch.randn(cu_seqlens_q[-1].item(), q_head_num, head_dim, dtype=dtype, device=device)
    key = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype, device=device)
    value = torch.randn(cu_seqlens_kv[-1].item(), kv_head_num, head_dim, dtype=dtype, device=device)

    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, gqa_interleave, key, value, cu_seqlens_q, cu_seqlens_kv


def generate_paged_prefill_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    gqa_interleave: bool,
    head_dim: int,
    max_q_len: int,
    max_kv_prefix_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = "npu",
    random_seed: Optional[int] = None,
):
    if random_seed is not None:
        set_seed(random_seed)
    if dtype == torch.float32:
        page_size = 64
    else:
        page_size = 128
    q_lens = torch.randint(max(max_q_len // 2, 1), max_q_len + 1, (bsz,), dtype=torch.int32, device=device)
    if max_kv_prefix_len > 0:
        kv_prefix_lens = torch.randint(
            max_kv_prefix_len // 2, max_kv_prefix_len, (bsz,), dtype=torch.int32, device=device
        )
    else:
        kv_prefix_lens = torch.zeros(bsz, dtype=torch.int32, device=device)
    kv_lens = kv_prefix_lens + q_lens
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0)])

    max_num_pages = (max_kv_prefix_len + max_q_len + page_size - 1) // page_size * bsz * 2

    allocated_pages = (kv_lens + (page_size - 1)) // page_size
    page_idxs = torch.randperm(allocated_pages.sum().item(), device=device)
    cu_alloc_pages = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), allocated_pages.cumsum(0)])
    block_table = torch.zeros(bsz, max_num_pages, dtype=torch.int32, device=device)
    for i in range(bsz):
        block_table[i, : allocated_pages[i]] = page_idxs[cu_alloc_pages[i] : cu_alloc_pages[i + 1]]

    query = torch.randn(cu_seqlens_q[-1].item(), q_head_num, head_dim, dtype=dtype, device=device)
    key_cache = torch.randn(max_num_pages, kv_head_num, page_size, head_dim, dtype=dtype, device=device)
    value_cache = torch.randn(max_num_pages, kv_head_num, page_size, head_dim, dtype=dtype, device=device)

    # blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, gqa_interleave, key_cache, value_cache, cu_seqlens_q, kv_lens, block_table


def generate_paged_decode_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    gqa_interleave: bool,
    head_dim: int,
    max_kv_len: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = "npu",
    random_seed: Optional[int] = None,
):
    query, gqa_interleave, key_cache, value_cache, cu_seqlens_q, kv_lens, block_table = generate_paged_prefill_test_data(
        bsz, q_head_num, kv_head_num, gqa_interleave, head_dim, 1, max_kv_len-1, dtype, device, random_seed,
    )
    torch.testing.assert_close(cu_seqlens_q.to(torch.int32), torch.arange(bsz+1, dtype=torch.int32, device=device))
    return query, gqa_interleave, key_cache, value_cache, kv_lens, block_table


@torch.no_grad
def test_swa_function(query, gqa_interleave, key, value, cu_seqlens_q, cu_seqlens_kv, profiler=None):
    import datetime

    local_window = 1023
    global_window = 4
    head_dim = query.shape[-1]
    scale = 1.0 / head_dim**0.5

    q_mojo = query
    k_mojo = key
    v_mojo = value
    torch.npu.synchronize()
    time = datetime.datetime.now()
    o_mojo = MojoSWA._registry.get("ttx")()(
        q_mojo,
        k_mojo,
        v_mojo,
        cu_seqlens_q,
        cu_seqlens_kv,
        True,
        local_window,
        global_window,
        scale,
        gqa_interleave,
    )

    elapsed_time = datetime.datetime.now() - time

    if profiler is not None:
        profiler.step()
    else:
        q_ref = query.clone()
        k_ref = key.clone()
        v_ref = value.clone()
        o_ref = flash_attn_sparse_torch(
            q_ref, k_ref, v_ref, cu_seqlens_q, cu_seqlens_kv, gqa_interleave, scale, local_window, global_window
        )
        assert_close(o_ref, o_mojo)
    return elapsed_time

@torch.no_grad
def test_paged_prefill_swa_function(query, gqa_interleave, key, value, cu_seqlens_q, kvlens, block_table, profiler=None):
    import datetime

    local_window = 1023
    global_window = 4
    head_dim = query.shape[-1]
    scale = 1.0 / head_dim**0.5
    q_mojo = query
    k_mojo = key
    v_mojo = value
    torch.npu.synchronize()
    time = datetime.datetime.now()
    o_mojo = MojoPagedPrefillSWA._registry.get("ttx")()(
        q_mojo,
        k_mojo,
        v_mojo,
        cu_seqlens_q,
        kvlens,
        block_table,
        True,
        local_window,
        global_window,
        scale,
        gqa_interleave,
    )
    torch.npu.synchronize()
    elapsed_time = datetime.datetime.now() - time

    if profiler is not None:
        profiler.step()
    else:
        q_ref = query.clone()
        k_ref = key.clone()
        v_ref = value.clone()
        o_ref = paged_prefill_attn_sparse_torch(
            q_ref,
            k_ref,
            v_ref,
            cu_seqlens_q,
            kvlens,
            block_table,
            gqa_interleave,
            scale,
            local_window,
            global_window,
        )
        assert_close(o_ref, o_mojo)
    return elapsed_time

@torch.no_grad
def test_paged_decode_swa_function(query, gqa_interleave, key, value, kvlens, block_table, profiler=None):
    import datetime

    head_dim = query.shape[-1]
    local_window = 1023
    global_window = 4
    scale = 1.0 / head_dim**0.5

    q_mojo = query
    k_mojo = key
    v_mojo = value
    torch.npu.synchronize()
    time = datetime.datetime.now()
    o_mojo = MojoPagedDecodeSWA._registry.get("ttx")()(
        q_mojo,
        k_mojo,
        v_mojo,
        kvlens,
        block_table,
        True,
        local_window,
        global_window,
        scale,
        gqa_interleave,
    )
    torch.npu.synchronize()
    elapsed_time = datetime.datetime.now() - time

    if profiler is not None:
        profiler.step()
    else:
        q_ref = query.clone()
        k_ref = key.clone()
        v_ref = value.clone()
        bsz = block_table.shape[0]
        o_ref = paged_prefill_attn_sparse_torch(
            q_ref,
            k_ref,
            v_ref,
            torch.arange(bsz+1, dtype=torch.int32),
            kvlens,
            block_table,
            gqa_interleave,
            scale,
            local_window,
            global_window,
        )
        assert_close(o_ref, o_mojo)
    return elapsed_time

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    import numpy

    numpy.random.seed(seed)

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
    test_func_map = {
        "infer": test_swa_function,
        "prefill": test_paged_prefill_swa_function,
        "decode": test_paged_decode_swa_function,
    }

    generate_test_data_func_map = {
        "infer": generate_test_data,
        "prefill": generate_paged_prefill_test_data,
        "decode": generate_paged_decode_test_data,
    }

    test_configs_map = {
        "infer": [
            # (1, 1, 1, False, 128, 256, 0, torch.float32),
            # (4, 4, 2, True, 128, 512, 0, torch.float32),
            # (4, 4, 2, False, 128, 256, 0, torch.bfloat16),
            (4, 16, 4, False, 128, 1024, 8192, torch.bfloat16),
        ],
        "prefill":[
            # (1, 1, 1, False, 128, 256, 0, torch.float32),
            # (4, 4, 2, True, 128, 512, 1024, torch.float32),
            # (4, 4, 2, False, 128, 256, 0, torch.bfloat16),
            (4, 16, 4, False, 128, 1024, 8192, torch.bfloat16),
        ],
        "decode": [
            # (1, 1, 1, False, 128, 256, torch.float32),
            # (4, 4, 2, True, 128, 1024, torch.float32),
            # (4, 4, 2, False, 128, 256, torch.bfloat16),
            (4, 16, 4, False, 128, 8192, torch.bfloat16),
        ],
    }
    
    import sys
    test_func = test_func_map[sys.argv[1]]
    generate_test_data_func = generate_test_data_func_map[sys.argv[1]]
    test_configs = test_configs_map[sys.argv[1]]

    for test_config in test_configs:
        print(*test_config)

        test_inputs = generate_test_data_func(
            *test_config, random_seed=42
        )

        e2e_times = []
        total_runs = 10
        for i in range(total_runs):
            e2e_times.append(test_func(*test_inputs))
            # torch.distributed.barrier()
        print("Avg E2E time:", sum((t.microseconds / 1000) for t in e2e_times[1:]) / (total_runs - 1), "ms")

        import torch_npu

        profiling_dir = "./npu_profiling"
        active = 5
        # 添加Profiling采集基础配置参数，详细参数介绍可参考下文的参数说明
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            l2_cache=False,
            data_simplification=False,
        )

        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=active, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(total_runs):
                # 启动性能数据采集
                test_func(*test_inputs, prof)
        
        import os
        import csv

        try:
            kernel_profiling_path = max(
                [
                    os.path.join(profiling_dir, d)
                    for d in os.listdir(profiling_dir)
                    if os.path.isdir(os.path.join(profiling_dir, d))
                ],
                key=os.path.getmtime,
            )
            csv_file_path = os.path.join(kernel_profiling_path, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")

            if not os.path.exists(csv_file_path):
                raise ValueError(f"File not found: {csv_file_path}")

        except Exception as e:
            raise ValueError(f"Failed to get Profiling folder name: {e}")

        total_avg_time_us = 0.0

        with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                avg_time = float(row["Total Time(us)"])
                total_avg_time_us += avg_time

        print("Avg device time:", total_avg_time_us / active, "us")