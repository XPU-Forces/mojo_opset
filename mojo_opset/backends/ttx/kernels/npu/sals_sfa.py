from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


def _expand_indices_by_group_union(
    indices_flat: torch.Tensor,
    seq_len_flat: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match xpu_gpt SALS semantics: each Q group shares union(KV-head blocks)."""
    G = int(indices_flat.shape[0])
    K = int(indices_flat.shape[2]) if indices_flat.dim() == 3 else 0
    unions = []
    max_union = 0
    for group in range(G):
        s_count = int(seq_len_flat[group].item()) if seq_len_flat.numel() > group else 0
        if s_count <= 0 or K <= 0:
            union = indices_flat.new_empty((0,), dtype=torch.int32)
        else:
            blocks = indices_flat[group, :, : min(s_count, K)].reshape(-1)
            blocks = blocks[blocks >= 0]
            union = torch.unique(blocks, sorted=True).to(torch.int32) if blocks.numel() > 0 else blocks.to(torch.int32)
        unions.append(union)
        max_union = max(max_union, int(union.numel()))

    out_k = max(max_union, 1)
    union_indices = torch.full(
        (G, num_kv_heads, out_k),
        -1,
        dtype=torch.int32,
        device=indices_flat.device,
    )
    union_seq_len = torch.zeros((G,), dtype=torch.int32, device=seq_len_flat.device)
    for group, union in enumerate(unions):
        keep = int(union.numel())
        union_seq_len[group] = keep
        if keep > 0:
            union_indices[group, :, :keep] = union.view(1, keep).expand(num_kv_heads, keep)
    return union_indices, union_seq_len


# ---- Cube helpers (proven patterns from original kernel) ----

@triton.jit
def cube_qkt_sfa(
    Q_ptr,
    Wksp_K_ptr,
    stride_q_t, stride_q_h, stride_q_d,
    stride_wk_slot, stride_wk_kv, stride_wk_d,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    offs_q,
    offs_kv,
    q_base,
    wk_base,
    cur_kv,
):
    """Q @ K^T → [BLOCK_Q, BLOCK_KV], D-tiled dot product."""
    loop_times: tl.constexpr = (head_dim + BLOCK_D - 1) // BLOCK_D
    mask_kv = offs_kv < cur_kv
    acc = tl.zeros([BLOCK_Q, BLOCK_KV], dtype=tl.float32)
    for d in range(loop_times):
        offs_d = d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        q = tl.load(
            Q_ptr + q_base + offs_q[:, None] * stride_q_t + offs_d[None, :] * stride_q_d,
            mask=mask_d[None, :],
        ).to(tl.float32)
        k = tl.load(
            Wksp_K_ptr + wk_base + offs_kv[:, None] * stride_wk_kv + offs_d[None, :] * stride_wk_d,
            mask=mask_kv[:, None] & mask_d[None, :],
        )
        acc += tl.dot(q, k.trans())
    return acc


@triton.jit
def cube_pv_sfa(
    probs,
    Wksp_V_ptr,
    stride_wv_slot, stride_wv_kv, stride_wv_d,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    offs_kv,
    wv_base,
    cur_kv,
):
    """probs @ V → [BLOCK_Q, head_dim]."""
    mask_kv = offs_kv < cur_kv
    offs_d = tl.arange(0, head_dim)
    v = tl.load(
        Wksp_V_ptr + wv_base + offs_kv[:, None] * stride_wv_kv + offs_d[None, :] * stride_wv_d,
        mask=mask_kv[:, None],
    )
    return tl.dot(probs, v)


# ---- Gather KV from paged cache to workspace ----

@triton.jit
def _gather_kv_to_wksp(
    Wksp_K_ptr, Wksp_V_ptr, Wksp_pos_ptr,
    K_cache_ptr, V_cache_ptr,
    k_scales_ptr, v_scales_ptr,
    block_tables_ptr, indices_flat_ptr,
    stride_wk_slot, stride_wk_kv, stride_wk_d,
    stride_wv_slot, stride_wv_kv, stride_wv_d,
    stride_wp_slot, stride_wp_kv,
    stride_kc_blk, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_vc_blk, stride_vc_h, stride_vc_s, stride_vc_d,
    stride_ks_h, stride_ks_d,
    stride_vs_h, stride_vs_d,
    stride_bt_req, stride_bt_blk,
    stride_idx_g, stride_idx_h, stride_idx_k,
    head_dim: tl.constexpr,
    sparse_block_size: tl.constexpr,
    cache_block_size: tl.constexpr,
    cache_layout: tl.constexpr,
    BLOCK_RS2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_KSCALES: tl.constexpr,
    HAS_VSCALES: tl.constexpr,
    group, hkv, qid, s_count, total_kv_len,
    beg_blk, cur_blk, wksp_id,
):
    """Gather sparse KV blocks from paged cache into workspace."""
    offs_sbs = tl.arange(0, sparse_block_size)
    loop_d_times: tl.constexpr = (head_dim + BLOCK_D - 1) // BLOCK_D
    wk_base_k = wksp_id * stride_wk_slot
    wk_base_v = wksp_id * stride_wv_slot
    wk_base_p = wksp_id * stride_wp_slot

    for j in range(BLOCK_RS2):
        blk_idx = beg_blk + j
        if j < cur_blk:
            logical_blk = tl.load(
                indices_flat_ptr + group * stride_idx_g
                + hkv * stride_idx_h + blk_idx * stride_idx_k,
            ).to(tl.int32)
            if logical_blk >= 0:
                token_start = logical_blk * sparse_block_size
                offs_token = token_start + offs_sbs
                mask_token = offs_token < total_kv_len
                page_ids = offs_token // cache_block_size
                page_offsets = offs_token - page_ids * cache_block_size
                phys_pages = tl.load(
                    block_tables_ptr + qid * stride_bt_req + page_ids * stride_bt_blk,
                ).to(tl.int32)
                valid_pages = phys_pages >= 0
                ws_offs_token = j * sparse_block_size + offs_sbs
                tl.store(
                    Wksp_pos_ptr + wk_base_p + ws_offs_token * stride_wp_kv,
                    offs_token, mask=mask_token & valid_pages,
                )

                for d in range(loop_d_times):
                    offs_d = d * BLOCK_D + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < head_dim
                    if cache_layout == 0:
                        k_offset = (phys_pages[:, None] * stride_kc_blk + hkv * stride_kc_h
                                    + page_offsets[:, None] * stride_kc_s + offs_d[None, :] * stride_kc_d)
                        v_offset = (phys_pages[:, None] * stride_vc_blk + hkv * stride_vc_h
                                    + page_offsets[:, None] * stride_vc_s + offs_d[None, :] * stride_vc_d)
                    else:
                        k_offset = (phys_pages[:, None] * stride_kc_blk + page_offsets[:, None] * stride_kc_s
                                    + hkv * stride_kc_h + offs_d[None, :] * stride_kc_d)
                        v_offset = (phys_pages[:, None] * stride_vc_blk + page_offsets[:, None] * stride_vc_s
                                    + hkv * stride_vc_h + offs_d[None, :] * stride_vc_d)
                    combined_mask = mask_token[:, None] & valid_pages[:, None] & mask_d[None, :]
                    k_vals = tl.load(K_cache_ptr + k_offset, mask=combined_mask).to(tl.float32)
                    v_vals = tl.load(V_cache_ptr + v_offset, mask=combined_mask).to(tl.float32)
                    if HAS_KSCALES:
                        k_scale = tl.load(
                            k_scales_ptr + hkv * stride_ks_h + offs_d[None, :] * stride_ks_d,
                            mask=mask_d[None, :],
                        )
                        k_vals = k_vals * k_scale
                    if HAS_VSCALES:
                        v_scale = tl.load(
                            v_scales_ptr + hkv * stride_vs_h + offs_d[None, :] * stride_vs_d,
                            mask=mask_d[None, :],
                        )
                        v_vals = v_vals * v_scale
                    tl.store(
                        Wksp_K_ptr + wk_base_k + ws_offs_token[:, None] * stride_wk_kv
                        + offs_d[None, :] * stride_wk_d, k_vals, mask=combined_mask,
                    )
                    tl.store(
                        Wksp_V_ptr + wk_base_v + ws_offs_token[:, None] * stride_wv_kv
                        + offs_d[None, :] * stride_wv_d, v_vals, mask=combined_mask,
                    )


# ---- Main kernel — per query-head parallelization ----

@triton.jit(do_not_specialize=[
    "T", "G", "cumsum_q_len_len", "tasks_per_prog",
])
def _sals_sfa_fwd_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr,
    k_scales_ptr, v_scales_ptr,
    block_tables_ptr, indices_flat_ptr, seq_len_flat_ptr,
    group_qid_ptr, group_q_start_ptr, group_q_len_ptr,
    cumsum_q_len_ptr, base_kv_len_ptr, group_use_dense_ptr,
    output_ptr,
    Wksp_K_ptr, Wksp_V_ptr, Wksp_pos_ptr,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kc_blk, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_vc_blk, stride_vc_h, stride_vc_s, stride_vc_d,
    stride_ks_h, stride_ks_d, stride_vs_h, stride_vs_d,
    stride_bt_req, stride_bt_blk,
    stride_idx_g, stride_idx_h, stride_idx_k,
    stride_o_t, stride_o_h, stride_o_d,
    stride_wk_slot, stride_wk_kv, stride_wk_d,
    stride_wv_slot, stride_wv_kv, stride_wv_d,
    stride_wp_slot, stride_wp_kv,
    T, G, cumsum_q_len_len, softmax_scale, tasks_per_prog,
    num_query_heads: tl.constexpr,
    g_ratio: tl.constexpr,
    head_dim: tl.constexpr,
    sparse_block_size: tl.constexpr,
    cache_block_size: tl.constexpr,
    cache_layout: tl.constexpr,
    HAS_KSCALES: tl.constexpr,
    HAS_VSCALES: tl.constexpr,
    HAS_USEDENSE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_RS2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """SALS SFA Forward — per (group, query_head) parallelization.
    Grid: (prog_num,), each program handles tasks_per_prog (group, hq) pairs.
    Per pair: gather KV for the corresponding kv_head, compute attention.
    """
    pid = tl.program_id(0)
    BLOCK_KV: tl.constexpr = BLOCK_RS2 * sparse_block_size

    for task_sub in range(tasks_per_prog):
        task = pid * tasks_per_prog + task_sub
        if task < G * num_query_heads:
            group = task // num_query_heads
            hq = task % num_query_heads
            hkv = hq // g_ratio
            qid = tl.load(group_qid_ptr + group).to(tl.int32)
            q_start = tl.load(group_q_start_ptr + group).to(tl.int32)
            q_len = tl.load(group_q_len_ptr + group).to(tl.int32)
            s_count = tl.load(seq_len_flat_ptr + group).to(tl.int32)
            should_process = (q_len > 0) & (q_start < T) & (s_count > 0)
            if HAS_USEDENSE:
                use_dense = tl.load(group_use_dense_ptr + group).to(tl.int32)
                should_process = should_process & (use_dense != 1)
            q_end = q_start + q_len
            if q_end > T:
                q_end = T
            actual_q_len = q_end - q_start
            should_process = should_process & (actual_q_len > 0)
            req_start = tl.load(cumsum_q_len_ptr + qid).to(tl.int32)
            if qid + 1 < cumsum_q_len_len:
                req_end = tl.load(cumsum_q_len_ptr + qid + 1).to(tl.int32)
            else:
                req_end = req_start
            q_len_req = req_end - req_start
            start_in_req = q_start - req_start
            base_kv = tl.load(base_kv_len_ptr + qid).to(tl.int32)
            total_kv_len = base_kv + q_len_req

            if should_process:
                num_q_tiles = tl.cdiv(actual_q_len, BLOCK_Q)
                q_base = q_start * stride_q_t + hq * stride_q_h
                o_base = q_start * stride_o_t + hq * stride_o_h
                wksp_id = pid
                wk_base_k = wksp_id * stride_wk_slot
                wk_base_v = wksp_id * stride_wv_slot
                wk_base_p = wksp_id * stride_wp_slot
                num_blk_batches = tl.cdiv(s_count, BLOCK_RS2)
                offs_kv = tl.arange(0, BLOCK_KV)

                for q_tile in range(num_q_tiles):
                    q_offs = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
                    mask_q = q_offs < actual_q_len
                    causal_thresh = (base_kv + start_in_req + q_offs).to(tl.int32)
                    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) + float('-inf')
                    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
                    o_i = tl.zeros([BLOCK_Q, head_dim], dtype=tl.float32)

                    for blk_batch in range(num_blk_batches):
                        beg_blk = blk_batch * BLOCK_RS2
                        cur_blk = s_count - beg_blk
                        if cur_blk > BLOCK_RS2:
                            cur_blk = BLOCK_RS2
                        cur_kv = cur_blk * sparse_block_size

                        _gather_kv_to_wksp(
                            Wksp_K_ptr, Wksp_V_ptr,
                            Wksp_pos_ptr,
                            K_cache_ptr, V_cache_ptr,
                            k_scales_ptr, v_scales_ptr,
                            block_tables_ptr,
                            indices_flat_ptr,
                            stride_wk_slot,
                            stride_wk_kv,
                            stride_wk_d,
                            stride_wv_slot,
                            stride_wv_kv,
                            stride_wv_d,
                            stride_wp_slot,
                            stride_wp_kv,
                            stride_kc_blk,
                            stride_kc_h,
                            stride_kc_s,
                            stride_kc_d,
                            stride_vc_blk,
                            stride_vc_h,
                            stride_vc_s,
                            stride_vc_d,
                            stride_ks_h,
                            stride_ks_d,
                            stride_vs_h,
                            stride_vs_d,
                            stride_bt_req,
                            stride_bt_blk,
                            stride_idx_g,
                            stride_idx_h,
                            stride_idx_k,
                            head_dim,
                            sparse_block_size,
                            cache_block_size,
                            cache_layout,
                            BLOCK_RS2, BLOCK_D,
                            HAS_KSCALES,
                            HAS_VSCALES,
                            group, hkv, qid,
                            s_count, total_kv_len,
                            beg_blk, cur_blk,
                            wksp_id,
                        )

                        token_pos = tl.load(
                            Wksp_pos_ptr + wk_base_p + offs_kv * stride_wp_kv,
                            mask=offs_kv < cur_kv, other=-1,
                        )
                        valid_kv = (offs_kv[None, :] < cur_kv) & (token_pos[None, :] >= 0) & (token_pos[None, :] < total_kv_len)
                        scores = cube_qkt_sfa(
                            Q_ptr, Wksp_K_ptr,
                            stride_q_t, stride_q_h, stride_q_d,
                            stride_wk_slot, stride_wk_kv, stride_wk_d,
                            head_dim, BLOCK_Q, BLOCK_KV, BLOCK_D,
                            q_offs, offs_kv, q_base, wk_base_k, cur_kv,
                        )
                        scores = scores * softmax_scale
                        causal_mask = token_pos[None, :] > causal_thresh[:, None]
                        scores = tl.where(valid_kv, scores, float('-inf'))
                        scores = tl.where(mask_q[:, None], scores, float('-inf'))
                        scores = tl.where(causal_mask, float('-inf'), scores)
                        row_max_j = tl.max(scores, axis=1)
                        m_new = tl.maximum(m_i, row_max_j)
                        alpha = tl.where(m_i > float('-inf'), tl.exp(m_i - m_new), 0.0)
                        p = tl.exp(scores - m_new[:, None])
                        p = tl.where(scores > float('-inf'), p, 0.0)
                        l_i = alpha * l_i + tl.sum(p, axis=1)
                        pv = cube_pv_sfa(
                            p, Wksp_V_ptr,
                            stride_wv_slot, stride_wv_kv, stride_wv_d,
                            head_dim, BLOCK_Q, BLOCK_KV,
                            offs_kv, wk_base_v, cur_kv,
                        )
                        o_i = alpha[:, None] * o_i + pv
                        m_i = m_new

                    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
                    out = o_i / safe_l[:, None]
                    offs_d_full = tl.arange(0, head_dim)
                    out_vals = out.to(output_ptr.dtype.element_ty)
                    tl.store(
                        output_ptr + o_base + q_offs[:, None] * stride_o_t
                        + offs_d_full[None, :] * stride_o_d,
                        out_vals, mask=mask_q[:, None],
                    )


def sals_sfa_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: Optional[torch.Tensor],
    v_scales: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    indices_flat: torch.Tensor,
    seq_len_flat: torch.Tensor,
    group_qid: torch.Tensor,
    group_q_start: torch.Tensor,
    group_q_len: torch.Tensor,
    cumsum_q_len: torch.Tensor,
    base_kv_len: torch.Tensor,
    group_use_dense: Optional[torch.Tensor],
    softmax_scale: float,
    num_kv_heads: int,
    num_query_heads: int,
    head_dim: int,
    sparse_block_size: int,
) -> torch.Tensor:
    T = q.shape[0]
    G = int(indices_flat.shape[0])
    device = q.device
    output = torch.zeros(T, num_query_heads, head_dim, dtype=q.dtype, device=device)
    if G == 0:
        return output

    indices_flat, seq_len_flat = _expand_indices_by_group_union(
        indices_flat, seq_len_flat, num_kv_heads,
    )

    if k_cache.shape[1] == num_kv_heads:
        cache_layout = 0
        cache_block_size = k_cache.shape[2]
    else:
        cache_layout = 1
        cache_block_size = k_cache.shape[1]

    g_ratio = num_query_heads // num_kv_heads
    BLOCK_Q = 16
    # Keep BLOCK_KV small enough for 910B2C local buffers. 64-token sparse
    # blocks overflow cbuf/ub with BLOCK_RS2=8, so shrink the block batch.
    if sparse_block_size >= 128:
        BLOCK_RS2 = 2
    elif sparse_block_size >= 64:
        BLOCK_RS2 = 4
    else:
        BLOCK_RS2 = 8
    BLOCK_D = head_dim
    BLOCK_KV = BLOCK_RS2 * sparse_block_size

    total_tasks = G * num_query_heads
    core_num = get_num_cores("cube")
    prog_num = min(total_tasks, core_num)
    tasks_per_prog = triton.cdiv(total_tasks, prog_num)

    wksp_count = prog_num
    workspace_k = torch.empty((wksp_count, BLOCK_KV, head_dim), dtype=torch.float32, device=device)
    workspace_v = torch.empty((wksp_count, BLOCK_KV, head_dim), dtype=torch.float32, device=device)
    workspace_pos = torch.full((wksp_count, BLOCK_KV), -1, dtype=torch.int64, device=device)

    if k_scales is None:
        k_scales = torch.empty(0, dtype=torch.float32, device=device)
    if v_scales is None:
        v_scales = torch.empty(0, dtype=torch.float32, device=device)
    if group_use_dense is None:
        group_use_dense = torch.empty(0, dtype=torch.int32, device=device)

    cumsum_q_len_len = cumsum_q_len.shape[0]
    s_q_t, s_q_h, s_q_d = q.stride(0), q.stride(1), q.stride(2)
    if cache_layout == 0:
        s_kc_blk, s_kc_h, s_kc_s, s_kc_d = (
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3))
    else:
        s_kc_blk, s_kc_s, s_kc_h, s_kc_d = (
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3))
    if cache_layout == 0:
        s_vc_blk, s_vc_h, s_vc_s, s_vc_d = (
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3))
    else:
        s_vc_blk, s_vc_s, s_vc_h, s_vc_d = (
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3))
    s_ks_h = k_scales.stride(0) if k_scales.numel() > 0 else 0
    s_ks_d = k_scales.stride(1) if k_scales.dim() > 1 and k_scales.numel() > 0 else 0
    s_vs_h = v_scales.stride(0) if v_scales.numel() > 0 else 0
    s_vs_d = v_scales.stride(1) if v_scales.dim() > 1 and v_scales.numel() > 0 else 0
    s_bt_req, s_bt_blk = block_tables.stride(0), block_tables.stride(1)
    s_idx_g, s_idx_h, s_idx_k = (
        indices_flat.stride(0), indices_flat.stride(1), indices_flat.stride(2))
    s_o_t, s_o_h, s_o_d = output.stride(0), output.stride(1), output.stride(2)
    s_wk_slot, s_wk_kv, s_wk_d = (
        workspace_k.stride(0), workspace_k.stride(1), workspace_k.stride(2))
    s_wv_slot, s_wv_kv, s_wv_d = (
        workspace_v.stride(0), workspace_v.stride(1), workspace_v.stride(2))
    s_wp_slot, s_wp_kv = workspace_pos.stride(0), workspace_pos.stride(1)

    grid = (prog_num,)
    _sals_sfa_fwd_kernel[grid](
        q, k_cache, v_cache, k_scales, v_scales,
        block_tables, indices_flat, seq_len_flat,
        group_qid, group_q_start, group_q_len,
        cumsum_q_len, base_kv_len, group_use_dense,
        output,
        workspace_k, workspace_v, workspace_pos,
        s_q_t, s_q_h, s_q_d,
        s_kc_blk, s_kc_h, s_kc_s, s_kc_d,
        s_vc_blk, s_vc_h, s_vc_s, s_vc_d,
        s_ks_h, s_ks_d, s_vs_h, s_vs_d,
        s_bt_req, s_bt_blk,
        s_idx_g, s_idx_h, s_idx_k,
        s_o_t, s_o_h, s_o_d,
        s_wk_slot, s_wk_kv, s_wk_d,
        s_wv_slot, s_wv_kv, s_wv_d,
        s_wp_slot, s_wp_kv,
        T, G, cumsum_q_len_len, softmax_scale, tasks_per_prog,
        num_query_heads=num_query_heads, g_ratio=g_ratio, head_dim=head_dim,
        sparse_block_size=sparse_block_size, cache_block_size=cache_block_size,
        cache_layout=cache_layout,
        HAS_KSCALES=k_scales.numel() > 0, HAS_VSCALES=v_scales.numel() > 0,
        HAS_USEDENSE=group_use_dense.numel() > 0,
        BLOCK_Q=BLOCK_Q, BLOCK_RS2=BLOCK_RS2, BLOCK_D=BLOCK_D,
        multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
    )
    return output
