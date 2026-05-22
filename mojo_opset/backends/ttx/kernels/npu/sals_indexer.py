from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


@triton.jit
def sals_indexer_kernel(
    query_ptr,
    key_ptr,
    block_table_ptr,
    actual_seq_lengths_key_ptr,
    act_n_counts_ptr,
    sparse_indices_ptr,
    sparse_seq_lengths_key_ptr,
    stride_qg, stride_qn, stride_qd,
    stride_kblk, stride_kbs, stride_kn, stride_kd,
    stride_btg, stride_btt,
    stride_si_g, stride_si_n, stride_si_k,
    N: tl.constexpr,
    D: tl.constexpr,
    SBS: tl.constexpr,
    CBS: tl.constexpr,
    max_sort_n: tl.constexpr,
    sparse_count: tl.constexpr,
    sparse_ratio_f32,
    fixed_tail_count: tl.constexpr,
    scale,
    t1_per_prog,
    G_total,
):
    pid = tl.program_id(0)
    offs_sbs = tl.arange(0, SBS)
    offs_d = tl.arange(0, D)
    offs_sort = tl.arange(0, max_sort_n)

    for idx_sub in range(0, t1_per_prog):
        flat = pid * t1_per_prog + idx_sub

        if flat < G_total * N:
            g = flat // N
            n = flat % N

            act_s2 = tl.load(actual_seq_lengths_key_ptr + g).to(tl.int32)
            act_n_count = tl.load(act_n_counts_ptr + g).to(tl.int32)

            if act_n_count > 0:
                if act_n_count < fixed_tail_count + 4:
                    keep_n = act_n_count
                    if keep_n > sparse_count:
                        keep_n = sparse_count
                    if n == 0:
                        tl.store(sparse_seq_lengths_key_ptr + g, keep_n)
                    for ki in range(0, sparse_count):
                        if ki < keep_n:
                            tl.store(
                                sparse_indices_ptr + g * stride_si_g + n * stride_si_n + ki * stride_si_k,
                                ki,
                            )
                else:
                    sort_n_count = act_n_count - fixed_tail_count
                    tmp = (sort_n_count * sparse_ratio_f32 + 0.5).to(tl.int32)
                    topk_n_count = sort_n_count
                    if tmp < 1:
                        tmp = 1
                    if tmp < sort_n_count:
                        topk_n_count = tmp
                    max_sparse_topk = sparse_count - fixed_tail_count
                    if max_sparse_topk < 1:
                        max_sparse_topk = 1
                    if topk_n_count > max_sparse_topk:
                        topk_n_count = max_sparse_topk

                    if n == 0:
                        tl.store(sparse_seq_lengths_key_ptr + g, topk_n_count + fixed_tail_count)

                    # Load Q[g, n, :] once — full D vector
                    q_vec = tl.load(
                        query_ptr + g * stride_qg + n * stride_qn + offs_d * stride_qd,
                    ).to(tl.float32)

                    # Stage 1: Compute block LSE scores into a register vector
                    scores_vec = tl.full([max_sort_n], -1e30, dtype=tl.float32)

                    for b in range(0, max_sort_n):
                        if b < sort_n_count:
                            block_start = b * SBS
                            mask_s = (block_start + offs_sbs) < act_s2
                            token_pos = block_start + offs_sbs
                            page_ids = token_pos // CBS
                            page_offsets = token_pos - page_ids * CBS
                            phys_id = tl.load(
                                block_table_ptr + g * stride_btg + page_ids * stride_btt
                            ).to(tl.int32)
                            valid_page = phys_id >= 0

                            # K[phys_id, :, n, :] → [SBS, D]
                            k_block = tl.load(
                                key_ptr + phys_id[:, None] * stride_kblk
                                + page_offsets[:, None] * stride_kbs
                                + n * stride_kn
                                + offs_d[None, :] * stride_kd,
                                mask=mask_s[:, None] & valid_page[:, None],
                                other=0.0,
                            ).to(tl.float32)

                            # dot: [SBS, D] @ [D] → [SBS] via broadcast multiply + sum
                            dot_scores = tl.sum(k_block * q_vec[None, :], axis=1)
                            dot_scores = dot_scores * scale

                            # logsumexp over SBS
                            masked_scores = tl.where(mask_s & valid_page, dot_scores, -1e30)
                            m = tl.max(masked_scores)
                            e = tl.exp(masked_scores - m)
                            lse = m + tl.log(tl.sum(e))

                            scores_vec = tl.where(offs_sort == b, lse, scores_vec)

                    # Stage 2: Vectorized TopK via iterative max
                    sort_mask = offs_sort < sort_n_count
                    working_scores = tl.where(sort_mask, scores_vec, -1e30)

                    for ki in range(0, max_sort_n):
                        if ki < topk_n_count:
                            best_score = tl.max(working_scores)
                            is_best = (working_scores == best_score) & sort_mask
                            idx_candidates = tl.where(
                                is_best,
                                offs_sort.to(tl.float32),
                                float(max_sort_n),
                            )
                            best_idx = tl.min(idx_candidates).to(tl.int32)

                            working_scores = tl.where(
                                offs_sort == best_idx,
                                tl.full([max_sort_n], -1e30, dtype=tl.float32),
                                working_scores,
                            )

                            tl.store(
                                sparse_indices_ptr + g * stride_si_g + n * stride_si_n + ki * stride_si_k,
                                best_idx,
                            )

                    # Stage 3: Tail append
                    for t in range(0, fixed_tail_count):
                        if topk_n_count + t < sparse_count:
                            tl.store(
                                sparse_indices_ptr + g * stride_si_g + n * stride_si_n + (topk_n_count + t) * stride_si_k,
                                sort_n_count + t,
                            )


def sals_indexer_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    act_n_counts: torch.Tensor,
    sparse_block_size: int,
    sparse_ratio: float,
    fixed_tail_count: int,
    sparse_count: int,
    score_mode: str,
    max_seqlen_key: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    G, N_dim, D_dim = query.shape
    T1 = G * N_dim

    core_num = get_num_cores("cube")
    if T1 <= core_num:
        prog_num = T1
        t1_per_prog = 1
    else:
        prog_num = core_num
        t1_per_prog = triton.cdiv(T1, core_num)

    max_sort_n = (max_seqlen_key + sparse_block_size - 1) // sparse_block_size
    scale = 1.0 / (D_dim ** 0.5)

    sparse_indices = torch.full((G, N_dim, sparse_count), -1, dtype=torch.int32, device=query.device)
    sparse_seq_lengths_key = torch.zeros((G,), dtype=torch.int32, device=query.device)

    q_s = query.stride()
    k_s = key.stride()
    bt_s = block_table.stride()
    si_s = sparse_indices.stride()

    grid = (prog_num,)
    sals_indexer_kernel[grid](
        query, key, block_table, actual_seq_lengths_key, act_n_counts,
        sparse_indices, sparse_seq_lengths_key,
        q_s[0], q_s[1], q_s[2],
        k_s[0], k_s[1], k_s[2], k_s[3],
        bt_s[0], bt_s[1] if len(bt_s) > 1 else 1,
        si_s[0], si_s[1], si_s[2],
        N_dim, D_dim, sparse_block_size, key.shape[1], max_sort_n, sparse_count,
        float(sparse_ratio), fixed_tail_count,
        scale,
        t1_per_prog=t1_per_prog,
        G_total=G,
    )

    return sparse_indices, sparse_seq_lengths_key
