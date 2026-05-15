from typing import Optional, Tuple

import math

import torch

from ..operator import MojoOperator


# ---------------------------------------------------------------------------
# Helper functions for SALS indexer (pure PyTorch reference)
# ---------------------------------------------------------------------------


def _get_data_from_pa_cache(
    key: torch.Tensor,
    block_table_1d: torch.Tensor,
    act_s2: int,
) -> torch.Tensor:
    """Gather a contiguous [S, N, D] view from a paged-attention cache.

    Args:
        key: [num_blocks, block_size, N, D]
        block_table_1d: [table_len] mapping logical blocks -> physical blocks
        act_s2: actual visible KV tokens

    Returns:
        now_k: [act_s2, N, D]
    """
    if act_s2 <= 0:
        return key.new_zeros((0, key.shape[2], key.shape[3]))

    num_blocks, block_size, n2, d = key.shape
    need_block_num = (int(act_s2) + int(block_size) - 1) // int(block_size)

    block_ids = block_table_1d[:need_block_num].to(torch.long)
    if (block_ids < 0).any():
        block_ids = block_ids[block_ids >= 0]
        need_block_num = int(block_ids.numel())
        if need_block_num == 0:
            return key.new_zeros((0, n2, d))

    gathered = key.index_select(0, block_ids)  # [need_block_num, block_size, N, D]
    flat = gathered.reshape(need_block_num * block_size, n2, d)
    return flat[:act_s2]


def _block_scores_from_token_scores(
    s_out: torch.Tensor,
    sort_n_count: int,
    sparse_block_size: int,
    act_s2: int,
    score_mode: str,
) -> torch.Tensor:
    """Compute per-block scores from per-token scores.

    Args:
        s_out: [N, 1, S]
    Returns:
        s_scores: [N, 1, sort_n_count]
    """
    n2 = s_out.shape[0]
    padded_len = int(sort_n_count) * int(sparse_block_size)
    if padded_len <= 0:
        return s_out.new_zeros((n2, 1, 0))

    s_out = s_out[:, :, : min(int(act_s2), padded_len)]
    if s_out.shape[-1] < padded_len:
        s_out = torch.nn.functional.pad(s_out, (0, padded_len - s_out.shape[-1]), value=float('-inf'))

    s_blocks = s_out.reshape(n2, 1, sort_n_count, sparse_block_size)

    mode = (score_mode or "").strip().lower()
    if mode == "softmax_sum":
        attn = torch.softmax(s_out, dim=-1)
        attn_blocks = attn.reshape(n2, 1, sort_n_count, sparse_block_size)
        return attn_blocks.sum(dim=3)

    if mode == "max_pooling":
        return s_blocks.max(dim=3)[0]

    if mode == "mean":
        return s_blocks.mean(dim=3)

    if mode == "weighted":
        max_v = s_blocks.max(dim=3)[0]
        mean_v = s_blocks.mean(dim=3)
        return 0.7 * max_v + 0.3 * mean_v

    if mode == "topk_mean":
        k = min(4, int(sparse_block_size))
        topk_vals = torch.topk(s_blocks, k=k, dim=3)[0]
        return topk_vals.mean(dim=3)

    if mode == "lse":
        return torch.logsumexp(s_blocks, dim=3)

    if mode == "minference_vertical_slash":
        max_score = s_blocks.max(dim=3)[0]
        attn = torch.softmax(s_out, dim=-1)
        attn_blocks = attn.reshape(n2, 1, sort_n_count, sparse_block_size)
        sum_attn = attn_blocks.sum(dim=3)
        return 0.6 * max_score + 0.4 * sum_attn

    if mode == "flexprefill_cumulative":
        attn = torch.softmax(s_out, dim=-1)
        attn_blocks = attn.reshape(n2, 1, sort_n_count, sparse_block_size)
        sum_attn = attn_blocks.sum(dim=3)
        sorted_attn, _ = sum_attn.sort(dim=-1, descending=True)
        cumsum = sorted_attn.cumsum(dim=-1)
        gamma = 0.9
        threshold = gamma * cumsum[:, :, -1:]
        weight = (cumsum <= threshold).float()
        return sum_attn * (1.0 + weight)

    return s_blocks.max(dim=3)[0]


def _quest_minmax_scores(
    now_q: torch.Tensor,
    now_k: torch.Tensor,
    sort_n_count: int,
    sparse_block_size: int,
    act_s2: int,
) -> torch.Tensor:
    """Quest min-max upper bound scoring.

    score = sum( max(q[i]*k_min[i], q[i]*k_max[i]) )
    """
    n2, _, d = now_q.shape
    padded_len = int(sort_n_count) * int(sparse_block_size)
    if padded_len <= 0:
        return now_q.new_zeros((n2, 1, 0))

    kk = now_k[: min(int(act_s2), padded_len)]
    if kk.shape[0] < padded_len:
        kk = torch.nn.functional.pad(kk, (0, 0, 0, 0, 0, padded_len - kk.shape[0]))

    kk_blocks = kk.reshape(sort_n_count, sparse_block_size, n2, d)
    k_min = kk_blocks.min(dim=1)[0]  # [sort_n_count, N, D]
    k_max = kk_blocks.max(dim=1)[0]  # [sort_n_count, N, D]

    q = now_q.squeeze(1)  # [N, D]

    q_mul_min = q.view(1, n2, d) * k_min  # [sort_n_count, N, D]
    q_mul_max = q.view(1, n2, d) * k_max  # [sort_n_count, N, D]
    upper_bound = torch.maximum(q_mul_min, q_mul_max).sum(dim=2)  # [sort_n_count, N]

    upper_bound = upper_bound / math.sqrt(d)

    return upper_bound.transpose(0, 1).unsqueeze(1)  # [N, 1, sort_n_count]


def _quest_threshold_scan_topk(scores_2d: torch.Tensor, topk_n: int) -> torch.Tensor:
    """Quest topk selection via threshold scan.

    Args:
        scores_2d: [N, S]
        topk_n: number to select
    Returns:
        indices: [N, topk_n] int32
    """
    n2, s = int(scores_2d.shape[0]), int(scores_2d.shape[1])
    k = max(1, min(int(topk_n), s))
    sorted_vals = torch.sort(scores_2d, dim=1).values
    thr = sorted_vals[:, -k]  # [N]

    out = torch.empty((n2, k), dtype=torch.int32, device=scores_2d.device)
    for h in range(n2):
        kept = []
        th = float(thr[h].item())
        row = scores_2d[h]
        for idx in range(s):
            if float(row[idx].item()) >= th:
                kept.append(idx)
                if len(kept) >= k:
                    break
        if len(kept) < k:
            for idx in range(s):
                if idx in kept:
                    continue
                kept.append(idx)
                if len(kept) >= k:
                    break
        out[h] = torch.tensor(kept[:k], dtype=torch.int32, device=scores_2d.device)
    return out


class MojoSALSIndexer(MojoOperator):
    def forward(
        self,
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
        G = int(query.shape[0])
        n2 = int(query.shape[1])
        d = int(query.shape[2])
        mode = (score_mode or "").strip().lower()

        sparse_indices = torch.full(
            (G, n2, sparse_count), -1, dtype=torch.int32, device=query.device
        )
        sparse_seq_lengths_key = torch.zeros(
            (G,), dtype=torch.int32, device=query.device
        )

        act_s2_list = actual_seq_lengths_key.tolist()
        act_n_counts_list = (
            act_n_counts.tolist()
            if torch.is_tensor(act_n_counts)
            else list(act_n_counts)
        )

        for g in range(G):
            act_s2 = int(act_s2_list[g])
            act_s2 = max(act_s2, 0)
            act_n_count = int(act_n_counts_list[g])
            if act_n_count <= 0:
                continue
            if act_n_count < fixed_tail_count + 4:
                continue

            sort_n_count = act_n_count - fixed_tail_count
            topk_n_count = max(
                1, min(int(sort_n_count * float(sparse_ratio) + 0.5), sort_n_count)
            )
            sparse_seq_lengths_key[g] = topk_n_count + fixed_tail_count

            now_q = query[g].reshape(n2, 1, d).to(torch.float32)
            now_k = _get_data_from_pa_cache(
                key, block_table[g], act_s2
            ).to(torch.float32)
            now_k_ndS = now_k.permute(1, 2, 0)
            s_out = torch.matmul(now_q, now_k_ndS) / math.sqrt(d)

            if mode in ("quest_minmax", "quest_page", "quest_full"):
                s_scores = _quest_minmax_scores(
                    now_q, now_k, sort_n_count, sparse_block_size, act_s2
                )
            else:
                s_scores = _block_scores_from_token_scores(
                    s_out, sort_n_count, sparse_block_size, act_s2, score_mode
                )

            s_scores_2d = s_scores.squeeze(1)

            if mode in ("quest_page", "quest_full"):
                topk_idx = _quest_threshold_scan_topk(s_scores_2d, topk_n_count)
            else:
                topk_idx = torch.topk(
                    s_scores_2d, k=topk_n_count, dim=1, largest=True, sorted=False
                )[1].to(torch.int32)
            sparse_indices[g, :, :topk_n_count] = topk_idx

            if fixed_tail_count > 0:
                tail = torch.arange(
                    sort_n_count,
                    sort_n_count + fixed_tail_count,
                    device=query.device,
                    dtype=torch.int32,
                )
                sparse_indices[
                    g, :, topk_n_count : topk_n_count + fixed_tail_count
                ] = tail.view(1, -1).expand(n2, -1)

        return sparse_indices, sparse_seq_lengths_key
