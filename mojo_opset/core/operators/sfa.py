from typing import Optional, Tuple

import torch

from ..operator import MojoOperator


# ---------------------------------------------------------------------------
# Helper functions for SALS SFA (pure PyTorch reference)
# ---------------------------------------------------------------------------


def _gather_kv_from_cache_tnd(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: Optional[torch.Tensor],
    v_scales: Optional[torch.Tensor],
    block_table: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_count: int,
    total_kv_len: int,
    num_kv_heads: int,
    sparse_block_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather KV from paged attention cache.

    Auto-detects cache layout:
    - PA_BNSD: [num_blocks, kvH, block_size, D]
    - PA_BSND: [num_blocks, block_size, kvH, D]

    Returns:
        k_sparse: [L, kvH, D]
        v_sparse: [L, kvH, D]
        token_positions: [L]
    """
    device = k_cache.device

    if k_cache.shape[1] == num_kv_heads:
        cache_block_size = k_cache.shape[2]
        head_dim = k_cache.shape[3]
        layout = "BNSD"
    else:
        cache_block_size = k_cache.shape[1]
        head_dim = k_cache.shape[3]
        layout = "BSND"

    sparse_block_size = int(sparse_block_size or cache_block_size)

    if not torch.is_tensor(block_table):
        block_table = torch.tensor(block_table, device=device, dtype=torch.long)
    block_table = block_table.to(device=device, dtype=torch.long).flatten()

    max_valid_block = (total_kv_len + sparse_block_size - 1) // sparse_block_size

    all_block_ids = sparse_indices[:, :sparse_count].flatten()
    all_block_ids = all_block_ids[(all_block_ids >= 0) & (all_block_ids < max_valid_block)]

    empty = (
        k_cache.new_zeros((0, num_kv_heads, head_dim)),
        v_cache.new_zeros((0, num_kv_heads, head_dim)),
        torch.empty((0,), device=device, dtype=torch.long),
    )

    if all_block_ids.numel() == 0:
        return empty

    valid_logical = torch.unique(all_block_ids, sorted=True)
    valid_logical = valid_logical[valid_logical < max_valid_block]
    if valid_logical.numel() == 0:
        return empty

    offsets = torch.arange(sparse_block_size, device=device, dtype=torch.long)
    token_positions_2d = valid_logical.view(-1, 1) * sparse_block_size + offsets.view(1, -1)

    token_positions_flat = token_positions_2d.flatten()
    valid_tokens = token_positions_flat < total_kv_len
    token_positions = token_positions_flat[valid_tokens]

    page_ids = torch.div(token_positions, cache_block_size, rounding_mode='floor')
    page_offsets = token_positions - page_ids * cache_block_size

    valid_pages = page_ids < block_table.numel()
    if not bool(valid_pages.all().item()):
        token_positions = token_positions[valid_pages]
        page_ids = page_ids[valid_pages]
        page_offsets = page_offsets[valid_pages]

    if token_positions.numel() == 0:
        return empty

    phys_blocks = block_table[page_ids.long()]
    valid_mask = phys_blocks >= 0
    if not bool(valid_mask.all().item()):
        token_positions = token_positions[valid_mask]
        page_offsets = page_offsets[valid_mask]
        phys_blocks = phys_blocks[valid_mask]

    if token_positions.numel() == 0:
        return empty

    if layout == "BNSD":
        k_sparse = k_cache[phys_blocks.long(), :, page_offsets.long(), :]
        v_sparse = v_cache[phys_blocks.long(), :, page_offsets.long(), :]
    else:
        k_sparse = k_cache[phys_blocks.long(), page_offsets.long(), :, :]
        v_sparse = v_cache[phys_blocks.long(), page_offsets.long(), :, :]

    if k_scales is not None and k_sparse.numel() > 0:
        k_sparse = k_sparse.float() * k_scales.view(1, num_kv_heads, -1)
    if v_scales is not None and v_sparse.numel() > 0:
        v_sparse = v_sparse.float() * v_scales.view(1, num_kv_heads, -1)

    return k_sparse, v_sparse, token_positions


def _compute_causal_mask_tnd(
    q_len: int,
    token_positions: torch.Tensor,
    base_kv: int,
    start_in_req: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute causal mask with base_kv offset.

    Returns:
        mask: [q_len, L] — True means masked (invisible).
    """
    L = token_positions.numel()
    if L == 0:
        return torch.zeros((q_len, 0), dtype=torch.bool, device=device)

    query_visible = base_kv + start_in_req + torch.arange(q_len, device=device, dtype=torch.long)
    mask = token_positions.view(1, L) > query_visible.view(q_len, 1)
    return mask


class MojoSALSSFA(MojoOperator):
    """SALS Sparse Flash Attention operator.

    Performs per-group sparse attention with Flash Attention V2 semantics
    on TND-packed queries against a paged KV cache.

    The torch backend (auto-registered as ``TorchSALSSFA``) provides the
    reference implementation; the ``ttx`` backend uses a Triton fused kernel.
    """

    def forward(
        self,
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
        device = q.device
        dtype = q.dtype
        g = num_query_heads // num_kv_heads

        output = torch.zeros(
            T, num_query_heads, head_dim, dtype=dtype, device=device,
        )

        G = int(indices_flat.shape[0])
        group_qid_py = group_qid.tolist()
        group_q_start_py = group_q_start.tolist()
        group_q_len_py = group_q_len.tolist()
        cumsum_q_py = cumsum_q_len.tolist()
        base_kv_py = base_kv_len.tolist()
        seq_len_py = seq_len_flat.tolist() if seq_len_flat is not None else [0] * G
        group_use_dense_py = group_use_dense.tolist() if group_use_dense is not None else None
        cumsum_len = len(cumsum_q_py)

        for i in range(G):
            qid = int(group_qid_py[i])
            q_start = int(group_q_start_py[i])
            q_len_val = int(group_q_len_py[i])

            if q_len_val <= 0 or q_start >= T:
                continue

            if group_use_dense_py is not None and i < len(group_use_dense_py):
                if int(group_use_dense_py[i]) == 1:
                    continue

            q_end = min(q_start + q_len_val, T)
            actual_q_len = q_end - q_start
            q_seg = q[q_start:q_end]

            req_start = int(cumsum_q_py[qid])
            req_end = int(cumsum_q_py[qid + 1]) if qid + 1 < cumsum_len else req_start
            q_len_req = req_end - req_start
            start_in_req = q_start - req_start
            base_kv = int(base_kv_py[qid])
            total_kv_len = base_kv + q_len_req

            block_ids = indices_flat[i]
            s_count = int(seq_len_py[i])

            k_sparse, v_sparse, token_positions = _gather_kv_from_cache_tnd(
                k_cache, v_cache, k_scales, v_scales,
                block_tables[qid], block_ids, s_count, total_kv_len,
                num_kv_heads, sparse_block_size,
            )

            if k_sparse.numel() == 0:
                continue

            L = k_sparse.shape[0]

            k_expanded = k_sparse.unsqueeze(2).expand(-1, -1, g, -1).reshape(L, num_query_heads, head_dim).to(dtype)
            v_expanded = v_sparse.unsqueeze(2).expand(-1, -1, g, -1).reshape(L, num_query_heads, head_dim).to(dtype)

            scores = torch.einsum("qhd,lhd->qhl", q_seg.float(), k_expanded.float()) * softmax_scale

            causal_mask = _compute_causal_mask_tnd(actual_q_len, token_positions, base_kv, start_in_req, device)
            causal_mask = causal_mask.unsqueeze(1).expand(-1, num_query_heads, -1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

            # Keep the torch reference portable on NPU: fp16 einsum/baddbmm is
            # not implemented in some CANN torch builds. bf16 is supported and
            # keeps long stress cases practical while staying within test tol.
            probs = torch.softmax(scores, dim=-1).nan_to_num(0)
            compute_dtype = torch.bfloat16 if dtype == torch.float16 else dtype
            out = torch.einsum(
                "qhl,lhd->qhd",
                probs.to(compute_dtype),
                v_expanded.to(compute_dtype),
            ).to(dtype)

            output[q_start:q_end] = out

        return output
