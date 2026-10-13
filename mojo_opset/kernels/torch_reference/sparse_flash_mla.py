"""Reference implementation for the sparse flash MLA inference function."""

from typing import Optional

import torch


def sparse_flash_mla_infer_fwd(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    seqused_ori_kv: torch.Tensor,
    sinks: Optional[torch.Tensor],
    win_left: int,
    win_right: int,
    softmax_scale: float,
    layout_kv: str,
    cu_seqlens_ori_kv: Optional[torch.Tensor],
    cmp_kv: Optional[torch.Tensor],
    cmp_block_table: Optional[torch.Tensor],
    seqused_cmp_kv: Optional[torch.Tensor],
    cu_seqlens_cmp_kv: Optional[torch.Tensor],
    cmp_residual_kv: Optional[torch.Tensor],
    cmp_sparse_indices: Optional[torch.Tensor],
    cmp_ratio: int,
    cmp_mask_mode: int,
) -> torch.Tensor:
    """Paged sparse MLA-style decode reference: MQA shared paged KV +
    sliding-window mask + attention-sink seeded online softmax, with an
    optional compressed-KV branch (HCA dense prefix / CSA sparse topk).

    Zero sinks seed the online softmax state with (max=0, sum=1): the
    denominator gets an extra +1 and the running row max is floored at 0.
    """
    _, n1, d = q.shape
    batch = cu_seqlens_q.numel() - 1
    scale = softmax_scale
    ori_flat = ori_kv.reshape(-1, d) if layout_kv == "TND" else None
    cmp_flat = cmp_kv.reshape(-1, d) if (cmp_kv is not None and layout_kv == "TND") else None
    out = torch.empty_like(q)

    for b in range(batch):
        begin, end = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        q_b = q[begin:end]
        s1 = end - begin
        rows = torch.arange(s1, device=q.device)

        # logical ori KV: paged (gather by table) or dense (per-batch prefix)
        if layout_kv == "PA_BBND":
            kv_b = ori_kv.index_select(0, ori_block_table[b].long()).reshape(-1, d).float()
        else:
            kv_b = ori_flat[int(cu_seqlens_ori_kv[b]) : int(cu_seqlens_ori_kv[b + 1])].float()
        s2 = int(seqused_ori_kv[b])
        k_ori = v_ori = kv_b[:s2]

        scores = torch.einsum("mhd,kd->mhk", q_b.float(), k_ori) * scale
        lo = (s2 - s1 + rows - win_left).clamp_min(0)
        hi = (s2 - s1 + rows + win_right + 1).clamp_max(s2)
        pos = torch.arange(s2, device=q.device)
        keep = (pos >= lo.unsqueeze(-1)) & (pos < hi.unsqueeze(-1))
        scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))

        v_extra = None
        if cmp_kv is not None:
            cmp_len = int(seqused_cmp_kv[b])
            residual = int(cmp_residual_kv[b]) if cmp_residual_kv is not None else 0
            if layout_kv == "PA_BBND":
                ckv_b = cmp_kv.index_select(0, cmp_block_table[b].long()).reshape(-1, d).float()
            else:
                ckv_b = cmp_flat[int(cu_seqlens_cmp_kv[b]) : int(cu_seqlens_cmp_kv[b + 1])].float()
            k_cmp = v_cmp = ckv_b[:cmp_len]
            # per-row valid compressed count, mirroring the kernel causal clamp
            if cmp_mask_mode == 3:
                revert = cmp_len * cmp_ratio + (0 if cmp_ratio == 1 else residual)
                nvalid = ((revert - s1 + rows + 1) // cmp_ratio).clamp_(0, cmp_len)
            else:
                nvalid = torch.full_like(rows, cmp_len)
            if cmp_sparse_indices is not None:
                k_slots = cmp_sparse_indices.shape[1]
                nvalid = nvalid.clamp(max=k_slots)
                sel = cmp_sparse_indices[begin:end, :k_slots].long().clamp_min(0)
                scores_cmp = torch.einsum("mhd,mkd->mhk", q_b.float(), k_cmp[sel]) * scale
                v_extra = v_cmp[sel]
            else:
                scores_cmp = torch.einsum("mhd,kd->mhk", q_b.float(), k_cmp) * scale
                v_extra = v_cmp.unsqueeze(0).expand(s1, -1, -1)
            slots = torch.arange(scores_cmp.shape[-1], device=q.device)
            cmp_keep = slots.unsqueeze(0) < nvalid.unsqueeze(-1)
            scores_cmp = scores_cmp.masked_fill(~cmp_keep.unsqueeze(1), float("-inf"))
            scores = torch.cat([scores, scores_cmp], dim=-1)

        # online-softmax seeded by zero sinks: max floored at 0, denominator +1
        m = scores.amax(dim=-1, keepdim=True).clamp_min(0.0)
        p = torch.nan_to_num(torch.exp(scores - m), nan=0.0)
        denom = 1.0 + p.sum(dim=-1, keepdim=True)
        acc = torch.einsum("mhk,kd->mhd", p[..., :s2], v_ori)
        if v_extra is not None:
            acc = acc + torch.einsum("mhk,mkd->mhd", p[..., s2:], v_extra)
        out[begin:end] = (acc / denom).to(q.dtype)
    return out
