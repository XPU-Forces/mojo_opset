"""Chunked SWA oracle retained from the original Mojo accuracy tests."""

from typing import Optional
from typing import Tuple

import torch


def _generate_window_mask_chunk(
    q_start: int,
    q_end: int,
    kv_seq_len: int,
    kv_computed_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    q_arange = torch.arange(q_start, q_end, device=device)
    kv_arange = torch.arange(0, kv_seq_len, device=device)
    causal_mask = (q_arange[:, None] + kv_computed_len) >= kv_arange[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (q_arange[:, None] + kv_computed_len <= kv_arange[None, :] + local_window_size)
            if local_window_size is not None
            else False
        )
        global_window_mask = (kv_arange < global_window_size)[None, :] if global_window_size is not None else False
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask
    return mask


def _chunked_swa_torch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    output_f32: bool = False,
    q_chunk_size: int = 1024,
) -> Tuple[torch.Tensor, ...]:
    total_q_len, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o_f32 = torch.empty_like(q, dtype=torch.float32)
    softmax_lse = torch.empty((n_q_heads, total_q_len), dtype=torch.float32, device=q.device)
    bsz = cu_q_lens.shape[0] - 1

    for i in range(bsz):
        q_batch_start = cu_q_lens[i].item()
        q_batch_end = cu_q_lens[i + 1].item()
        kv_batch_start = cu_total_seq_lens[i].item()
        kv_batch_end = cu_total_seq_lens[i + 1].item()
        q_seq_len = q_batch_end - q_batch_start
        kv_seq_len = kv_batch_end - kv_batch_start
        kv_computed_len = kv_seq_len - q_seq_len

        k_i = k[kv_batch_start:kv_batch_end]
        v_i = v[kv_batch_start:kv_batch_end]

        k_i_T = k_i.permute(1, 2, 0)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_T = k_i_T.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                k_i_T = k_i_T.repeat_interleave(n_q_heads // n_kv_heads, dim=0)

        v_i_perm = v_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i_perm = v_i_perm.repeat((n_q_heads // n_kv_heads, 1, 1))
            else:
                v_i_perm = v_i_perm.repeat_interleave(n_q_heads // n_kv_heads, dim=0)

        for qc_start in range(0, q_seq_len, q_chunk_size):
            qc_end = min(qc_start + q_chunk_size, q_seq_len)

            q_chunk = q[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)

            s_chunk = torch.bmm(q_chunk, k_i_T).float() * softmax_scale

            if is_causal:
                s_mask = _generate_window_mask_chunk(
                    qc_start,
                    qc_end,
                    kv_seq_len,
                    kv_computed_len,
                    local_window_size,
                    global_window_size,
                    device=s_chunk.device,
                )
                s_chunk = torch.where(s_mask, s_chunk, float("-inf"))

            m_chunk = torch.max(s_chunk, dim=-1, keepdim=True).values
            s_chunk = s_chunk - m_chunk
            p_chunk = torch.exp(s_chunk)
            l_chunk = torch.sum(p_chunk, dim=-1, keepdim=True)
            p_chunk = p_chunk.to(v.dtype)

            o_chunk = torch.bmm(p_chunk, v_i_perm).float()
            o_chunk = o_chunk / l_chunk
            o_chunk = o_chunk.permute(1, 0, 2)
            o_f32[q_batch_start + qc_start : q_batch_start + qc_end] = o_chunk

            lse_chunk = m_chunk + torch.log(l_chunk)
            softmax_lse[:, q_batch_start + qc_start : q_batch_start + qc_end] = lse_chunk.squeeze(-1)

    o = o_f32.to(q.dtype)
    if output_f32:
        return o, softmax_lse, o_f32
    else:
        return o, softmax_lse


def _chunked_swa_torch_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    q_chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.sum(o.float() * do.float(), dim=-1)
    bsz = cu_q_lens.shape[0] - 1
    group_size = n_q_heads // n_kv_heads if n_q_heads != n_kv_heads else 1

    for i in range(bsz):
        q_batch_start = cu_q_lens[i].item()
        q_batch_end = cu_q_lens[i + 1].item()
        kv_batch_start = cu_total_seq_lens[i].item()
        kv_batch_end = cu_total_seq_lens[i + 1].item()
        q_seq_len = q_batch_end - q_batch_start
        kv_seq_len = kv_batch_end - kv_batch_start
        kv_computed_len = kv_seq_len - q_seq_len

        k_i = k[kv_batch_start:kv_batch_end]
        v_i = v[kv_batch_start:kv_batch_end]

        k_i_perm = k_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                k_i_expanded = k_i_perm.repeat((group_size, 1, 1))
            else:
                k_i_expanded = k_i_perm.repeat_interleave(group_size, dim=0)
        else:
            k_i_expanded = k_i_perm

        v_i_perm = v_i.permute(1, 0, 2)
        if n_q_heads != n_kv_heads:
            if gqa_interleave:
                v_i_expanded = v_i_perm.repeat((group_size, 1, 1))
            else:
                v_i_expanded = v_i_perm.repeat_interleave(group_size, dim=0)
        else:
            v_i_expanded = v_i_perm

        dk_i = torch.zeros((kv_seq_len, n_kv_heads, head_dim), dtype=k.dtype, device=k.device)
        dv_i = torch.zeros((kv_seq_len, n_kv_heads, head_dim), dtype=v.dtype, device=v.device)

        for qc_start in range(0, q_seq_len, q_chunk_size):
            qc_end = min(qc_start + q_chunk_size, q_seq_len)

            q_chunk = q[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)
            do_chunk = do[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0, 2)

            s_chunk = torch.bmm(q_chunk, k_i_expanded.mT).float() * softmax_scale

            if is_causal:
                s_mask = _generate_window_mask_chunk(
                    qc_start,
                    qc_end,
                    kv_seq_len,
                    kv_computed_len,
                    local_window_size,
                    global_window_size,
                    device=s_chunk.device,
                )
                s_chunk = torch.where(s_mask, s_chunk, float("-inf"))

            lse_chunk = softmax_lse[:, q_batch_start + qc_start : q_batch_start + qc_end]
            p_chunk = torch.exp(s_chunk - lse_chunk.unsqueeze(-1))

            dp_chunk = torch.bmm(do_chunk, v_i_expanded.mT).float()
            delta_chunk = delta[q_batch_start + qc_start : q_batch_start + qc_end].permute(1, 0).unsqueeze(-1)
            ds_chunk = p_chunk * (dp_chunk - delta_chunk)
            ds_chunk = ds_chunk * softmax_scale
            ds_chunk = ds_chunk.to(do_chunk.dtype)
            p_chunk = p_chunk.to(do_chunk.dtype)

            dq_chunk = torch.bmm(ds_chunk, k_i_expanded)
            dq[q_batch_start + qc_start : q_batch_start + qc_end] = dq_chunk.permute(1, 0, 2)

            if n_q_heads != n_kv_heads:
                if gqa_interleave:
                    ds_reduced = ds_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    q_reduced = q_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    p_reduced = p_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                    do_reduced = do_chunk.unflatten(0, (group_size, n_kv_heads)).permute(1, 0, 2, 3)
                else:
                    ds_reduced = ds_chunk.unflatten(0, (n_kv_heads, group_size))
                    q_reduced = q_chunk.unflatten(0, (n_kv_heads, group_size))
                    p_reduced = p_chunk.unflatten(0, (n_kv_heads, group_size))
                    do_reduced = do_chunk.unflatten(0, (n_kv_heads, group_size))

                ds_reduced = ds_reduced.flatten(1, 2)
                q_reduced = q_reduced.flatten(1, 2)
                p_reduced = p_reduced.flatten(1, 2)
                do_reduced = do_reduced.flatten(1, 2)
            else:
                ds_reduced = ds_chunk
                q_reduced = q_chunk
                p_reduced = p_chunk
                do_reduced = do_chunk

            dk_i += torch.bmm(ds_reduced.mT, q_reduced).permute(1, 0, 2)
            dv_i += torch.bmm(p_reduced.mT, do_reduced).permute(1, 0, 2)

        dk[kv_batch_start:kv_batch_end] = dk_i
        dv[kv_batch_start:kv_batch_end] = dv_i

    return dq, dk, dv
