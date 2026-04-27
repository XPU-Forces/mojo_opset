"""
ILU: paged prefill GQA — KV gather + GQA expand in PyTorch; causal attention in Triton.
Aligned with MojoPagedPrefillGQA (mask j <= i + (kv_seq_len - q_seq_len)).

Paged decode (``paged_attention_decode_impl``)
-----------------------------------------------
Gathers paged KV to dense ``[T, H, D]`` per batch row, GQA-expands, then
``_launch_causal_attn_triton`` with ``q_seq_len=1`` (same as prefill numerics).
Works for any KV ``block_size`` (e.g. 32, 128, 1024).

"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import libentry


@libentry()
@triton.jit
def _paged_prefill_causal_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_j,
    stride_k_h,
    stride_k_d,
    stride_v_j,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    Tq: tl.constexpr,
    Tk: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    sm_scale,
    diag_off: tl.constexpr,
    OUT_T: tl.constexpr,
):
    """
    For each (query row t_i, head h): softmax over key j with causal mask
    j <= t_i + diag_off, diag_off = kv_seq_len - q_seq_len.

    D_PAD is next power of 2 >= D (ILU Triton needs power-of-2 vector tiles for arange/zeros).
    """
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)
    total = Tq * H

    offs_d = tl.arange(0, D_PAD)
    mask_d = offs_d < D

    for flat in tl.range(pid, total, pnum):
        t_i = flat // H
        h = flat % H

        q_base = t_i * stride_q_t + h * stride_q_h
        q_vec = tl.load(q_ptr + q_base + offs_d * stride_q_d, mask=mask_d, other=0.0).to(tl.float32)

        m_max = tl.full((), -float("inf"), tl.float32)
        for j in range(Tk):
            allowed = j <= t_i + diag_off
            k_base = j * stride_k_j + h * stride_k_h
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            m_max = tl.maximum(m_max, s)

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((D_PAD,), dtype=tl.float32)
        for j in range(Tk):
            allowed = j <= t_i + diag_off
            k_base = j * stride_k_j + h * stride_k_h
            v_base = j * stride_v_j + h * stride_v_h
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + v_base + offs_d * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            p = tl.exp(s - m_max)
            denom = denom + p
            acc = acc + p * v_vec

        if Tk > 0:
            out_vec = acc / denom
        else:
            out_vec = acc
        o_base = t_i * stride_o_t + h * stride_o_h
        tl.store(out_ptr + o_base + offs_d * stride_o_d, out_vec.to(OUT_T), mask=mask_d)


def _launch_causal_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    q_seq_len: int,
    kv_seq_len: int,
) -> None:
    tq, h, d = q.shape
    tk = k.shape[0]
    assert k.shape == (tk, h, d) and v.shape == (tk, h, d)
    diag_off = kv_seq_len - q_seq_len

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    total_tasks = tq * h
    block = 256
    grid = (triton.cdiv(total_tasks, block),)

    d_pad = triton.next_power_of_2(d)

    _paged_prefill_causal_attn_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        Tq=tq,
        Tk=tk,
        H=h,
        D=d,
        D_PAD=d_pad,
        sm_scale=float(sm_scale),
        diag_off=int(diag_off),
        OUT_T=out_t,
    )


def paged_attention_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    aux_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del aux_mask

    total_q_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    outputs = torch.zeros(total_q_tokens, num_q_heads, head_dim, dtype=q.dtype, device=q.device)

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    batch_size = len(q_lens)

    for i in range(batch_size):
        q_seq_len = q_lens[i].item()
        start_loc = cu_seqlens_q[i].item()
        end_loc = cu_seqlens_q[i + 1].item()
        q_batch = q[start_loc:end_loc].contiguous()
        if seqlens_kv is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = seqlens_kv[i].item()

        num_blocks_for_seq = (kv_seq_len + block_size - 1) // block_size
        k_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
        v_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)

        for j in range(num_blocks_for_seq):
            physical_block_id = block_tables[i, j].item()

            start_pos_in_seq = j * block_size
            end_pos_in_seq = min(start_pos_in_seq + block_size, kv_seq_len)
            tokens_in_block = end_pos_in_seq - start_pos_in_seq

            k_slice = key_cache[physical_block_id, :, :tokens_in_block, :]
            k_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = k_slice.permute(1, 0, 2)

            v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]
            v_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = v_slice.permute(1, 0, 2)

        if num_q_heads != num_kv_heads:
            g = num_q_heads // num_kv_heads
            # repeat: head dim becomes [kv_0..kv_{K-1}] tiled g times → aligns with decode
            #   kv_head_id = q_head_id % num_kv_heads when gqa_interleave / GQA_INTERLEAVE.
            # repeat_interleave: each kv head repeated g times in order → aligns with
            #   kv_head_id = q_head_id // g when not interleaved. See module docstring.
            if gqa_interleave:
                k_expanded = k_unpadded.repeat((1, g, 1))
                v_expanded = v_unpadded.repeat((1, g, 1))
            else:
                k_expanded = k_unpadded.repeat_interleave(g, dim=1)
                v_expanded = v_unpadded.repeat_interleave(g, dim=1)
        else:
            k_expanded = k_unpadded
            v_expanded = v_unpadded

        k_expanded = k_expanded.contiguous()
        v_expanded = v_expanded.contiguous()
        out_slice = torch.empty_like(q_batch)
        _launch_causal_attn_triton(q_batch, k_expanded, v_expanded, out_slice, softmax_scale, q_seq_len, kv_seq_len)
        outputs[start_loc:end_loc] = out_slice

    return outputs


def _paged_decode_gather_and_causal(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """ILU paged decode: unpage KV + causal attention (one query token per batch row)."""
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    o = torch.zeros_like(q)

    for i in range(batch_size):
        kv_seq_len = int(seqlens[i].item())
        if kv_seq_len <= 0 or int(block_tables[i, 0].item()) < 0:
            continue
        q_batch = q[i : i + 1].contiguous()
        num_blocks_for_seq = (kv_seq_len + block_size - 1) // block_size

        k_unpadded = torch.zeros(
            kv_seq_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device
        )
        v_unpadded = torch.zeros(
            kv_seq_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device
        )

        for j in range(num_blocks_for_seq):
            physical_block_id = int(block_tables[i, j].item())
            if physical_block_id < 0:
                break
            start_pos_in_seq = j * block_size
            end_pos_in_seq = min(start_pos_in_seq + block_size, kv_seq_len)
            tokens_in_block = end_pos_in_seq - start_pos_in_seq

            k_slice = key_cache[physical_block_id, :, :tokens_in_block, :]
            k_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = k_slice.permute(1, 0, 2)

            v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]
            v_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = v_slice.permute(1, 0, 2)

        if num_q_heads != num_kv_heads:
            g = num_q_heads // num_kv_heads
            if gqa_interleave:
                k_expanded = k_unpadded.repeat((1, g, 1))
                v_expanded = v_unpadded.repeat((1, g, 1))
            else:
                k_expanded = k_unpadded.repeat_interleave(g, dim=1)
                v_expanded = v_unpadded.repeat_interleave(g, dim=1)
        else:
            k_expanded = k_unpadded
            v_expanded = v_unpadded

        k_expanded = k_expanded.contiguous()
        v_expanded = v_expanded.contiguous()
        out_slice = torch.empty_like(q_batch)
        _launch_causal_attn_triton(
            q_batch, k_expanded, v_expanded, out_slice, softmax_scale, 1, kv_seq_len
        )
        o[i] = out_slice[0]

    return o


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Paged KV decode attention (one query step per batch row)."""
    _, _, head_dim = q.shape
    _, _, _, head_dim_cache = key_cache.shape

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    return _paged_decode_gather_and_causal(
        q,
        key_cache,
        value_cache,
        seqlens,
        block_tables,
        gqa_interleave,
        float(softmax_scale),
    )
