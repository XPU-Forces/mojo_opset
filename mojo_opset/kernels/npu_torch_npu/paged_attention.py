"""torch_npu fused-attention leaves with explicit layout and capability boundaries."""

from typing import Optional

import torch
import torch_npu


def _heads(query, kv_heads, layout, *, reverse=False, axis=1):
    if layout == "AABB" or query.shape[axis] == kv_heads:
        return query
    query = query.movedim(axis, -2)
    groups = query.shape[-2] // kv_heads
    order = (kv_heads, groups) if reverse else (groups, kv_heads)
    query = query.reshape(*query.shape[:-2], *order, query.shape[-1])
    return query.transpose(-3, -2).flatten(-3, -2).movedim(-2, axis).contiguous()


def _check(query, cache, *, paged=True):
    if query.shape[-1] % 128:
        raise NotImplementedError("torch_npu fused attention requires head_dim divisible by 128")
    if paged and (cache.shape[2] % 128 or cache.shape[2] > 512):
        raise NotImplementedError("torch_npu paged GQA requires block_size 128, 256, 384, or 512")


@torch.library.custom_op("mojo_npu_torch_npu::prefill_gqa_infer", mutates_args=())
def prefill_gqa_infer_fwd(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    softmax_scale: Optional[float],
    is_causal: bool,
    gqa_layout: str,
) -> torch.Tensor:
    _check(query, k_cache, paged=False)
    if not is_causal:
        raise NotImplementedError("prefill_gqa_infer retains the original causal-only contract")
    heads, kv_heads, length = query.shape[1], k_cache.shape[1], query.shape[2]
    mask = torch.ones((length, length), dtype=torch.bool, device=query.device).triu(1)
    # FusedInferAttentionScore V5 requires 3D/4D masks for sparse_mode=0.
    # Singleton batch/head dimensions preserve the shared causal mask.
    mask = mask[None, None]
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=_heads(query, kv_heads, gqa_layout),
        key=k_cache,
        value=v_cache,
        num_heads=heads,
        num_key_value_heads=kv_heads,
        input_layout="BNSD",
        scale=softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5,
        atten_mask=mask,
        sparse_mode=0,
        pre_tokens=65535,
        next_tokens=0,
    )
    return _heads(output, kv_heads, gqa_layout, reverse=True).transpose(1, 2).contiguous()


@prefill_gqa_infer_fwd.register_fake
def _prefill_fake(query, k_cache, v_cache, cu_q_lens, softmax_scale, is_causal, gqa_layout):
    return query.new_empty((query.shape[0], query.shape[2], query.shape[1], query.shape[3]))


@torch.library.custom_op("mojo_npu_torch_npu::paged_prefill_gqa_infer", mutates_args=())
def _paged_prefill_gqa(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: Optional[float],
    cu_total_seq_lens: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    max_q_len: Optional[int],
    max_total_seq_len: Optional[int],
    is_causal: bool,
    gqa_layout: str,
) -> torch.Tensor:
    _check(query, key_cache)
    if cu_total_seq_lens is not None:
        raise NotImplementedError("torch_npu paged TND prefill does not support cached-prefix lengths")
    if not is_causal or mask is not None:
        raise NotImplementedError("torch_npu paged TND prefill requires causal attention without a custom mask")
    kv_heads = key_cache.shape[1]
    compress_mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device).triu(1)
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=_heads(query, kv_heads, gqa_layout),
        key=key_cache,
        value=value_cache,
        atten_mask=compress_mask,
        block_table=block_tables,
        input_layout="TND",
        block_size=key_cache.shape[2],
        actual_seq_lengths=cu_q_lens[1:],
        actual_seq_lengths_kv=cu_q_lens[1:] - cu_q_lens[:-1],
        num_key_value_heads=kv_heads,
        num_heads=query.shape[1],
        scale=softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5,
        sparse_mode=3,
    )
    return _heads(output, kv_heads, gqa_layout, reverse=True).contiguous()


@_paged_prefill_gqa.register_fake
def _paged_prefill_fake(
    query,
    key_cache,
    value_cache,
    cu_q_lens,
    block_tables,
    softmax_scale,
    cu_total_seq_lens,
    mask,
    max_q_len,
    max_total_seq_len,
    is_causal,
    gqa_layout,
):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def paged_prefill_gqa_infer_fwd(
    query,
    key_cache,
    value_cache,
    cu_q_lens,
    block_tables,
    softmax_scale,
    cu_total_seq_lens,
    mask,
    max_q_len,
    max_total_seq_len,
    is_causal,
    gqa_layout,
    scheduler_metadata=None,
):
    return _paged_prefill_gqa(
        query,
        key_cache,
        value_cache,
        cu_q_lens,
        block_tables,
        softmax_scale,
        cu_total_seq_lens,
        mask,
        max_q_len,
        max_total_seq_len,
        is_causal,
        gqa_layout,
    )


@torch.library.custom_op("mojo_npu_torch_npu::paged_decode_gqa_infer", mutates_args=())
def paged_decode_gqa_infer_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: Optional[float],
    mask: Optional[torch.Tensor],
    max_total_seq_len: Optional[int],
    is_causal: bool,
    gqa_layout: str,
) -> torch.Tensor:
    _check(query, key_cache)
    if mask is not None:
        raise NotImplementedError("torch_npu paged decode does not support a custom mask")
    kv_heads = key_cache.shape[1]
    output, _ = torch_npu.npu_fused_infer_attention_score(
        _heads(query, kv_heads, gqa_layout).unsqueeze(2),
        key_cache,
        value_cache,
        input_layout="BNSD",
        block_table=block_tables,
        block_size=key_cache.shape[2],
        num_heads=query.shape[1],
        num_key_value_heads=kv_heads,
        actual_seq_lengths=torch.ones_like(total_seq_lens),
        actual_seq_lengths_kv=total_seq_lens,
        scale=softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5,
    )
    return _heads(output.squeeze(2), kv_heads, gqa_layout, reverse=True).contiguous()


@paged_decode_gqa_infer_fwd.register_fake
def _paged_decode_fake(
    query,
    key_cache,
    value_cache,
    total_seq_lens,
    block_tables,
    softmax_scale,
    mask,
    max_total_seq_len,
    is_causal,
    gqa_layout,
):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_torch_npu::swa_fused_attention", mutates_args=())
def _swa_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    q_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        input_layout="BNSD",
        key=key_cache,
        value=value_cache,
        block_table=block_table,
        block_size=key_cache.shape[2],
        actual_seq_lengths=q_lens,
        actual_seq_lengths_kv=kv_lens,
        num_key_value_heads=key_cache.shape[1],
        num_heads=query.shape[1],
        scale=softmax_scale,
        atten_mask=mask,
        sparse_mode=0,
    )
    return output


@_swa_attention.register_fake
def _swa_attention_fake(query, key_cache, value_cache, q_lens, kv_lens, block_table, mask, softmax_scale):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def _swa(
    query,
    key_cache,
    value_cache,
    q_lens,
    kv_lens,
    block_table,
    softmax_scale,
    is_causal,
    gqa_layout,
    global_window_size,
    local_window_size,
    cu_q_lens=None,
    max_q_len=None,
    max_total_seq_len=None,
):
    _check(query, key_cache, paged=False)
    kv_heads, heads = key_cache.shape[1], query.shape[1]
    q_lengths, kv_lengths = q_lens.tolist(), kv_lens.tolist()
    max_q = max_q_len if max_q_len is not None else max(q_lengths)
    max_kv = max(block_table.shape[1] * key_cache.shape[2], max_total_seq_len or max(kv_lengths))
    if cu_q_lens is None:
        padded = query.unsqueeze(2)
    else:
        padded = query.new_zeros((len(q_lengths), heads, max_q, query.shape[-1]))
        offset = 0
        for i, length in enumerate(q_lengths):
            padded[i, :, :length] = query[offset : offset + length].transpose(0, 1)
            offset += length
    mask = torch.ones((len(q_lengths), 1, max_q, max_kv), dtype=torch.bool, device=query.device)
    for i, (qlen, kvlen) in enumerate(zip(q_lengths, kv_lengths)):
        qpos = torch.arange(qlen, device=query.device)[:, None] + kvlen - qlen
        kpos = torch.arange(kvlen, device=query.device)[None, :]
        visible = torch.ones((qlen, kvlen), dtype=torch.bool, device=query.device)
        if is_causal:
            visible &= qpos >= kpos
            if local_window_size is not None or global_window_size is not None:
                window = torch.zeros_like(visible)
                if local_window_size is not None:
                    window |= qpos <= kpos + local_window_size
                if global_window_size is not None:
                    window |= kpos < global_window_size
                visible &= window
        mask[i, 0, :qlen, :kvlen] = ~visible
    output = _swa_attention(
        _heads(padded, kv_heads, gqa_layout),
        key_cache,
        value_cache,
        q_lens,
        kv_lens,
        block_table,
        mask,
        softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5,
    )
    output = _heads(output, kv_heads, gqa_layout, reverse=True)
    if cu_q_lens is None:
        return output.squeeze(2).contiguous()
    return torch.cat([output[i, :, :length].transpose(0, 1) for i, length in enumerate(q_lengths)]).contiguous()


def paged_prefill_swa_infer_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: Optional[float],
    cu_total_seq_lens: Optional[torch.Tensor],
    max_q_len: Optional[int],
    max_total_seq_len: Optional[int],
    is_causal: bool,
    gqa_layout: str,
    global_window_size: Optional[int],
    local_window_size: Optional[int],
) -> torch.Tensor:
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    kv_lens = q_lens if cu_total_seq_lens is None else cu_total_seq_lens[1:] - cu_total_seq_lens[:-1]
    return _swa(
        query,
        key_cache,
        value_cache,
        q_lens,
        kv_lens,
        block_table,
        softmax_scale,
        is_causal,
        gqa_layout,
        global_window_size,
        local_window_size,
        cu_q_lens,
        max_q_len,
        max_total_seq_len,
    )


def paged_decode_swa_infer_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: Optional[float],
    max_total_seq_len: Optional[int],
    is_causal: bool,
    gqa_layout: str,
    global_window_size: Optional[int],
    local_window_size: Optional[int],
) -> torch.Tensor:
    return _swa(
        query,
        key_cache,
        value_cache,
        torch.ones_like(total_seq_lens),
        total_seq_lens,
        block_table,
        softmax_scale,
        is_causal,
        gqa_layout,
        global_window_size,
        local_window_size,
        max_total_seq_len=max_total_seq_len,
    )
