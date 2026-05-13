import math
from typing import Optional

import pytest
import torch

from mojo_opset import MojoPagedDecodeQuantGQA
from mojo_opset import MojoPagedDecodeQuantSWA
from mojo_opset import MojoPagedPrefillQuantGQA
from mojo_opset import MojoPagedPrefillQuantSWA
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented

def _quantize_query(
    query: torch.Tensor,
    query_dtype: torch.Tensor,
):
    if query_dtype == torch.int8:
        int_dtype = torch.int8
        qmax = 2 ** (8 - 1) - 1
        qmin = -(2 ** (8 - 1))
    else:
        assert query_dtype == torch.bfloat16
        return query.to(query_dtype), None
    
    query_f = query.float()
    amax = query_f.abs().amax(dim=-1, keepdim=True)  # -> [num_tokens, num_q_heads, 1]
    qscale = (amax / qmax).clamp(min=1e-5)
    quant = torch.round(query_f / qscale).clamp(qmin, qmax).to(int_dtype)
    return quant, qscale.to(torch.bfloat16)

def _quantize_kv_cache(
    cache: torch.Tensor,  # [n_blocks, n_kv_heads, block_size, head_dim] in float dtype
    context_dtype: torch.dtype,
):
    """Per-channel dynamic quantize a float KV cache along the head_dim axis.

    Returns:
        quant_cache: integer tensor with dtype matching `context_dtype`, same shape as input.
        qscale: per-channel scale of shape (n_kv_heads, head_dim) in the input dtype.
    """
    if context_dtype == torch.int8:
        int_dtype = torch.int8
        qmax = 2 ** (8 - 1) - 1
        qmin = -(2 ** (8 - 1))
    else:
        assert False, f"Context dtype {context_dtype} not supported yet"
    # amax over (n_blocks, block_size) per (head, dim) channel
    cache_f = cache.float()
    amax = cache_f.abs().amax(dim=(0, 2))  # -> [n_kv_heads, head_dim]
    qscale = (amax / qmax).clamp(min=1e-5)
    quant = torch.round(cache_f / qscale.unsqueeze(0).unsqueeze(2)).clamp(qmin, qmax).to(int_dtype)
    return quant, qscale.to(torch.bfloat16)


def generate_paged_decode_quant_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    """Generate decode test data; KV cache kept in float dtype and quantized inside the test."""
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)

    if max_seq_len > 0:
        total_seq_lens = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int32)
        total_seq_lens = torch.clamp(total_seq_lens, min=1)
    else:
        total_seq_lens = torch.randperm(batch_size, dtype=torch.int32)

    max_total_seq_len = total_seq_lens.max().item()
    max_num_blocks_per_seq = (max_total_seq_len + block_size - 1) // block_size
    total_blocks_needed = int(
        torch.div(total_seq_lens + block_size - 1, block_size, rounding_mode="floor").sum().item()
    )

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    # float KV cache; will be quantized inside the test according to context_dtype
    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = total_seq_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len


def generate_paged_prefill_quant_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    """Generate prefill test data; KV cache kept in float dtype and quantized inside the test."""
    if max_q_len > 0:
        q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
        q_lens = torch.clamp(q_lens, min=1)
    else:
        q_lens = torch.randperm(batch_size, dtype=torch.int32)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    if max_kv_computed_len <= 0:
        cu_total_seq_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(
            max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32
        )
        kv_lens = q_lens + kv_cache_lens
        kv_lens = torch.where(q_lens > 0, kv_lens, torch.zeros_like(kv_lens))
        cu_total_seq_lens = torch.cat(
            [torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)]
        )

    total_q_tokens = cu_q_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = max(1, (kv_lens.max().item() + block_size - 1) // block_size)
    total_blocks_needed = int(torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        if num_blocks_for_seq == 0:
            continue
        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens


# ===========================================================================
# MojoPagedPrefillQuantGQA
# ===========================================================================

test_configs_prefill_quant_gqa = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 512, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 2, 128, 1024, 0, 128, torch.bfloat16, "M_BF16_GROUP"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens",
    [
        pytest.param(
            *generate_paged_prefill_quant_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_quant_gqa
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@pytest.mark.parametrize("query_dtype, context_dtype, compute_dtype", 
    [
        (torch.bfloat16, torch.int8, torch.bfloat16),
        (torch.bfloat16, torch.int8, torch.int8),
    ]
)
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_quant_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    gqa_layout: str,
    query_dtype: torch.dtype,
    context_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):
    query_q, _query_scale = _quantize_query(query, query_dtype)

    k_cache_q, key_scale = _quantize_kv_cache(k_cache, context_dtype)
    v_cache_q, value_scale = _quantize_kv_cache(v_cache, context_dtype)

    op = MojoPagedPrefillQuantGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )
    op_ref = MojoPagedPrefillQuantGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)
    seqlens_kv = None if cu_total_seq_lens is None else cu_total_seq_lens[1:] - cu_total_seq_lens[:-1]

    op.forward_diff_with(
        op_ref,
        query_q,
        k_cache_q,
        key_scale,
        v_cache_q,
        value_scale,
        cu_q_lens,
        block_tables,
        softmax_scale=softmax_scale,
        seqlens_kv=seqlens_kv,
        atol=5e-2 if query.dtype != torch.float32 else 1e-5,
        rtol=5e-2 if query.dtype != torch.float32 else 1e-6,
        ptol=0.90,
    )


# ===========================================================================
# MojoPagedDecodeQuantGQA
# ===========================================================================

test_configs_decode_quant_gqa = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (4, 8, 1, 128, 8192, 1024, torch.bfloat16, "M_BF16_LONG"),
    (4, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (4, 8, 2, 128, 2048, 128, torch.bfloat16, "M_BF16_GROUP"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len",
    [
        pytest.param(
            *generate_paged_decode_quant_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode_quant_gqa
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@pytest.mark.parametrize("query_dtype, context_dtype, compute_dtype", 
    [
        (torch.bfloat16, torch.int8, torch.bfloat16),
        (torch.bfloat16, torch.int8, torch.int8),
    ]
)
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_quant_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_total_seq_len: int,
    gqa_layout: str,
    query_dtype: torch.dtype,
    context_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):
    query_q, query_scale = _quantize_query(query, query_dtype)

    k_cache_q, key_scale = _quantize_kv_cache(k_cache, context_dtype)
    v_cache_q, value_scale = _quantize_kv_cache(v_cache, context_dtype)

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    op = MojoPagedDecodeQuantGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedDecodeQuantGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    op.forward_diff_with(
        op_ref,
        query_q,
        query_scale,
        k_cache_q,
        key_scale,
        v_cache_q,
        value_scale,
        total_seq_lens,
        block_tables,
        softmax_scale=softmax_scale,
        atol=atol,
        rtol=rtol,
    )


# ===========================================================================
# MojoPagedPrefillQuantSWA
# ===========================================================================

test_configs_prefill_quant_swa = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 2048, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 256, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 2, 128, 2048, 0, 1024, torch.bfloat16, "M_BF16_GROUP1"),
    (2, 24, 8, 128, 1024, 1024, 1024, torch.bfloat16, "M_BF16_GROUP2"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens",
    [
        pytest.param(
            *generate_paged_prefill_quant_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_quant_swa
    ],
)
@pytest.mark.parametrize(
    "gqa_layout, global_window, local_window",
    [
        ("ABAB", 4, 255),
        ("AABB", 4, 1023),
    ],
)
@pytest.mark.parametrize("query_dtype, context_dtype, compute_dtype", 
    [
        (torch.bfloat16, torch.int8, torch.bfloat16),
        (torch.bfloat16, torch.int8, torch.int8),
    ]
)
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_quant_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    gqa_layout: str,
    global_window: int,
    local_window: int,
    query_dtype: torch.dtype,
    context_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):
    query_q, query_scale = _quantize_query(query, query_dtype)
    k_cache_q, key_scale = _quantize_kv_cache(k_cache, context_dtype)
    v_cache_q, value_scale = _quantize_kv_cache(v_cache, context_dtype)

    op = MojoPagedPrefillQuantSWA(
        is_causal=True,
        gqa_layout=gqa_layout,  
        local_window_size=local_window,
        global_window_size=global_window,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedPrefillQuantSWA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        local_window_size=local_window,
        global_window_size=global_window,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    op.forward_diff_with(
        op_ref,
        query_q,
        query_scale,
        k_cache_q,
        key_scale,
        v_cache_q,
        value_scale,
        cu_q_lens,
        block_tables,
        softmax_scale=softmax_scale,
        cu_total_seq_lens=cu_total_seq_lens,
        atol=2e-2 if query.dtype != torch.float32 else 1e-5,
        rtol=2e-2 if query.dtype != torch.float32 else 1e-6,
        # int8 compute can have a few round-boundary outliers across millions of elements.
        ptol=0.99999 if compute_dtype == torch.int8 else 1.0,
    )


# ===========================================================================
# MojoPagedDecodeQuantSWA
# ===========================================================================

test_configs_decode_quant_swa = [
    (4, 16, 4, 128, 1024, 512, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 2048, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 4096, 128, torch.bfloat16, "M_BF16_LONG"),
    (2, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 2, 128, 2048, 1024, torch.bfloat16, "M_BF16_GROUP1"),
    (2, 24, 8, 128, 2048, 1024, torch.bfloat16, "M_BF16_GROUP2"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len",
    [
        pytest.param(
            *generate_paged_decode_quant_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode_quant_swa
    ],
)
@pytest.mark.parametrize(
    "gqa_layout, global_window, local_window",
    [
        ("ABAB", 4, 255),
        ("AABB", 4, 1023),
    ],
)
@pytest.mark.parametrize("query_dtype, context_dtype, compute_dtype", 
    [
        (torch.bfloat16, torch.int8, torch.bfloat16),
        (torch.bfloat16, torch.int8, torch.int8),
    ]
)
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_quant_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_total_seq_len: int,
    gqa_layout: str,
    global_window: int,
    local_window: int,
    query_dtype: torch.dtype,
    context_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):
    query_q, query_scale = _quantize_query(query, query_dtype)
    k_cache_q, key_scale = _quantize_kv_cache(k_cache, context_dtype)
    v_cache_q, value_scale = _quantize_kv_cache(v_cache, context_dtype)

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    op = MojoPagedDecodeQuantSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedDecodeQuantSWA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    op.forward_diff_with(
        op_ref,
        query_q,
        query_scale,
        k_cache_q,
        key_scale,
        v_cache_q,
        value_scale,
        total_seq_lens,
        block_tables,
        softmax_scale=softmax_scale,
        atol=atol,
        rtol=rtol,
    )
