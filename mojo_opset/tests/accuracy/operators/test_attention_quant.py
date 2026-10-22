import math
from typing import Optional, Sequence, Tuple, Union

import pytest
import torch

from mojo_opset.experimental import MojoPagedDecodeGQAWithKVDequant
from mojo_opset.experimental import MojoPagedDecodeSWAWithKVDequant
from mojo_opset import MojoPagedPrefillGQAFp8
from mojo_opset import MojoPagedDecodeGQAFp8
from mojo_opset import MojoPagedDecodeSWAFp8
from mojo_opset import MojoPagedPrefillSWAFp8
from mojo_opset.experimental import MojoPagedPrefillGQAWithKVDequant
from mojo_opset.experimental import MojoPagedPrefillSWAWithKVDequant
from mojo_opset.experimental import MojoPagedPrefillSageGQA
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


def _make_varlen_positive_int32(lengths: torch.Tensor, upper_bound: int) -> torch.Tensor:
    """Make per-batch lengths explicitly varlen while keeping them positive."""
    lengths = lengths.to(torch.int32).clone()
    if lengths.numel() <= 1 or upper_bound <= 1:
        return torch.clamp(lengths, min=1)

    offsets = torch.arange(lengths.numel(), dtype=torch.int32)
    span = max(upper_bound - 1, 1)
    lengths = ((lengths + offsets) % span) + 1
    return lengths


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
        total_seq_lens = _make_varlen_positive_int32(total_seq_lens, max_seq_len)
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
        q_lens = _make_varlen_positive_int32(q_lens, max_q_len)
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
        kv_cache_lens = _make_varlen_positive_int32(kv_cache_lens, max_kv_computed_len)
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

    max_q_len = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
    max_total_seq_len = int(kv_lens.max().item()) if kv_lens.numel() > 0 else 0
    return query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_len, max_total_seq_len

def get_scale_and_quant(
    x: torch.Tensor,
    scale: Optional[torch.Tensor],
    quant_dims,
    q_max: int = 127,
    q_min: int = -128,
    eps: float = 1e-6,
    quant_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scale is None or isinstance(scale, torch.Tensor), \
        f"scale must be None or Tensor, got {type(scale)}"

    if scale is None:
        scale = x.abs().amax(dim=quant_dims, keepdim=True).float() / q_max
        scale = scale.clamp(min=eps)

    q = torch.clamp(torch.round(x.float() / scale), q_min, q_max).to(quant_dtype)
    return q, scale

def per_block_int8(
    x: torch.Tensor,
    xm: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    *,
    blk: int = 16,
    dim1: int = -2,
    dim2: int = -1,
    q_max: int = 127,
    q_min: int = -128,
):
    if xm is not None:
        x = x - xm
    dim1_norm = dim1 % x.ndim
    dim2_norm = dim2 % x.ndim

    seq_len = x.shape[dim1_norm]
    assert seq_len % blk == 0, (
        f"per_block_int8: dim {dim1_norm} (size {seq_len}) must be divisible by blk={blk}"
    )
    num_blocks = seq_len // blk

    # Reshape seq_dim from S into (S // blk, blk) so the inner ``blk`` axis
    # can be reduced independently.
    new_shape = (
        list(x.shape[:dim1_norm])
        + [num_blocks, blk]
        + list(x.shape[dim1_norm + 1:])
    )
    x_reshaped = x.reshape(new_shape)

    q, scale_kd = get_scale_and_quant(
        x_reshaped,
        scale,
        quant_dims=(dim1_norm, dim2_norm),
        q_max=q_max,
        q_min=q_min,
    )

    q = q.reshape(x.shape)
    scale_out = scale_kd.squeeze(dim1_norm + 1).repeat_interleave(num_blocks, dim=dim1_norm)
    return q, scale_out


def per_token_int8(
    x: torch.Tensor,
    xm: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    *,
    q_max: int = 127,
    q_min: int = -128,
):
    if xm is not None:
        x = x - xm
    return get_scale_and_quant(x, scale, quant_dims=-1, q_max=q_max, q_min=q_min)


def per_channel_int8(
    x: torch.Tensor,
    xm: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    *,
    seq_dim: Union[int, Sequence[int]] = 0,
    q_max: int = 127,
    q_min: int = -128,
):
    if xm is not None:
        x = x - xm
    return get_scale_and_quant(x, scale, quant_dims=seq_dim, q_max=q_max, q_min=q_min)

# ===========================================================================
# MojoPagedPrefillGQAWithKVDequant
# ===========================================================================

test_configs_prefill_gqa_with_kv_dequant = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 512, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 2, 128, 1024, 0, 128, torch.bfloat16, "M_BF16_GROUP"),
    (3, 12, 3, 64, 257, 513, 16, torch.bfloat16, "M_BF16_VARLEN_BLK16_D64"),
    (4, 20, 5, 192, 193, 769, 256, torch.bfloat16, "M_BF16_VARLEN_BLK256_D192"),
    (3, 24, 6, 80, 321, 641, 64, torch.bfloat16, "M_BF16_VARLEN_BLK64_D80"),
    (1, 16, 4, 128, 128, 0, 16, torch.bfloat16, "M_BF16_SMALL_BLK16"),
    (2, 24, 6, 128, 255, 129, 32, torch.bfloat16, "M_BF16_H24_VARLEN"),
    (3, 16, 4, 128, 513, 257, 64, torch.bfloat16, "M_BF16_VARLEN_513"),
    (4, 24, 6, 128, 769, 511, 128, torch.bfloat16, "M_BF16_H24_BLK128"),
    (5, 16, 4, 128, 1025, 333, 256, torch.bfloat16, "M_BF16_ODD_KV_333"),
    (6, 24, 6, 128, 1537, 777, 128, torch.bfloat16, "M_BF16_H24_ODD_777"),
    (5, 16, 4, 128, 2049, 1025, 256, torch.bfloat16, "M_BF16_LONG_ODD"),
    (4, 24, 6, 128, 3073, 1537, 256, torch.bfloat16, "M_BF16_H24_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_len, max_total_seq_len",
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
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_gqa_with_kv_dequant
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
def test_paged_prefill_gqa_with_kv_dequant(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    max_q_len: int,
    max_total_seq_len: int,
    gqa_layout: str,
    query_dtype: torch.dtype,
    context_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):
    query_q, query_scale = _quantize_query(query, query_dtype)
    k_cache_q, key_scale = _quantize_kv_cache(k_cache, context_dtype)
    v_cache_q, value_scale = _quantize_kv_cache(v_cache, context_dtype)

    op = MojoPagedPrefillGQAWithKVDequant(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedPrefillGQAWithKVDequant._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
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
        max_q_len=max_q_len,
        max_total_seq_len=max_total_seq_len,
        atol=5e-2 if query.dtype != torch.float32 else 1e-5,
        rtol=5e-2 if query.dtype != torch.float32 else 1e-6,
        ptol=0.90,
    )


# ===========================================================================
# MojoPagedDecodeGQAWithKVDequant
# ===========================================================================

test_configs_decode_gqa_with_kv_dequant = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (4, 8, 1, 128, 8192, 1024, torch.bfloat16, "M_BF16_LONG"),
    (4, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (4, 8, 2, 128, 2048, 128, torch.bfloat16, "M_BF16_GROUP"),
    (5, 12, 3, 64, 257, 16, torch.bfloat16, "M_BF16_VARLEN_BLK16_D64"),
    (3, 20, 5, 192, 1537, 256, torch.bfloat16, "M_BF16_VARLEN_BLK256_D192"),
    (6, 24, 6, 80, 513, 64, torch.bfloat16, "M_BF16_VARLEN_BLK64_D80"),
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
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode_gqa_with_kv_dequant
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
def test_paged_decode_gqa_with_kv_dequant(
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

    op = MojoPagedDecodeGQAWithKVDequant(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedDecodeGQAWithKVDequant._registry.get("torch")(
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
        max_total_seq_len=max_total_seq_len,
        atol=atol,
        rtol=rtol,
    )


# ===========================================================================
# MojoPagedPrefillSWAWithKVDequant
# ===========================================================================

test_configs_prefill_swa_with_kv_dequant = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 2048, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 256, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 2, 128, 2048, 0, 1024, torch.bfloat16, "M_BF16_GROUP1"),
    (2, 24, 8, 128, 1024, 1024, 1024, torch.bfloat16, "M_BF16_GROUP2"),
    (3, 12, 3, 64, 257, 513, 16, torch.bfloat16, "M_BF16_VARLEN_BLK16_D64"),
    (3, 20, 5, 256, 193, 769, 256, torch.bfloat16, "M_BF16_VARLEN_BLK256_D256"),
    (3, 12, 3, 80, 257, 513, 64, torch.bfloat16, "M_BF16_VARLEN_BLK64_D80"),
    (1, 16, 4, 128, 128, 0, 16, torch.bfloat16, "M_BF16_SMALL_BLK16"),
    (2, 24, 6, 128, 255, 129, 32, torch.bfloat16, "M_BF16_H24_VARLEN"),
    (3, 16, 4, 128, 513, 257, 64, torch.bfloat16, "M_BF16_VARLEN_513"),
    (4, 24, 6, 128, 769, 511, 128, torch.bfloat16, "M_BF16_H24_BLK128"),
    (5, 16, 4, 128, 1025, 333, 256, torch.bfloat16, "M_BF16_ODD_KV_333"),
    (6, 24, 6, 128, 1537, 777, 128, torch.bfloat16, "M_BF16_H24_ODD_777"),
    (5, 16, 4, 128, 2049, 1025, 256, torch.bfloat16, "M_BF16_LONG_ODD"),
    (4, 24, 6, 128, 3073, 1537, 256, torch.bfloat16, "M_BF16_H24_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_len, max_total_seq_len",
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
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_swa_with_kv_dequant
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
def test_paged_prefill_swa_with_kv_dequant(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    max_q_len: int,
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

    op = MojoPagedPrefillSWAWithKVDequant(
        is_causal=True,
        gqa_layout=gqa_layout,  
        local_window_size=local_window,
        global_window_size=global_window,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedPrefillSWAWithKVDequant._registry.get("torch")(
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
        max_q_len=max_q_len,
        max_total_seq_len=max_total_seq_len,
        atol=2e-2 if query.dtype != torch.float32 else 1e-5,
        rtol=2e-2 if query.dtype != torch.float32 else 1e-6,
        # int8 compute can have a few round-boundary outliers across millions of elements.
        ptol=0.9999 if compute_dtype == torch.int8 else 1.0,
    )


# ===========================================================================
# MojoPagedDecodeSWAWithKVDequant
# ===========================================================================

test_configs_decode_swa_with_kv_dequant = [
    (4, 16, 4, 128, 1024, 512, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 2048, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 4096, 128, torch.bfloat16, "M_BF16_LONG"),
    (2, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (2, 8, 2, 128, 2048, 1024, torch.bfloat16, "M_BF16_GROUP1"),
    (2, 24, 8, 128, 2048, 1024, torch.bfloat16, "M_BF16_GROUP2"),
    (5, 12, 3, 64, 257, 16, torch.bfloat16, "M_BF16_VARLEN_BLK16_D64"),
    (3, 20, 5, 256, 1537, 256, torch.bfloat16, "M_BF16_VARLEN_BLK256_D256"),
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
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode_swa_with_kv_dequant
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
def test_paged_decode_swa_with_kv_dequant(
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

    op = MojoPagedDecodeSWAWithKVDequant(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedDecodeSWAWithKVDequant._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )

    atol = 5e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 5e-2 if query.dtype != torch.float32 else 1e-6

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
        max_total_seq_len=max_total_seq_len,
        atol=atol,
        rtol=rtol,
        ptol=0.90,
    )


# ===========================================================================
# MojoPagedPrefillSageGQA
# ===========================================================================

test_configs_prefill_sage_gqa = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 8, 2, 128, 1024, 0, 128, torch.bfloat16, "M_BF16_GROUP"),
    (2, 8, 1, 128, 512, 1024, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_len, max_total_seq_len",
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
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_sage_gqa
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@pytest.mark.parametrize(
    "query_dtype, context_dtype, compute_dtype",
    [
        (torch.int8, torch.int8, torch.int8),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_sage_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    max_q_len: int,
    max_total_seq_len: int,
    gqa_layout: str,
    query_dtype: torch.dtype,
    context_dtype: torch.dtype,
    compute_dtype: torch.dtype,
):
    # Sage runs dynamic per-token int8 quant *inside* its forward, so the
    # caller still passes float K/V cache and ``None`` scales — the dtype
    # arguments only configure the operator's internal quant matmul path.
    op = MojoPagedPrefillSageGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )
    op_ref = MojoPagedPrefillSageGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        query_dtype=query_dtype,
        context_dtype=context_dtype,
        compute_dtype=compute_dtype,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # q/k: per-token int8 (quant along last dim D)
    # v:   per-channel int8 (scale shared across blocks and tokens, kept per [Hkv, D])
    query_int8, query_scale = per_token_int8(query, q_max=127, q_min=-128)
    key_cache_int8, key_scale = per_token_int8(k_cache, q_max=127, q_min=-128)
    value_cache_int8, value_scale = per_channel_int8(v_cache, seq_dim=[0, 2], q_max=127, q_min=-128)
    # forward expects: query_scale [Hq, T], key_scale [N_blocks, Hkv, block_size], value_scale [Hkv, D]
    query_scale = query_scale.squeeze(-1).transpose(0, 1)
    key_scale = key_scale.squeeze(-1)
    value_scale = value_scale.squeeze(2).squeeze(0)

    atol = 5e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 5e-2 if query.dtype != torch.float32 else 1e-6
    op.forward_diff_with(
        op_ref,
        query_int8,
        query_scale,
        key_cache_int8,
        key_scale,
        value_cache_int8,
        value_scale,
        cu_q_lens,
        block_tables,
        softmax_scale=softmax_scale,
        cu_total_seq_lens=cu_total_seq_lens,
        max_q_len=max_q_len,
        max_total_seq_len=max_total_seq_len,
        atol=atol,
        rtol=rtol,
        ptol=0.90,
    )


# ===========================================================================
# MojoPagedPrefillGQAFp8 / MojoPagedPrefillSWAFp8
# ===========================================================================

# e4m3 saturates at 448; quantization keeps the dynamic range inside that.
_FP8_E4M3_MAX = 448.0


def _quantize_fp8_decode_inputs(query, key_cache, value_cache, fp8_dtype):
    """Decode uses per-query-head and per-cache-token/channel scales."""
    fp8_max = torch.finfo(fp8_dtype).max
    quantized, scales = [], []
    for tensor, layout in zip((query, key_cache, value_cache), ("token", "token", "channel")):
        tensor = tensor.float()
        if layout == "channel":
            scale = (tensor.abs().amax(dim=(0, 2)) / fp8_max).clamp(min=1e-6)
            broadcast_scale = scale[None, :, None, :]
        else:
            scale = (tensor.abs().amax(dim=-1) / fp8_max).clamp(min=1e-6)
            broadcast_scale = scale.unsqueeze(-1)
        quantized.append((tensor / broadcast_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype))
        scales.append(scale.contiguous())
    return quantized, scales


@pytest.mark.parametrize(
    "batch_size, num_q_heads, num_kv_heads, head_dim, max_seq_len, block_size",
    [
        (1, 1, 1, 128, 17, 16),
        (3, 16, 4, 128, 63, 16),
        (2, 12, 3, 256, 257, 32),
        (4, 16, 2, 128, 1025, 32),
        (2, 8, 1, 128, 8192, 16),
    ],
)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_default_scale", [False, True])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_gqa_fp8(
    batch_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_seq_len,
    block_size,
    fp8_dtype,
    output_dtype,
    use_default_scale,
):
    from mojo_opset.utils.platform import get_torch_device

    torch.manual_seed(42)
    query, key, value, lengths, tables, max_length = generate_paged_decode_quant_data(
        batch_size, num_q_heads, num_kv_heads, head_dim, max_seq_len, block_size, output_dtype
    )
    device = get_torch_device()
    query, key, value, lengths, tables = (
        tensor.to(device) for tensor in (query, key, value, lengths, tables)
    )
    (query, key, value), (q_scale, k_scale, v_scale) = _quantize_fp8_decode_inputs(
        query, key, value, fp8_dtype
    )
    op = MojoPagedDecodeGQAFp8(output_dtype=output_dtype)
    ref = MojoPagedDecodeGQAFp8._registry.get("torch")(output_dtype=output_dtype)
    # The kernel also quantizes attention probabilities to FP8. E5M2 has one
    # fewer mantissa bit than E4M3, so allow twice its rounding tolerance.
    tolerance_factor = 2 if fp8_dtype == torch.float8_e5m2 else 1
    result = op.forward_diff_with(
        ref,
        query,
        q_scale,
        key,
        k_scale,
        value,
        v_scale,
        lengths,
        tables,
        # Exercise both the default and a non-default softmax scale.
        softmax_scale=None if use_default_scale else 0.7 / math.sqrt(head_dim),
        max_total_seq_len=max_length,
        atol=4e-2 * tolerance_factor,
        rtol=2e-2 * tolerance_factor,
    )
    assert result.shape == query.shape
    assert result.dtype == output_dtype


@pytest.mark.parametrize(
    "batch_size, num_q_heads, num_kv_heads, head_dim, max_seq_len, block_size",
    [(3, 16, 4, 128, 97, 16), (2, 12, 3, 256, 257, 32), (2, 8, 1, 128, 4097, 16)],
)
@pytest.mark.parametrize(
    "global_window, local_window",
    [(None, None), (0, None), (0, 15), (None, 31), (4, 15), (64, 127)],
)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_swa_fp8(
    batch_size, num_q_heads, num_kv_heads, head_dim, max_seq_len, block_size,
    global_window, local_window, fp8_dtype, output_dtype,
):
    from mojo_opset.utils.platform import get_torch_device

    torch.manual_seed(42)
    query, key, value, lengths, tables, max_length = generate_paged_decode_quant_data(
        batch_size, num_q_heads, num_kv_heads, head_dim, max_seq_len, block_size, output_dtype
    )
    query, key, value, lengths, tables = (
        tensor.to(get_torch_device()) for tensor in (query, key, value, lengths, tables)
    )
    (query, key, value), (q_scale, k_scale, v_scale) = _quantize_fp8_decode_inputs(
        query, key, value, fp8_dtype
    )
    config = dict(
        global_window_size=global_window, local_window_size=local_window, output_dtype=output_dtype
    )
    op = MojoPagedDecodeSWAFp8(**config)
    ref = MojoPagedDecodeSWAFp8._registry.get("torch")(**config)
    # As for GQA, account for the kernel's FP8 probability rounding.
    tolerance_factor = 2 if fp8_dtype == torch.float8_e5m2 else 1
    result = op.forward_diff_with(
        ref, query, q_scale, key, k_scale, value, v_scale, lengths, tables,
        softmax_scale=None if global_window is None else 0.7 / math.sqrt(head_dim),
        max_total_seq_len=max_length,
        atol=4e-2 * tolerance_factor,
        rtol=2e-2 * tolerance_factor,
    )
    assert result.shape == query.shape
    assert result.dtype == output_dtype


@pytest.mark.parametrize("global_window, local_window", [(None, None), (0, None), (0, 1), (2, 3), (15, 3)])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_swa_fp8_window_boundaries(global_window, local_window, fp8_dtype):
    """Zero logits give the exact mean over the union, including overlaps."""
    from mojo_opset.utils.platform import get_torch_device

    device = get_torch_device()
    sequence_lengths = [9, 17]
    physical_blocks = [[3, 1], [2, 0]]
    lengths = torch.tensor(sequence_lengths, device=device, dtype=torch.int32)
    tables = torch.tensor(physical_blocks, device=device, dtype=torch.int32)
    query = torch.zeros(2, 4, 128, device=device).to(fp8_dtype)
    key = torch.zeros(4, 2, 16, 128, device=device).to(fp8_dtype)
    values = torch.zeros(4, 2, 16, 128, device=device)
    expected = torch.empty(2, 4, 128, device=device, dtype=torch.bfloat16)
    for batch, length in enumerate(sequence_lengths):
        for position in range(length):
            for head in range(2):
                values[physical_blocks[batch][position // 16], head, position % 16] = position % 4 + head * 4
        visible = [
            pos for pos in range(length)
            if (local_window is None and not global_window)
            or (local_window is not None and pos >= length - 1 - local_window)
            or (global_window is not None and pos < global_window)
        ]
        for head in range(4):
            expected[batch, head] = sum((pos % 4 + (head // 2) * 4) * 0.5 for pos in visible) / len(visible)
    inputs = (
        query, torch.ones(2, 4, device=device), key, torch.ones(4, 2, 16, device=device),
        values.to(fp8_dtype), torch.full((2, 128), 0.5, device=device), lengths, tables,
    )
    config = dict(global_window_size=global_window, local_window_size=local_window)
    for op in (MojoPagedDecodeSWAFp8(**config), MojoPagedDecodeSWAFp8._registry.get("torch")(**config)):
        result = op(*inputs, max_total_seq_len=17)
        torch.testing.assert_close(result, expected, atol=2e-2, rtol=1e-2)


def _to_fp8_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def _quantize_fp8_kv_cache(
    cache: torch.Tensor,  # [n_blocks, n_kv_heads, block_size, head_dim]
    batch_size: int,
    per_head: bool,
):
    """Quantize a float KV cache to e4m3 with a per-tensor or per-(batch, kv head) scale.

    The kernel reads a per-head scale as ``[batch, num_kv_heads]``, but a paged
    cache block is not owned by a single batch, so the per-head scales are
    computed over the whole cache and then broadcast across the batch rows.
    """
    cache_f = cache.float()
    if per_head:
        amax = cache_f.abs().amax(dim=(0, 2, 3)).clamp(min=1e-6)  # -> [n_kv_heads]
        scale = amax / _FP8_E4M3_MAX
        quant = _to_fp8_e4m3(cache_f, scale.view(1, -1, 1, 1))
        return quant, scale.unsqueeze(0).expand(batch_size, -1).contiguous()

    scale = (cache_f.abs().amax() / _FP8_E4M3_MAX).clamp(min=1e-6).reshape(1)
    return _to_fp8_e4m3(cache_f, scale), scale


def _quantize_fp8_query(
    query: torch.Tensor,  # [total_q_tokens, num_q_heads, head_dim]
    cu_q_lens: torch.Tensor,
    num_kv_heads: int,
    per_head: bool,
    gqa_interleave: bool,
):
    """Quantize query tokens to e4m3; a per-head scale is shared across each GQA group."""
    query_f = query.float()
    if not per_head:
        scale = (query_f.abs().amax() / _FP8_E4M3_MAX).clamp(min=1e-6).reshape(1)
        return _to_fp8_e4m3(query_f, scale), scale

    total_q_tokens, num_q_heads, head_dim = query.shape
    group = num_q_heads // num_kv_heads
    # Split the head axis so the GQA group of each KV head can be reduced as a whole.
    group_shape = (
        (total_q_tokens, group, num_kv_heads, head_dim)
        if gqa_interleave
        else (total_q_tokens, num_kv_heads, group, head_dim)
    )
    head_axis = 2 if gqa_interleave else 1
    reduce_dims = tuple(d for d in range(4) if d != head_axis)

    quant = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    scale = torch.ones(cu_q_lens.numel() - 1, num_kv_heads, device=query.device, dtype=torch.float32)
    for i in range(cu_q_lens.numel() - 1):
        start_loc = int(cu_q_lens[i].item())
        end_loc = int(cu_q_lens[i + 1].item())
        if end_loc <= start_loc:
            continue
        seq = query_f[start_loc:end_loc].view(end_loc - start_loc, *group_shape[1:])
        seq_scale = (seq.abs().amax(dim=reduce_dims) / _FP8_E4M3_MAX).clamp(min=1e-6)
        scale[i] = seq_scale
        broadcast = [1, 1, 1, 1]
        broadcast[head_axis] = num_kv_heads
        quant[start_loc:end_loc] = _to_fp8_e4m3(seq, seq_scale.view(broadcast)).view(
            end_loc - start_loc, num_q_heads, head_dim
        )
    return quant, scale


test_configs_prefill_fp8 = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_FP8"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_FP8_PADDIM"),
    (2, 8, 1, 128, 512, 1024, 128, torch.bfloat16, "M_FP8_WITH_CACHE"),
    (2, 8, 2, 128, 1024, 0, 128, torch.bfloat16, "M_FP8_GROUP"),
    (3, 12, 3, 64, 257, 513, 16, torch.bfloat16, "M_FP8_VARLEN_BLK16_D64"),
    (4, 20, 5, 192, 193, 769, 256, torch.bfloat16, "M_FP8_VARLEN_BLK256_D192"),
    (3, 24, 6, 80, 321, 641, 64, torch.bfloat16, "M_FP8_VARLEN_BLK64_D80"),
    (1, 16, 4, 128, 128, 0, 16, torch.bfloat16, "M_FP8_SMALL_BLK16"),
    (2, 24, 6, 128, 255, 129, 32, torch.bfloat16, "M_FP8_H24_VARLEN"),
    (3, 16, 4, 128, 513, 257, 64, torch.bfloat16, "M_FP8_VARLEN_513"),
    (4, 24, 6, 128, 769, 511, 128, torch.bfloat16, "M_FP8_H24_BLK128"),
    (5, 16, 4, 128, 1025, 333, 256, torch.bfloat16, "M_FP8_ODD_KV_333"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_len, max_total_seq_len",
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
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_fp8
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@pytest.mark.parametrize("per_head_scale", [False, True])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_gqa_fp8(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    max_q_len: int,
    max_total_seq_len: int,
    gqa_layout: str,
    per_head_scale: bool,
):
    head_dim = query.shape[-1]
    if head_dim % 64 != 0:
        pytest.skip(
            f"ixattn fp8 FMHA requires head_dim % 64 == 0, got {head_dim}."
        )
    batch_size = cu_q_lens.numel() - 1
    num_kv_heads = k_cache.shape[1]
    query_q, query_scale = _quantize_fp8_query(
        query, cu_q_lens, num_kv_heads, per_head_scale, gqa_layout == "ABAB"
    )
    k_cache_q, key_scale = _quantize_fp8_kv_cache(k_cache, batch_size, per_head_scale)
    v_cache_q, value_scale = _quantize_fp8_kv_cache(v_cache, batch_size, per_head_scale)

    op = MojoPagedPrefillGQAFp8(is_causal=True, gqa_layout=gqa_layout)
    op_ref = MojoPagedPrefillGQAFp8._registry.get("torch")(
        is_causal=True, gqa_layout=gqa_layout
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
        max_q_len=max_q_len,
        max_total_seq_len=max_total_seq_len,
        atol=5e-2,
        rtol=5e-2,
        ptol=0.90,
    )


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_len, max_total_seq_len",
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
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_fp8
    ],
)
@pytest.mark.parametrize(
    "gqa_layout, global_window, local_window",
    [
        ("ABAB", 4, 255),
        ("AABB", 4, 1023),
        ("AABB", 0, 127),
    ],
)
@pytest.mark.parametrize("per_head_scale", [False, True])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_swa_fp8(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    max_q_len: int,
    max_total_seq_len: int,
    gqa_layout: str,
    global_window: int,
    local_window: int,
    per_head_scale: bool,
):
    head_dim = query.shape[-1]
    if head_dim % 64 != 0:
        pytest.skip(
            f"ixattn fp8 FMHA requires head_dim % 64 == 0, got {head_dim}."
        )
    batch_size = cu_q_lens.numel() - 1
    num_kv_heads = k_cache.shape[1]
    query_q, query_scale = _quantize_fp8_query(
        query, cu_q_lens, num_kv_heads, per_head_scale, gqa_layout == "ABAB"
    )
    k_cache_q, key_scale = _quantize_fp8_kv_cache(k_cache, batch_size, per_head_scale)
    v_cache_q, value_scale = _quantize_fp8_kv_cache(v_cache, batch_size, per_head_scale)

    op = MojoPagedPrefillSWAFp8(
        is_causal=True,
        gqa_layout=gqa_layout,
        local_window_size=local_window,
        global_window_size=global_window,
    )
    op_ref = MojoPagedPrefillSWAFp8._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        local_window_size=local_window,
        global_window_size=global_window,
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
        max_q_len=max_q_len,
        max_total_seq_len=max_total_seq_len,
        atol=5e-2,
        rtol=5e-2,
        ptol=0.90,
    )
