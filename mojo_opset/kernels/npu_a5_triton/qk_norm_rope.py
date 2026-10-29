"""A5 Triton implementation of fused Q/K RMSNorm and half-split RoPE."""

import torch
import triton
import triton.language as tl

HEAD_DIM = 128
ROTARY_DIM = 96
ROTARY_HALF = ROTARY_DIM // 2
MAX_HEADS_PER_BLOCK = 64


@triton.jit
def _qk_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    rope_table_ptr,
    out_ptr,
    x_stride_t,
    x_stride_h,
    x_stride_d,
    rope_stride_t,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_tokens,
    num_heads,
    eps,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_HALF: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
):
    tokens = (tl.program_id(0) * TOKENS_PER_BLOCK + tl.arange(0, TOKENS_PER_BLOCK)).to(tl.int64)
    heads = tl.program_id(1) * HEADS_PER_BLOCK + tl.arange(0, HEADS_PER_BLOCK)
    dims = tl.arange(0, HEAD_DIM)
    half_dims = tl.arange(0, ROTARY_HALF)
    tail_size: tl.constexpr = HEAD_DIM - ROTARY_DIM
    token_mask = tokens < num_tokens
    head_mask = heads < num_heads
    mask = token_mask[:, None, None] & head_mask[None, :, None]

    x_offsets = (
        tokens[:, None, None] * x_stride_t + heads[None, :, None] * x_stride_h + dims[None, None, :] * x_stride_d
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + dims).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=2) / HEAD_DIM + eps)

    # The BF16 round belongs before RoPE and is shared by both pair elements.
    normalized = (x * inv_rms[:, :, None] * weight[None, None, :]).to(tl.bfloat16)
    first = tl.extra.cann.extension.extract_slice(
        normalized,
        [0, 0, 0],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, ROTARY_HALF],
        [1, 1, 1],
    )
    second = tl.extra.cann.extension.extract_slice(
        normalized,
        [0, 0, ROTARY_HALF],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, ROTARY_HALF],
        [1, 1, 1],
    )
    passthrough = tl.extra.cann.extension.extract_slice(
        normalized,
        [0, 0, ROTARY_DIM],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, tail_size],
        [1, 1, 1],
    )

    table_offsets = tokens[:, None] * rope_stride_t + half_dims[None, :]
    cos = tl.load(rope_table_ptr + table_offsets, mask=token_mask[:, None], other=0.0).to(tl.float32)
    sin = tl.load(
        rope_table_ptr + table_offsets + ROTARY_HALF,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cos = cos.reshape(TOKENS_PER_BLOCK, 1, ROTARY_HALF)
    sin = sin.reshape(TOKENS_PER_BLOCK, 1, ROTARY_HALF)
    first_f32 = first.to(tl.float32)
    second_f32 = second.to(tl.float32)

    output = tl.zeros((TOKENS_PER_BLOCK, HEADS_PER_BLOCK, HEAD_DIM), dtype=tl.bfloat16)
    output = tl.extra.cann.extension.insert_slice(
        output,
        (first_f32 * cos - second_f32 * sin).to(tl.bfloat16),
        [0, 0, 0],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, ROTARY_HALF],
        [1, 1, 1],
    )
    output = tl.extra.cann.extension.insert_slice(
        output,
        (second_f32 * cos + first_f32 * sin).to(tl.bfloat16),
        [0, 0, ROTARY_HALF],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, ROTARY_HALF],
        [1, 1, 1],
    )
    output = tl.extra.cann.extension.insert_slice(
        output,
        passthrough,
        [0, 0, ROTARY_DIM],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, tail_size],
        [1, 1, 1],
    )

    out_offsets = (
        tokens[:, None, None] * out_stride_t + heads[None, :, None] * out_stride_h + dims[None, None, :] * out_stride_d
    )
    tl.store(out_ptr + out_offsets, output, mask=mask)


def _tile_config(num_tokens: int, num_heads: int) -> tuple[int, int]:
    heads_per_block = min(MAX_HEADS_PER_BLOCK, triton.next_power_of_2(num_heads))
    tokens_per_block = 1 if num_tokens == 1 or num_heads <= 16 else 2
    return tokens_per_block, heads_per_block


def _launch_qk_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope_table: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    num_tokens, num_heads, _ = x.shape
    if num_tokens == 0:
        return
    tokens_per_block, heads_per_block = _tile_config(num_tokens, num_heads)
    grid = (
        triton.cdiv(num_tokens, tokens_per_block),
        triton.cdiv(num_heads, heads_per_block),
    )
    _qk_norm_rope_kernel[grid](
        x,
        weight,
        rope_table,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rope_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_tokens,
        num_heads,
        eps,
        HEAD_DIM=HEAD_DIM,
        ROTARY_DIM=ROTARY_DIM,
        ROTARY_HALF=ROTARY_HALF,
        TOKENS_PER_BLOCK=tokens_per_block,
        HEADS_PER_BLOCK=heads_per_block,
    )


def fused_qk_norm_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one RMSNorm+RoPE launch to each of Q and K."""
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    rope_table = rope_table.contiguous()
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if q.shape[0] == 0:
        return q_out, k_out
    _launch_qk_norm_rope(q, q_weight, rope_table, q_out, eps)
    _launch_qk_norm_rope(k, k_weight, rope_table, k_out, eps)
    return q_out, k_out


@torch.library.custom_op("mojo_npu_triton_a5::fused_qk_norm_rope", mutates_args=())
def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_qk_norm_rope_fwd(q, k, q_weight, k_weight, rope_table, eps)


@fused_qk_norm_rope.register_fake
def _fused_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, rope_table, eps
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )
