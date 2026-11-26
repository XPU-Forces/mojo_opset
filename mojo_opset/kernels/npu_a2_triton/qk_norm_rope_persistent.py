"""Persistent Triton implementation of fused Q/K RMSNorm and half-split RoPE."""

import torch
import triton
import triton.language as tl

HEAD_DIM = 128
ROTARY_DIM = 96
ROTARY_HALF = ROTARY_DIM // 2
NUM_VECTOR_CORES = 40
TOKENS_PER_BLOCK = 8
HEADS_PER_BLOCK = 8
TOKEN_STAGES = 2


@triton.jit
def _norm_rope_tile(
    x_ptr,
    weight,
    cos,
    sin,
    out_ptr,
    tokens,
    heads,
    token_mask,
    head_mask,
    x_stride_t: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    eps,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_HALF: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
):
    dims = tl.arange(0, HEAD_DIM)
    tail_size: tl.constexpr = HEAD_DIM - ROTARY_DIM
    mask = token_mask[:, None, None] & head_mask[None, :, None]
    offsets = tokens[:, None, None] * x_stride_t + heads[None, :, None] * x_stride_h + dims[None, None, :] * x_stride_d
    x_bf16 = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x_bf16.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=2) / HEAD_DIM + eps)
    normalized = (x * inv_rms[:, :, None] * weight[None, None, :]).to(tl.bfloat16)

    first = tl.extra.cann.extension.extract_slice(
        normalized,
        [0, 0, 0],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, ROTARY_HALF],
        [1, 1, 1],
    ).to(tl.float32)
    second = tl.extra.cann.extension.extract_slice(
        normalized,
        [0, 0, ROTARY_HALF],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, ROTARY_HALF],
        [1, 1, 1],
    ).to(tl.float32)
    passthrough = tl.extra.cann.extension.extract_slice(
        normalized,
        [0, 0, ROTARY_DIM],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, tail_size],
        [1, 1, 1],
    )

    output = tl.zeros((TOKENS_PER_BLOCK, HEADS_PER_BLOCK, HEAD_DIM), dtype=tl.bfloat16)
    output = tl.extra.cann.extension.insert_slice(
        output,
        (first * cos - second * sin).to(tl.bfloat16),
        [0, 0, 0],
        [TOKENS_PER_BLOCK, HEADS_PER_BLOCK, ROTARY_HALF],
        [1, 1, 1],
    )
    output = tl.extra.cann.extension.insert_slice(
        output,
        (second * cos + first * sin).to(tl.bfloat16),
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


@triton.jit
def _persistent_qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rope_table_ptr,
    q_out_ptr,
    k_out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    q_out_stride_t: tl.constexpr,
    q_out_stride_h: tl.constexpr,
    q_out_stride_d: tl.constexpr,
    k_out_stride_t: tl.constexpr,
    k_out_stride_h: tl.constexpr,
    k_out_stride_d: tl.constexpr,
    rope_stride_t: tl.constexpr,
    num_tokens: tl.constexpr,
    eps: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_HALF: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    TOKEN_STAGES: tl.constexpr,
    USE_INT32_OFFSETS: tl.constexpr,
):
    pid = tl.program_id(0)
    dims = tl.arange(0, HEAD_DIM)
    rope_dims = tl.arange(0, ROTARY_DIM)
    q_weight = tl.load(q_weight_ptr + dims).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + dims).to(tl.float32)
    tokens_per_core: tl.constexpr = (num_tokens + GRID_SIZE - 1) // GRID_SIZE
    num_iterations: tl.constexpr = (tokens_per_core + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK

    for iteration in tl.range(num_iterations, num_stages=TOKEN_STAGES):
        token_indices = pid * tokens_per_core + iteration * TOKENS_PER_BLOCK + tl.arange(0, TOKENS_PER_BLOCK)
        if USE_INT32_OFFSETS:
            tokens = token_indices.to(tl.int32)
        else:
            tokens = token_indices.to(tl.int64)
        token_mask = tokens < num_tokens
        table_offsets = tokens[:, None] * rope_stride_t + rope_dims[None, :]
        rope = tl.load(rope_table_ptr + table_offsets, mask=token_mask[:, None], other=0.0).to(tl.float32)
        cos = tl.extra.cann.extension.extract_slice(
            rope,
            [0, 0],
            [TOKENS_PER_BLOCK, ROTARY_HALF],
            [1, 1],
        ).reshape(TOKENS_PER_BLOCK, 1, ROTARY_HALF)
        sin = tl.extra.cann.extension.extract_slice(
            rope,
            [0, ROTARY_HALF],
            [TOKENS_PER_BLOCK, ROTARY_HALF],
            [1, 1],
        ).reshape(TOKENS_PER_BLOCK, 1, ROTARY_HALF)

        for head_start in tl.range(0, NUM_HEADS, HEADS_PER_BLOCK):
            heads = head_start + tl.arange(0, HEADS_PER_BLOCK)
            if NUM_HEADS % HEADS_PER_BLOCK == 0:
                head_mask = tl.full((HEADS_PER_BLOCK,), True, tl.int1)
            else:
                head_mask = heads < NUM_HEADS
            _norm_rope_tile(
                q_ptr,
                q_weight,
                cos,
                sin,
                q_out_ptr,
                tokens,
                heads,
                token_mask,
                head_mask,
                q_stride_t,
                q_stride_h,
                q_stride_d,
                q_out_stride_t,
                q_out_stride_h,
                q_out_stride_d,
                eps,
                HEAD_DIM,
                ROTARY_DIM,
                ROTARY_HALF,
                TOKENS_PER_BLOCK,
                HEADS_PER_BLOCK,
            )
            _norm_rope_tile(
                k_ptr,
                k_weight,
                cos,
                sin,
                k_out_ptr,
                tokens,
                heads,
                token_mask,
                head_mask,
                k_stride_t,
                k_stride_h,
                k_stride_d,
                k_out_stride_t,
                k_out_stride_h,
                k_out_stride_d,
                eps,
                HEAD_DIM,
                ROTARY_DIM,
                ROTARY_HALF,
                TOKENS_PER_BLOCK,
                HEADS_PER_BLOCK,
            )


def _launch_persistent(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    eps: float,
    compile_options: dict | None = None,
) -> None:
    num_tokens, num_heads, _ = q.shape
    grid = (min(NUM_VECTOR_CORES, num_tokens),)
    max_offset = max(
        (q.shape[0] - 1) * q.stride(0) + (q.shape[1] - 1) * q.stride(1) + (q.shape[2] - 1) * q.stride(2),
        (k.shape[0] - 1) * k.stride(0) + (k.shape[1] - 1) * k.stride(1) + (k.shape[2] - 1) * k.stride(2),
        (q_out.shape[0] - 1) * q_out.stride(0)
        + (q_out.shape[1] - 1) * q_out.stride(1)
        + (q_out.shape[2] - 1) * q_out.stride(2),
        (k_out.shape[0] - 1) * k_out.stride(0)
        + (k_out.shape[1] - 1) * k_out.stride(1)
        + (k_out.shape[2] - 1) * k_out.stride(2),
        (rope_table.shape[0] - 1) * rope_table.stride(0) + (rope_table.shape[1] - 1) * rope_table.stride(1),
    )
    use_int32_offsets = max_offset < 2**31
    launch_options = {"num_stages": 1, "ops_reorder": True} if compile_options is None else compile_options
    _persistent_qk_norm_rope_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        q_out,
        k_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        rope_table.stride(0),
        num_tokens,
        eps,
        GRID_SIZE=grid[0],
        HEAD_DIM=HEAD_DIM,
        ROTARY_DIM=ROTARY_DIM,
        ROTARY_HALF=ROTARY_HALF,
        NUM_HEADS=num_heads,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        TOKEN_STAGES=TOKEN_STAGES,
        USE_INT32_OFFSETS=use_int32_offsets,
        **launch_options,
    )


def fused_qk_norm_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=q.dtype, device=k.device)
    if q.shape[0] == 0:
        return q_out, k_out
    _launch_persistent(q, k, q_weight, k_weight, rope_table, q_out, k_out, eps)
    return q_out, k_out
