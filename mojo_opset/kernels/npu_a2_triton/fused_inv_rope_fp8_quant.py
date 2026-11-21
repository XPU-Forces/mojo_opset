"""Ascend A2 (910B) Triton inverse RoPE (+ optional FP8 quantisation, unsupported).

The rotation itself is arch-neutral and uses the vllm original structure (one
program per token x group-head). A2 has no FP8 (the DAV_2201 platform ini carries
no ``DT_FLOAT8_*`` and Triton's ``f32 -> fp8`` cast aborts in ``arith::TruncFOp``
lowering), so ``quantize=True`` fails fast on the host; ``quantize=False`` keeps
the FP8-free epilogue and runs here.

A2 tiling: the 950 tile structure (32 tokens x head) is not enabled because it was
never measured on 910B, so this provider always uses the per-(token, head)
structure.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_tokens"])
def _fused_inv_rope_fp8_quant_fp32(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    # Strides are int64: a 32-bit pid * stride can wrap past 2^31 on large
    # prefill batches and fault.
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)
    o_stride_token = o_stride_token.to(tl.int64)
    o_stride_head = o_stride_head.to(tl.int64)
    cache_stride_pos = cache_stride_pos.to(tl.int64)
    out_stride_group = out_stride_group.to(tl.int64)
    out_stride_token = out_stride_token.to(tl.int64)
    if pid_token >= num_tokens:
        return

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    qb_start = head_in_group * CHUNKS_PER_HEAD
    input_base = o_ptr + pid_token * o_stride_token + pid_gh * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= NOPE_DIM
    rope_local = offsets - NOPE_DIM
    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32)
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    rotated = tl.where(
        (rope_local & 1) == 0,
        x * cos_v + x_partner * sin_v,
        x * cos_v - x_partner * sin_v,
    )
    x = tl.where(is_rope, rotated, x)

    out_base = (
        out_ptr
        + g * out_stride_group
        + pid_token * out_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(out_base + offsets, x)


def _tma_aligned(num_tokens, element_size=4):
    multiple = 16 // element_size
    return (num_tokens + multiple - 1) // multiple * multiple


def _fused_inv_rope_fp8_quant_impl(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    quant_group_size,
    tma_aligned_scales,
    quantize,
):
    if quantize:
        raise NotImplementedError(
            "fused_inv_rope_fp8_quant: 910B (A2/DAV_2201) has no FP8 support "
            "(no DT_FLOAT8_* cast); call with quantize=False on this arch"
        )
    if o.dtype == torch.float8_e4m3fn:
        # The no-quantize path stores in o.dtype = fp8, and the f32->fp8 cast
        # before the store aborts bishengir-compile (deterministic SIGABRT,
        # observed on 386 generalized cases on 2026-11-19); same root cause as
        # the quantize guard (no FP8 hardware cast), so fail fast here instead
        # of letting the compiler crash.
        raise NotImplementedError(
            "fused_inv_rope_fp8_quant: 910B (A2/DAV_2201) has no FP8 support "
            "(no DT_FLOAT8_* cast); fp8 o input needs an f32->fp8 store cast, "
            "call with a bf16/fp16 o on this arch"
        )
    num_tokens, _, head_dim = o.shape
    d = heads_per_group * head_dim
    chunks_per_head = head_dim // quant_group_size
    out_buf = torch.empty((n_groups, num_tokens, d), dtype=o.dtype, device=o.device)
    output = out_buf.transpose(0, 1)
    scales = torch.empty(0, dtype=torch.float32, device=o.device)
    if num_tokens == 0:
        return output, scales
    _fused_inv_rope_fp8_quant_fp32[(num_tokens, n_groups * heads_per_group)](
        o,
        positions,
        cos_sin_cache,
        out_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        out_stride_group=out_buf.stride(0),
        out_stride_token=out_buf.stride(1),
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        NOPE_DIM=nope_dim,
        HALF_ROPE=rope_dim // 2,
        num_stages=1,
        num_warps=1,
    )
    return output, scales


@torch.library.custom_op("mojo_npu_triton_a2::fused_inv_rope_fp8_quant_fwd", mutates_args=())
def fused_inv_rope_fp8_quant_fwd(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
    quantize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fused_inv_rope_fp8_quant_impl(
        o.contiguous(),
        positions.contiguous(),
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        quant_group_size,
        tma_aligned_scales,
        quantize,
    )


@fused_inv_rope_fp8_quant_fwd.register_fake
def _fused_inv_rope_fp8_quant_fake(
    o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim, rope_dim,
    quant_group_size, tma_aligned_scales, quantize,
):
    num_tokens, _, head_dim = o.shape
    d = heads_per_group * head_dim
    out_buf = o.new_empty((n_groups, num_tokens, d), dtype=o.dtype)
    return out_buf.transpose(0, 1), torch.empty(
        0, dtype=torch.float32, device=positions.device
    )
