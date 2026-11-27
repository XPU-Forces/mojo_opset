"""Ascend A5 (950) Triton inverse RoPE + block FP8 quantisation.

Two structures, matching the vllm kernel semantics:

- **Tile path** (``quant_group_size > rope_dim``, ``chunks_per_head >= 2``,
  ``tma_aligned_scales`` off): each program owns ``TPB`` tokens x ``HPB`` heads.
  A head is read as three contiguous runs ``[nope | tail | rope]``, cos/sin are
  read once per token and joined into pairs, so there is no per-lane gather. When
  the launch provably has no tail tile and no alignment padding row the row mask
  disappears at compile time (``MASK_T=False``).
- **Original structure**: one program per (token, group-head), used for
  ``tma_aligned_scales``, ``chunks_per_head == 1`` and other fallbacks.

``TPB=32`` from 64 tokens up, ``TPB=4`` floor for small batches (a 32-row tile
spends 3/4 of its lanes masked at T=1), ``HPB=1`` (``HPB=2`` was slower, and
``HPB=4/8`` overflow UB); constants from a cold-cache calibration on this arch.

Scales are the vllm UE8M0 form: ``2 ** ceil(log2(max(absmax, 1e-10) / 448))``,
stored MN-major (token stride 1, block stride ``align(tokens, 4)``) with the
alignment padding rows zeroed; ``tma_aligned_scales`` packs four UE8M0 bytes per
``int32`` word little-endian.

A NaN inside a block makes its scale and all its fp8 values NaN. An infinite
element makes the scale +inf, the finite elements of the block zero and the
infinite ones NaN. This is the torch reference's behaviour; the vllm kernel
drops the NaN in the block maximum instead.
"""

import torch
import triton
import triton.language as tl

_TPB = 32
_TPB_MIN = 4
_HPB = 1


@triton.jit
def _ue8m0_scale(absmax, fp8_max: tl.constexpr, eps: tl.constexpr):
    """UE8M0 scale ``2 ** ceil(log2(max(absmax, eps) / fp8_max))`` as float32.

    The scale is assembled from its integer exponent, ``(e + 127) << 23``.
    ``tl.math.exp2`` is not correctly rounded for integer arguments on this
    backend: for most exponents the result lies a few ULP off ``2 ** e``, and one
    ULP below it reads as ``2 ** (e - 1)`` from the exponent field, which is where
    the packed UE8M0 byte and the reciprocal below are taken from. A NaN or
    infinite ``absmax`` gives the all-ones exponent, i.e. NaN or +inf.
    """
    e = tl.ceil(tl.log2(tl.maximum(absmax, eps, propagate_nan=tl.PropagateNan.ALL) * (1.0 / fp8_max)))
    nan = e != e
    # Cap before the int conversion (undefined for infinities): 128 is the
    # all-ones exponent.
    e = tl.where(nan, 0.0, tl.minimum(e, 128.0))
    return tl.where(nan, float("nan"), ((e + 127.0).to(tl.uint32) << 23).to(tl.float32, bitcast=True))


@triton.jit
def _recip(scale):
    """1 / scale for a scale that is a power of two, NaN or +inf, as float32.

    The finite case comes straight from the exponent field: 2**k is
    ``(k + 127) << 23`` and ``2**-k`` is ``(254 - (k + 127)) << 23``, no float
    division. NaN and +inf share the all-ones field, so they are selected by
    value: 1 / NaN is NaN and 1 / +inf is 0.
    """
    bits = scale.to(tl.uint32, bitcast=True)
    inv = ((254 - (bits >> 23)) << 23).to(tl.float32, bitcast=True)
    return tl.where(scale != scale, float("nan"), tl.where(scale == float("inf"), 0.0, inv))


@triton.jit(do_not_specialize=["num_tokens", "scale_stride_k"])
def _fused_inv_rope_fp8_quant_native(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    # Every stride is int64: a 32-bit pid * stride can wrap past 2^31 on large
    # prefill batches and fault.
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)
    o_stride_token = o_stride_token.to(tl.int64)
    o_stride_head = o_stride_head.to(tl.int64)
    cache_stride_pos = cache_stride_pos.to(tl.int64)
    out_stride_group = out_stride_group.to(tl.int64)
    out_stride_token = out_stride_token.to(tl.int64)
    scale_stride_group = scale_stride_group.to(tl.int64)
    scale_stride_k = scale_stride_k.to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD
    # Padding rows in the TMA-aligned scale buffer: fill with zero and skip quant.
    if pid_token >= num_tokens:
        if not QUANTIZE:
            return
        if TMA_ALIGNED_SCALES:
            packed_offsets = tl.arange(0, CHUNKS_PER_HEAD // 4)
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + (head_in_group * (CHUNKS_PER_HEAD // 4) + packed_offsets)
                * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((CHUNKS_PER_HEAD // 4,), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = NOPE_DIM
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32)
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    rotated = tl.where((rope_local & 1) == 0, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    if not QUANTIZE:
        out_base = (
            out_ptr
            + g * out_stride_group
            + pid_token * out_stride_token
            + qb_start * QUANT_GROUP_SIZE
        )
        tl.store(out_base + offsets, x)
        return

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    scales = _ue8m0_scale(tl.max(x_2d, axis=1, propagate_nan=tl.PropagateNan.ALL), fp8_max, eps)

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(
        x / scales_exp, -fp8_max, fp8_max, propagate_nan=tl.PropagateNan.ALL
    ).to(tl.float8e4nv)

    out_base = (
        out_ptr
        + g * out_stride_group
        + pid_token * out_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(out_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(
            tl.reshape(ue8m0_bytes, (CHUNKS_PER_HEAD // 4, 4))
            << (tl.arange(0, 4)[None, :] * 8),
            axis=1,
        )
        packed_offsets = tl.arange(0, CHUNKS_PER_HEAD // 4)
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + (head_in_group * (CHUNKS_PER_HEAD // 4) + packed_offsets) * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


@triton.jit(do_not_specialize=["num_tokens", "num_tok_aligned", "scale_stride_k"])
def _fused_inv_rope_fp8_quant_tile(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    scale_ptr,
    num_tokens,
    num_tok_aligned,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    TPB: tl.constexpr,
    HPB: tl.constexpr,
    N_HCHUNKS: tl.constexpr,
    MASK_T: tl.constexpr,
):
    # Same int64 stride casts as the native kernel: a 32-bit pid * stride can
    # wrap past 2^31 on large prefill batches and fault.
    pid_t = tl.program_id(0).to(tl.int64)
    pid_hc = tl.program_id(1).to(tl.int64)
    o_stride_token = o_stride_token.to(tl.int64)
    o_stride_head = o_stride_head.to(tl.int64)
    cache_stride_pos = cache_stride_pos.to(tl.int64)
    out_stride_group = out_stride_group.to(tl.int64)
    out_stride_token = out_stride_token.to(tl.int64)
    scale_stride_group = scale_stride_group.to(tl.int64)
    scale_stride_k = scale_stride_k.to(tl.int64)
    g = pid_hc // N_HCHUNKS
    hc = pid_hc % N_HCHUNKS

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    NOPE_TILE: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE
    NB: tl.constexpr = CHUNKS_PER_HEAD - 1
    TAIL: tl.constexpr = QUANT_GROUP_SIZE - ROPE_DIM
    NPAIR: tl.constexpr = ROPE_DIM // 2

    toks = pid_t * TPB + tl.arange(0, TPB)
    hs = hc * HPB + tl.arange(0, HPB)
    t_mask = toks < num_tokens
    m3 = t_mask[:, None, None]
    row_mask = m3 if MASK_T else None
    row_other = 0.0 if MASK_T else None
    t_mask_2d = t_mask[:, None] if MASK_T else None

    in_base = (
        o_ptr
        + toks[:, None, None] * o_stride_token
        + (g * heads_per_group + hs[None, :, None]) * o_stride_head
    )
    out_base = (
        out_ptr
        + g * out_stride_group
        + toks[:, None, None] * out_stride_token
        + hs[None, :, None] * HEAD_DIM
    )

    nope_off = tl.arange(0, NOPE_TILE)
    tail_off = NOPE_TILE + tl.arange(0, TAIL)
    rope_off = NOPE_TILE + TAIL + tl.arange(0, ROPE_DIM)
    x_nope = tl.load(in_base + nope_off[None, None, :], mask=row_mask, other=row_other).to(tl.float32)
    x_tail = tl.load(in_base + tail_off[None, None, :], mask=row_mask, other=row_other).to(tl.float32)
    x_rope = tl.load(in_base + rope_off[None, None, :], mask=row_mask, other=row_other).to(tl.float32)

    # Inverse GPT-J rotation: cos/sin are 32+32 contiguous values per token,
    # joined into adjacent pairs; the partner element is a split/join swap.
    pos = tl.load(
        positions_ptr + toks,
        mask=t_mask if MASK_T else None,
        other=0 if MASK_T else None,
    )
    cb = cos_sin_cache_ptr + pos[:, None] * cache_stride_pos
    half_idx = tl.arange(0, NPAIR)
    cos2 = tl.load(cb + half_idx[None, :], mask=t_mask_2d, other=(1.0 if MASK_T else None))
    sin2 = tl.load(cb + HALF_ROPE + half_idx[None, :], mask=t_mask_2d, other=(0.0 if MASK_T else None))
    a, b = tl.split(tl.reshape(x_rope, (TPB * HPB * NPAIR, 2)))
    a3 = tl.reshape(a, (TPB, HPB, NPAIR))
    b3 = tl.reshape(b, (TPB, HPB, NPAIR))
    cos3 = cos2[:, None, :]
    sin3 = sin2[:, None, :]
    x_rope = tl.reshape(
        tl.join(a3 * cos3 + b3 * sin3, b3 * cos3 - a3 * sin3),
        (TPB, HPB, ROPE_DIM),
    )

    if not QUANTIZE:
        tl.store(out_base + nope_off[None, None, :], x_nope, mask=row_mask)
        tl.store(out_base + tail_off[None, None, :], x_tail, mask=row_mask)
        tl.store(out_base + rope_off[None, None, :], x_rope, mask=row_mask)
        return

    # The first NB blocks come from the nope run; block NB takes
    # max(|tail|, |rope_rotated|) because the group straddles both runs.
    xn = tl.reshape(tl.abs(x_nope), (TPB * HPB * NB, QUANT_GROUP_SIZE))
    sc_n = _ue8m0_scale(tl.max(xn, axis=1, propagate_nan=tl.PropagateNan.ALL), fp8_max, eps)
    amax3 = tl.maximum(
        tl.max(tl.abs(x_tail), axis=2, propagate_nan=tl.PropagateNan.ALL),
        tl.max(tl.abs(x_rope), axis=2, propagate_nan=tl.PropagateNan.ALL),
        propagate_nan=tl.PropagateNan.ALL,
    )
    sc3 = _ue8m0_scale(amax3, fp8_max, eps)
    inv_n = _recip(sc_n)
    qn = tl.clamp(
        tl.reshape(x_nope, (TPB * HPB * NB, QUANT_GROUP_SIZE)) * inv_n[:, None],
        -fp8_max,
        fp8_max,
        propagate_nan=tl.PropagateNan.ALL,
    ).to(tl.float8e4nv)
    tl.store(
        out_base + nope_off[None, None, :],
        tl.reshape(qn, (TPB, HPB, NOPE_TILE)),
        mask=row_mask,
    )
    inv3 = _recip(sc3)[:, :, None]
    tl.store(
        out_base + tail_off[None, None, :],
        tl.clamp(x_tail * inv3, -fp8_max, fp8_max, propagate_nan=tl.PropagateNan.ALL).to(tl.float8e4nv),
        mask=row_mask,
    )
    tl.store(
        out_base + rope_off[None, None, :],
        tl.clamp(x_rope * inv3, -fp8_max, fp8_max, propagate_nan=tl.PropagateNan.ALL).to(tl.float8e4nv),
        mask=row_mask,
    )

    # Scales: padding rows (num_tokens <= t < num_tok_aligned) are zeroed.
    s_mask = (toks < num_tok_aligned)[:, None, None]
    scn3 = tl.reshape(sc_n, (TPB, HPB, NB))
    k3 = hs[None, :, None] * CHUNKS_PER_HEAD + tl.arange(0, NB)[None, None, :]
    sc_base = scale_ptr + g * scale_stride_group
    tl.store(
        sc_base + toks[:, None, None] + k3 * scale_stride_k,
        scn3 if not MASK_T else tl.where(m3, scn3, 0.0),
        mask=s_mask if MASK_T else None,
    )
    tl.store(
        sc_base
        + toks[:, None, None]
        + (hs[None, :, None] * CHUNKS_PER_HEAD + NB) * scale_stride_k,
        sc3[:, :, None] if not MASK_T else tl.where(m3, sc3[:, :, None], 0.0),
        mask=s_mask if MASK_T else None,
    )


def _tma_aligned(num_tokens, element_size=4):
    """vllm get_tma_aligned_size(x, es) = align(x, 16 // es)."""
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
    num_tokens, num_heads, head_dim = o.shape
    d = heads_per_group * head_dim
    blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    tma_aligned_t = _tma_aligned(num_tokens) if quantize else num_tokens
    if quantize and tma_aligned_scales:
        inner = (blocks + 3) // 4
    elif quantize:
        inner = blocks
    else:
        inner = 0

    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    out_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn if quantize else o.dtype,
        device=o.device,
    )
    if quantize:
        scale_buf = torch.empty(
            n_groups * inner * tma_aligned_t, dtype=scale_dtype, device=o.device
        ).as_strided(
            (n_groups, num_tokens, inner),
            (inner * tma_aligned_t, 1, tma_aligned_t),
        )
    else:
        scale_buf = torch.empty(0, dtype=scale_dtype, device=o.device)
    output = out_buf.transpose(0, 1)
    scales = scale_buf.transpose(0, 1) if quantize else scale_buf

    if num_tokens == 0:
        return output, scales

    tile_ok = (
        not tma_aligned_scales
        and quant_group_size > rope_dim
        and chunks_per_head >= 2
    )
    if tile_ok:
        # Small batches shrink the tile: a 32-row tile wastes 3/4 of its lanes at
        # T=1.
        tpb = min(_TPB, max(_TPB_MIN, triton.next_power_of_2(num_tokens)))
        hpb = _HPB if heads_per_group % _HPB == 0 else 1
        mask_t = not (num_tokens == tma_aligned_t and tma_aligned_t % tpb == 0)
        _fused_inv_rope_fp8_quant_tile[
            (triton.cdiv(tma_aligned_t, tpb), n_groups * (heads_per_group // hpb))
        ](
            o,
            positions,
            cos_sin_cache,
            out_buf,
            scale_buf,
            num_tokens,
            tma_aligned_t,
            heads_per_group=heads_per_group,
            o_stride_token=o.stride(0),
            o_stride_head=o.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0),
            out_stride_group=out_buf.stride(0),
            out_stride_token=out_buf.stride(1),
            scale_stride_group=scale_buf.stride(0) if quantize else 0,
            scale_stride_k=scale_buf.stride(2) if quantize else 0,
            fp8_max=fp8_max,
            eps=1e-10,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_DIM=rope_dim,
            HALF_ROPE=rope_dim // 2,
            QUANTIZE=quantize,
            TPB=tpb,
            HPB=hpb,
            N_HCHUNKS=heads_per_group // hpb,
            MASK_T=mask_t,
            num_stages=1,
            num_warps=1,
        )
    else:
        _fused_inv_rope_fp8_quant_native[(tma_aligned_t, n_groups * heads_per_group)](
            o,
            positions,
            cos_sin_cache,
            out_buf,
            scale_buf,
            num_tokens,
            heads_per_group=heads_per_group,
            o_stride_token=o.stride(0),
            o_stride_head=o.stride(1),
            cache_stride_pos=cos_sin_cache.stride(0),
            out_stride_group=out_buf.stride(0),
            out_stride_token=out_buf.stride(1),
            scale_stride_group=scale_buf.stride(0) if quantize else 0,
            scale_stride_k=scale_buf.stride(2) if quantize else 0,
            fp8_max=fp8_max,
            eps=1e-10,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            NOPE_DIM=nope_dim,
            HALF_ROPE=rope_dim // 2,
            QUANTIZE=quantize,
            TMA_ALIGNED_SCALES=tma_aligned_scales,
            num_stages=1,
            num_warps=1,
        )
    return output, scales


@torch.library.custom_op("mojo_npu_triton_a5::fused_inv_rope_fp8_quant_fwd", mutates_args=())
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
    blocks = d // quant_group_size
    inner = ((blocks + 3) // 4 if tma_aligned_scales else blocks) if quantize else 0
    aligned = _tma_aligned(num_tokens) if quantize else num_tokens
    out_buf = o.new_empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn if quantize else o.dtype,
    )
    if quantize:
        scale_buf = o.new_empty(
            (n_groups * inner * aligned,),
            dtype=torch.int32 if tma_aligned_scales else torch.float32,
        ).as_strided((n_groups, num_tokens, inner), (inner * aligned, 1, aligned))
        return out_buf.transpose(0, 1), scale_buf.transpose(0, 1)
    return out_buf.transpose(0, 1), torch.empty(
        0, dtype=torch.float32, device=positions.device
    )
