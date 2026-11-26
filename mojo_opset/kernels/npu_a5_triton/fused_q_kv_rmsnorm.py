"""Ascend A5 (950) Triton fused Q/KV RMSNorm and MXFP8 Q quantisation.

Structure relative to the vllm original (grid-y task axis, one shared
``BLOCK=next_pow2(max(q_size, kv_size))`` for both halves):

1. Q and KV are launched as two kernels with their own ``BLOCK``, so each side
   runs at its own width and ``SIZE`` stays a compile-time constant (the
   original's ``if pid_task == 0`` degrades it to a runtime value).
2. ``RPP`` rows are merged per program, tile ``[RPP, BLOCK]``; the norm
   reduction runs along the tile's column axis and yields ``RPP`` inverse RMS
   values at once. ``RPP`` follows the row count (``_RPP_STEPS``).
3. Below ``_SPLIT_MIN_ROWS`` / ``_SPLIT_MIN_ROWS_QUANT`` rows the two launches
   cost more than the structure saves, so the original single-kernel layout is
   used instead.
4. An empty batch returns empty outputs without launching.

The split thresholds (128 rows for the norm entry, 64 for the quant entry), the
``RPP`` ladder (1/2/4/8 at <=2/<=32/<=128/above) and the ``RPP=8`` UB ceiling
(``RPP=16`` fails with "ub overflow") come from a cold-cache calibration on this
arch.

The MXFP8 chain and the F8_128x4 swizzle are the vllm ``query_quant`` ones
verbatim: 32-element groups, UE8M0 exponent with mantissa carry and a clamp to
254, and scales scattered with ``[row/128, group/4, row%32, row%128/32, group%4]``.
"""

import torch
import triton
import triton.language as tl

_RPP_STEPS = ((2, 1), (32, 2), (128, 4), (None, 8))
_SPLIT_MIN_ROWS = 128
_SPLIT_MIN_ROWS_QUANT = 64
_NUM_WARPS = 8
# The row tile's next_pow2 width stops compiling above this; the native kernel
# takes over (its own limit is 8192, enforced at the entry).
_ROW_MAX_WIDTH = 2048


@triton.jit
def _fused_q_kv_rmsnorm_native(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


@triton.jit
def _norm_row(x, weight, row, stride, eps, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    w = tl.load(weight + cols, cols < SIZE, 0).to(tl.float32)
    values = tl.load(x + row * stride + cols, cols < SIZE, 0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(values * values, 0) / SIZE + eps)
    return values * rrms * w


@triton.jit
def mxfp8_scale_offset(row, group, GROUPS: tl.constexpr):
    """F8_128x4 byte offset: [row/128, group/4, row%32, row%128/32, group%4]."""
    offset = (row // 128 * triton.cdiv(GROUPS, 4) + group // 4) * 512
    return offset + row % 32 * 16 + row % 128 // 32 * 4 + group % 4


def mxfp8_scale_bytes(tokens, width):
    """Bytes of F8_128x4 scales for ``tokens`` rows of ``width`` MXFP8 values."""
    return ((tokens + 127) // 128) * 128 * (((width // 32) + 3) // 4) * 4


@triton.jit
def _mxfp8_sf(amax):
    """Group amax -> UE8M0 exponent byte. Single definition for every quant path.

    A NaN or infinite amax saturates the exponent field; it takes the 254 clamp
    like the torch reference rather than the zero branch (on this backend
    ``nan <= 0`` is true, so the ordering matters).
    """
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    return tl.where((exponent == 255) & (mantissa != 0), 254, sf)


@triton.jit
def _mxfp8_inv(sf):
    """UE8M0 exponent byte -> 2^(127 - sf) (0 when sf == 0)."""
    inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
    return inv_bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["num_tokens"])
def _q_kv_norm_quant_native(
    q,
    kv,
    qw,
    kvw,
    qo,
    kvo,
    scales,
    num_tokens,
    q_stride,
    kv_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if tl.program_id(1) == 0:
        groups = tl.arange(0, BLOCK // 32)
        if row < num_tokens:
            y = _norm_row(q, qw, row, q_stride, eps, Q_SIZE, BLOCK)
            y = y.to(q.dtype.element_ty).to(tl.float32)
            grouped = tl.reshape(y, (BLOCK // 32, 32))
            amax = tl.max(tl.abs(grouped), 1, propagate_nan=tl.PropagateNan.ALL)
            sf = _mxfp8_sf(amax)
            inv_scale = _mxfp8_inv(sf)
            quantized = tl.reshape(grouped * inv_scale[:, None], (BLOCK,))
            tl.store(qo + row * Q_SIZE + cols, quantized, cols < Q_SIZE)
        else:
            sf = tl.full((BLOCK // 32,), 0, tl.uint32)
        # Padding rows and groups past q_size must read as scale 0: consumers load
        # the whole F8_128x4 block.
        padded_groups: tl.constexpr = triton.cdiv(Q_SIZE // 32, 4) * 4
        sf = tl.where(groups < Q_SIZE // 32, sf, 0)
        offsets = mxfp8_scale_offset(row, groups, Q_SIZE // 32)
        tl.store(scales + offsets, sf, groups < padded_groups)
    elif row < num_tokens:
        y = _norm_row(kv, kvw, row, kv_stride, eps, KV_SIZE, BLOCK)
        tl.store(kvo + row * KV_SIZE + cols, y, cols < KV_SIZE)


@triton.jit
def _rmsnorm_rows(
    x_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    in_stride,
    out_stride,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    RPP: tl.constexpr,
):
    # Row offsets are int64 like the native kernel: a 32-bit pid * stride
    # wraps past 2^31 on large prefill batches.
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * RPP + tl.arange(0, RPP)
    row_mask = rows < num_rows
    cols = tl.arange(0, BLOCK)
    col_mask = cols < SIZE
    m = row_mask[:, None] & col_mask[None, :]
    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + rows[:, None] * in_stride + cols[None, :], mask=m, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / SIZE
    rrms = tl.rsqrt(variance + eps)
    y = x * rrms[:, None] * w[None, :]
    tl.store(
        out_ptr + rows[:, None] * out_stride + cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=m,
    )


@triton.jit(do_not_specialize=["num_rows", "padded_rows"])
def _rmsnorm_rows_quant(
    x_ptr,
    w_ptr,
    qo_ptr,
    scales_ptr,
    num_rows,
    padded_rows,
    in_stride,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    RPP: tl.constexpr,
):
    # Row offsets are int64 like the native kernel: a 32-bit pid * stride
    # wraps past 2^31 on large prefill batches.
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * RPP + tl.arange(0, RPP)
    row_mask = rows < num_rows
    cols = tl.arange(0, BLOCK)
    # The F8_128x4 scale byte offset is q_size/32 times smaller than the element
    # offset, so int32 covers every batch the int64 row offset is needed for;
    # int64 offsets into the uint8 scale buffer fault on this backend.
    rows_s = rows.to(tl.int32)
    if pid * RPP < num_rows:
        col_mask = cols < SIZE
        m = row_mask[:, None] & col_mask[None, :]
        w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rows[:, None] * in_stride + cols[None, :], mask=m, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=1) / SIZE
        rrms = tl.rsqrt(variance + eps)
        y = (x * rrms[:, None] * w[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
        grouped = tl.reshape(y, (RPP, BLOCK // 32, 32))
        # propagate_nan=ALL matches torch.amax (NaN propagates); on finite input it
        # is bit-identical and lowers to a single vcmax per group.
        amax = tl.max(tl.abs(grouped), 2, propagate_nan=tl.PropagateNan.ALL)
        sf = _mxfp8_sf(amax)
        inv_scale = _mxfp8_inv(sf)
        quantized = tl.reshape(grouped * inv_scale[:, :, None], (RPP, BLOCK))
        tl.store(qo_ptr + rows[:, None] * SIZE + cols[None, :], quantized, mask=m)
    else:
        # The whole tile sits in the 128-row padding tail: only scale zeros are
        # needed there (the original writes them too).
        sf = tl.zeros((RPP, BLOCK // 32), tl.uint32)
    groups = tl.arange(0, BLOCK // 32)
    padded_groups: tl.constexpr = triton.cdiv(SIZE // 32, 4) * 4
    sf = tl.where(groups[None, :] < SIZE // 32, sf, 0)
    offsets = mxfp8_scale_offset(rows_s[:, None], groups[None, :], SIZE // 32)
    tl.store(
        scales_ptr + offsets,
        sf,
        mask=(rows_s[:, None] < padded_rows) & (groups[None, :] < padded_groups),
    )
    # Group columns between SIZE // 32 and the 4-group padding of the F8_128x4
    # layout are not covered by the tile above; write their zero scales too.
    PAD_GROUPS: tl.constexpr = padded_groups - SIZE // 32
    if PAD_GROUPS > 0:
        pad_groups = SIZE // 32 + tl.arange(0, 4)
        tl.store(
            scales_ptr + mxfp8_scale_offset(rows_s[:, None], pad_groups[None, :], SIZE // 32),
            tl.zeros((RPP, 4), tl.uint8),
            mask=(rows_s[:, None] < padded_rows) & (pad_groups[None, :] < padded_groups),
        )



@triton.jit(do_not_specialize=["num_rows", "padded_rows"])
def _rmsnorm_rows_quant_split(
    x_ptr,
    w_ptr,
    qo_ptr,
    scales_ptr,
    num_rows,
    padded_rows,
    in_stride,
    eps,
    SIZE: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    RPP: tl.constexpr,
):
    """Exact-coverage Q quantisation for ``SIZE = A + B`` (both powers of two).

    The masked tile spends ``next_pow2(SIZE) - SIZE`` lanes on nothing in the
    square/sum/mul/abs/max/quantise chain (768 of 2048 at the production
    ``q_size=1280``). Splitting the row into two contiguous blocks keeps the
    32-element group structure and removes those lanes.
    """
    # Row offsets are int64 like the native kernel: a 32-bit pid * stride
    # wraps past 2^31 on large prefill batches.
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * RPP + tl.arange(0, RPP)
    row_mask = rows < num_rows
    rm = row_mask[:, None]
    # The F8_128x4 scale byte offset is q_size/32 times smaller than the element
    # offset, so int32 covers every batch the int64 row offset is needed for;
    # int64 offsets into the uint8 scale buffer fault on this backend.
    rows_s = rows.to(tl.int32)
    GA: tl.constexpr = A // 32
    GB: tl.constexpr = B // 32
    GROUPS: tl.constexpr = SIZE // 32
    ca = tl.arange(0, A)
    cb = tl.arange(0, B)
    if pid * RPP < num_rows:
        wa = tl.load(w_ptr + ca).to(tl.float32)
        wb = tl.load(w_ptr + A + cb).to(tl.float32)
        base = rows[:, None] * in_stride
        xa = tl.load(x_ptr + base + ca[None, :], mask=rm, other=0.0).to(tl.float32)
        xb = tl.load(x_ptr + base + A + cb[None, :], mask=rm, other=0.0).to(tl.float32)
        variance = (tl.sum(xa * xa, 1) + tl.sum(xb * xb, 1)) / SIZE
        rrms = tl.rsqrt(variance + eps)
        dt = x_ptr.dtype.element_ty
        ya = (xa * rrms[:, None] * wa[None, :]).to(dt).to(tl.float32)
        yb = (xb * rrms[:, None] * wb[None, :]).to(dt).to(tl.float32)

        ga3 = tl.reshape(ya, (RPP, GA, 32))
        gb3 = tl.reshape(yb, (RPP, GB, 32))
        sfa = _mxfp8_sf(tl.max(tl.abs(ga3), 2, propagate_nan=tl.PropagateNan.ALL))
        sfb = _mxfp8_sf(tl.max(tl.abs(gb3), 2, propagate_nan=tl.PropagateNan.ALL))
        inva = _mxfp8_inv(sfa)
        invb = _mxfp8_inv(sfb)
        tl.store(
            qo_ptr + rows[:, None] * SIZE + ca[None, :],
            tl.reshape(ga3 * inva[:, :, None], (RPP, A)),
            mask=rm,
        )
        tl.store(
            qo_ptr + rows[:, None] * SIZE + A + cb[None, :],
            tl.reshape(gb3 * invb[:, :, None], (RPP, B)),
            mask=rm,
        )
    else:
        # Entire tile inside the 128-row padding tail: only scale zeros are needed.
        sfa = tl.zeros((RPP, GA), tl.uint32)
        sfb = tl.zeros((RPP, GB), tl.uint32)

    padded_groups: tl.constexpr = triton.cdiv(GROUPS, 4) * 4
    gid_a = tl.arange(0, GA)
    gid_b = GA + tl.arange(0, GB)
    row_ok = rows_s[:, None] < padded_rows
    tl.store(
        scales_ptr + mxfp8_scale_offset(rows_s[:, None], gid_a[None, :], GROUPS),
        sfa,
        mask=row_ok & (gid_a[None, :] < padded_groups),
    )
    tl.store(
        scales_ptr + mxfp8_scale_offset(rows_s[:, None], gid_b[None, :], GROUPS),
        sfb,
        mask=row_ok & (gid_b[None, :] < padded_groups),
    )
    PAD_GROUPS: tl.constexpr = padded_groups - GROUPS
    if PAD_GROUPS > 0:
        pad_groups = GROUPS + tl.arange(0, 4)
        tl.store(
            scales_ptr + mxfp8_scale_offset(rows_s[:, None], pad_groups[None, :], GROUPS),
            tl.zeros((RPP, 4), tl.uint8),
            mask=row_ok & (pad_groups[None, :] < padded_groups),
        )


def _pick_rpp(num_rows):
    for limit, rpp in _RPP_STEPS:
        if limit is None or num_rows <= limit:
            return rpp
    return 1


def _fused_q_kv_rmsnorm_impl(qr, kv, q_weight, kv_weight, eps):
    q_out = torch.empty(qr.shape, dtype=qr.dtype, device=qr.device)
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    num_tokens = qr.shape[0]
    if num_tokens == 0:
        return q_out, kv_out
    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    if num_tokens < _SPLIT_MIN_ROWS or max(q_size, kv_size) > _ROW_MAX_WIDTH:
        block_size = triton.next_power_of_2(max(q_size, kv_size))
        _fused_q_kv_rmsnorm_native[(num_tokens, 2)](
            qr, q_out, q_weight, qr.stride(0), q_out.stride(0),
            kv, kv_out, kv_weight, kv.stride(0), kv_out.stride(0),
            eps,
            Q_SIZE=q_size, KV_SIZE=kv_size, BLOCK_SIZE=block_size,
            num_warps=8 if block_size >= 2048 else 4,
        )
        return q_out, kv_out
    rpp = _pick_rpp(num_tokens)
    _rmsnorm_rows[(triton.cdiv(num_tokens, rpp),)](
        qr, q_weight, q_out, num_tokens, qr.stride(0), q_out.stride(0), eps,
        SIZE=q_size, BLOCK=triton.next_power_of_2(q_size), RPP=rpp, num_warps=_NUM_WARPS,
    )
    _rmsnorm_rows[(triton.cdiv(num_tokens, rpp),)](
        kv, kv_weight, kv_out, num_tokens, kv.stride(0), kv_out.stride(0), eps,
        SIZE=kv_size, BLOCK=triton.next_power_of_2(kv_size), RPP=rpp, num_warps=_NUM_WARPS,
    )
    return q_out, kv_out


def _fused_q_kv_rmsnorm_quant_impl(qr, kv, q_weight, kv_weight, eps):
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    q8 = torch.empty(qr.shape, dtype=torch.float8_e4m3fn, device=qr.device)
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    padded_tokens = ((tokens + 127) // 128) * 128
    scales = torch.empty(
        mxfp8_scale_bytes(tokens, q_size), dtype=torch.uint8, device=qr.device
    )
    if tokens == 0:
        return q8, scales, kv_out
    if tokens < _SPLIT_MIN_ROWS_QUANT or max(q_size, kv_size) > _ROW_MAX_WIDTH:
        block = triton.next_power_of_2(max(q_size, kv_size))
        _q_kv_norm_quant_native[(padded_tokens, 2)](
            qr, kv, q_weight, kv_weight, q8, kv_out, scales, tokens,
            qr.stride(0), kv.stride(0), eps,
            Q_SIZE=q_size, KV_SIZE=kv_size, BLOCK=block,
            num_warps=8 if block >= 2048 else 4,
        )
        return q8, scales, kv_out
    rpp = _pick_rpp(tokens)
    # Exact-coverage tile when q_size = 1024 + B (B a power of two >= 32), else the
    # masked next_pow2 tile. Production q_lora_rank=1280 -> A=1024, B=256.
    b_side = q_size - 1024
    if q_size > 1024 and b_side >= 32 and (b_side & (b_side - 1)) == 0:
        _rmsnorm_rows_quant_split[(triton.cdiv(padded_tokens, rpp),)](
            qr, q_weight, q8, scales, tokens, padded_tokens, qr.stride(0), eps,
            SIZE=q_size, A=1024, B=b_side, RPP=rpp, num_warps=_NUM_WARPS,
        )
    else:
        _rmsnorm_rows_quant[(triton.cdiv(padded_tokens, rpp),)](
            qr, q_weight, q8, scales, tokens, padded_tokens, qr.stride(0), eps,
            SIZE=q_size, BLOCK=triton.next_power_of_2(q_size), RPP=rpp,
            num_warps=_NUM_WARPS,
        )
    _rmsnorm_rows[(triton.cdiv(tokens, rpp),)](
        kv, kv_weight, kv_out, tokens, kv.stride(0), kv_out.stride(0), eps,
        SIZE=kv_size, BLOCK=triton.next_power_of_2(kv_size), RPP=rpp, num_warps=_NUM_WARPS,
    )
    return q8, scales, kv_out


@torch.library.custom_op("mojo_npu_triton_a5::fused_q_kv_rmsnorm_fwd", mutates_args=())
def fused_q_kv_rmsnorm_fwd(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fused_q_kv_rmsnorm_impl(
        qr.contiguous(), kv.contiguous(), q_weight, kv_weight, eps
    )


@fused_q_kv_rmsnorm_fwd.register_fake
def _fused_q_kv_rmsnorm_fake(qr, kv, q_weight, kv_weight, eps):
    return torch.empty_like(qr, memory_format=torch.contiguous_format), torch.empty_like(
        kv, memory_format=torch.contiguous_format
    )


@torch.library.custom_op("mojo_npu_triton_a5::fused_q_kv_rmsnorm_quant_fwd", mutates_args=())
def fused_q_kv_rmsnorm_quant_fwd(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fused_q_kv_rmsnorm_quant_impl(
        qr.contiguous(), kv.contiguous(), q_weight, kv_weight, eps
    )


@fused_q_kv_rmsnorm_quant_fwd.register_fake
def _fused_q_kv_rmsnorm_quant_fake(qr, kv, q_weight, kv_weight, eps):
    scales = qr.new_empty(
        (mxfp8_scale_bytes(qr.shape[0], qr.shape[1]),), dtype=torch.uint8
    )
    return (
        qr.new_empty(qr.shape, dtype=torch.float8_e4m3fn),
        scales,
        torch.empty_like(kv, memory_format=torch.contiguous_format),
    )
