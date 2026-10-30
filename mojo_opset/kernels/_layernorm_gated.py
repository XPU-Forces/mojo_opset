"""Shared Triton-Ascend forward kernel for fused gated normalization.

Adapted from SGLang v0.5.22 ``python/sglang/kernels/ops/attention/fla/layernorm_gated.py``
(kernel L74-209, wrapper L234-348) so that LayerNorm/RMSNorm, gating, affine and the
FP8 per-token epilogue stay inside one kernel launch on Ascend NPUs.
"""

import os

from functools import lru_cache

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores

# Batch-invariant dispatch pins the row tile to this value, matching SGLang.
MAX_ROWS_PER_BLOCK = 4
BATCH_INVARIANT_ENV = "LN_GATED_BATCH_INVARIANT"
_ROWS_PER_BLOCK_ENV = "LN_GATED_ROWS_PER_BLOCK"
_NUM_WARPS_ENV = "LN_GATED_NUM_WARPS"
_TRUE_VALUES = {"1", "true", "yes", "on"}
# The reference kernel supports at most 64 KiB per normalization group.
MAX_FUSED_BYTES = 65536
# A strided 3D gate is staged as one [tokens, heads, N] block above this element
# count: with per-row addressing the DMA engine issues one scattered segment per
# row, which measured 526us versus 315us on A5 for a 524288x128 BF16 gate.
Z3D_TILE_MIN_ELEMENTS = 32 * 1024 * 1024
# Larger groups keep occupancy with a 64-row tile, which measurably loses both
# vector and MTE utilisation on streaming tensors of at least 32M elements
# (AIV vector ratio 68.4% -> 91.9%, MTE2 72.6% -> 97.5% on A2 at 32768x128).
STREAMING_ELEMENTS = 32 * 1024 * 1024


@triton.jit
def _layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    Z_3D_TILE: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    Q=None,
    S=None,
    QUANT: tl.constexpr = False,
    QUANT_TOKEN_GROUP: tl.constexpr = 1,
    Y_IS_BF16: tl.constexpr = False,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols[None, :] < N
    col_offsets = cols[None, :] + group * N
    if N == BLOCK_N:
        weight = tl.load(W + col_offsets).to(tl.float32)
    else:
        weight = tl.load(W + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        if N == BLOCK_N:
            bias = tl.load(B + col_offsets).to(tl.float32)
        else:
            bias = tl.load(B + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    num_row_tasks = (M + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    # One stage is the measured optimum; a two-stage range was equal on the
    # plain shapes and slower on the grouped ones.
    for row_task_id in tl.range(pid, num_row_tasks, num_programs, num_stages=1):
        row_start = row_task_id * ROWS_PER_BLOCK
        rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
        # Predicated vector accesses cost roughly twice an unpredicated one on
        # Ascend, so only the final row tile pays for a mask; every earlier tile
        # covers complete rows and loads/stores without predicates.
        full_rows = EVEN_ROWS or row_start + ROWS_PER_BLOCK < M
        if EVEN_ROWS:
            mask = col_mask
        else:
            mask = (rows[:, None] < M) & col_mask
        x_offsets = rows[:, None] * stride_x_row + col_offsets
        if N == BLOCK_N and full_rows:
            x = tl.load(X + x_offsets).to(tl.float32)
        elif N == BLOCK_N:
            x = tl.load(X + x_offsets, mask=mask, other=0.0).to(tl.float32)
        elif full_rows:
            x = tl.where(col_mask, tl.load(X + x_offsets), 0.0).to(tl.float32)
        else:
            x = tl.load(X + x_offsets, mask=mask, other=0.0).to(tl.float32)

        if HAS_Z:
            if Z_3D_TILE:
                z_tokens = row_start // Z_HEADS + tl.arange(0, ROWS_PER_BLOCK // Z_HEADS)
                z_heads = tl.arange(0, Z_HEADS)
                z_offsets = (
                    z_tokens[:, None, None] * stride_z_token
                    + z_heads[None, :, None] * stride_z_head
                    + col_offsets[None, :]
                )
                z_mask = ((z_tokens[:, None, None] * Z_HEADS + z_heads[None, :, None]) < M) & (cols[None, None, :] < N)
            elif Z_IS_3D:
                z_offsets = (
                    (rows[:, None] // Z_HEADS) * stride_z_token
                    + (rows[:, None] % Z_HEADS) * stride_z_head
                    + col_offsets
                )
                z_mask = mask
            else:
                z_offsets = rows[:, None] * stride_z_row + col_offsets
                z_mask = mask

        if HAS_Z and not NORM_BEFORE_GATE:
            if full_rows and N == BLOCK_N:
                z = tl.load(Z + z_offsets)
            else:
                z = tl.load(Z + z_offsets, mask=z_mask, other=0.0)
            if Z_3D_TILE:
                z = tl.reshape(z, (ROWS_PER_BLOCK, BLOCK_N))
            z = z.to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                x *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                x *= tl.sigmoid(z)

        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N
            if EVEN_ROWS and N == BLOCK_N:
                xbar = x - mean[:, None]
            else:
                xbar = tl.where(mask, x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
        else:
            # x already holds zeros in every padded lane (masked load or select).
            xbar = x
            var = tl.sum(xbar * xbar, axis=1) / N

        rstd = tl.rsqrt(var + eps)
        if IS_RMS_NORM:
            tl.store(Rstd + group * M + rows, rstd, mask=rows < M)
        else:
            tl.store(Mean + group * M + rows, mean, mask=rows < M)
            tl.store(Rstd + group * M + rows, rstd, mask=rows < M)

        if IS_RMS_NORM:
            y = x * rstd[:, None] * weight
        else:
            y = (x - mean[:, None]) * rstd[:, None] * weight
        if HAS_BIAS:
            y += bias

        if HAS_Z and NORM_BEFORE_GATE:
            if full_rows and N == BLOCK_N:
                z = tl.load(Z + z_offsets)
            else:
                z = tl.load(Z + z_offsets, mask=z_mask, other=0.0)
            if Z_3D_TILE:
                z = tl.reshape(z, (ROWS_PER_BLOCK, BLOCK_N))
            z = z.to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                y *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                y *= tl.sigmoid(z)

        if QUANT:
            # The epilogue must quantise the value the unfused path would see, so
            # round to the output dtype before the amax.  A BF16 round trip on the
            # tile costs 26.5us per call on A5, while the equivalent integer RNE on
            # the fp32 bit pattern costs 5us and keeps the tile in one layout.
            if Y_IS_BF16:
                bits = y.to(tl.int32, bitcast=True)
                y = ((bits + 32767 + ((bits >> 16) & 1)) & -65536).to(tl.float32, bitcast=True)
            else:
                y = y.to(Y.dtype.element_ty).to(tl.float32)
            if not (EVEN_ROWS and N == BLOCK_N):
                y = tl.where(mask, y, 0.0)
            if QUANT_TOKEN_GROUP > 1:
                # Two tokens per tile: reducing both tokens in one pass measured
                # 27us against 52us for one token per tile on A5.
                rows_per_token: tl.constexpr = ROWS_PER_BLOCK // QUANT_TOKEN_GROUP
                y3 = tl.reshape(y, (QUANT_TOKEN_GROUP, rows_per_token, BLOCK_N))
                amax = tl.max(tl.max(tl.abs(y3), axis=2), axis=1)
                # torch lowers tensor / 448.0 to a multiply by the fp32 reciprocal
                # on Ascend, so the reciprocal form keeps the scale bit-exact.
                scale = tl.maximum(amax.to(tl.float32), 1e-10) * (1.0 / 448.0)
                recip = tl.reshape(
                    tl.broadcast_to(
                        tl.reshape(1.0 / scale, (QUANT_TOKEN_GROUP, 1, 1)),
                        (QUANT_TOKEN_GROUP, rows_per_token, BLOCK_N),
                    ),
                    (ROWS_PER_BLOCK, BLOCK_N),
                )
            else:
                scale = tl.maximum(tl.max(tl.max(tl.abs(y), axis=1), axis=0), 1e-10) * (1.0 / 448.0)
                recip = 1.0 / scale
            # |y| <= amax holds for every element of the tile, so the scaled value
            # stays inside +-448*(1+2ulp) and never reaches the 464 midpoint where
            # e4m3fn would overflow to NaN; the reference's clamp is a no-op here
            # and removing it saved 30us per call on A5.
            quantized = (y * recip).to(Q.dtype.element_ty)
            q_offsets = rows[:, None] * N + col_offsets
            if EVEN_ROWS and N == BLOCK_N:
                tl.store(Q + q_offsets, quantized)
            else:
                tl.store(Q + q_offsets, quantized, mask=mask)
            if QUANT_TOKEN_GROUP > 1:
                tl.store(S + row_task_id * QUANT_TOKEN_GROUP + tl.arange(0, QUANT_TOKEN_GROUP), scale)
            else:
                tl.store(S + row_task_id, scale)
        else:
            y_offsets = rows[:, None] * stride_y_row + col_offsets
            if full_rows and N == BLOCK_N:
                tl.store(Y + y_offsets, y)
            else:
                tl.store(Y + y_offsets, y, mask=mask)


def is_batch_invariant() -> bool:
    """Read the batch-invariant switch on every dispatch, like SGLang's mode flag."""

    return os.getenv(BATCH_INVARIANT_ENV, "").strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=None)
def vector_core_count() -> int:
    return int(get_num_cores("vector"))


def calc_rows_per_block(M: int, N: int, ngroups: int = 1) -> int:
    """Pick a row tile for the kernel from the normalization width and row count.

    Batch-invariant dispatch ignores both so that one row always reduces with the
    same tile shape, keeping results independent of the batch size.
    """

    if is_batch_invariant():
        return MAX_ROWS_PER_BLOCK
    override = os.getenv(_ROWS_PER_BLOCK_ENV)
    if override:
        return int(override)
    if N <= 128:
        # A grouped normalization already fills the grid along its second axis,
        # so the shorter tile leaves more room for overlapped DMA.
        rows_per_block = 32 if (ngroups > 1 or M * N >= STREAMING_ELEMENTS) else 64
    elif N <= 256:
        rows_per_block = 32
    elif N <= 512:
        rows_per_block = 8
    elif N <= 1024:
        rows_per_block = 4
    else:
        rows_per_block = 2
    return min(rows_per_block, triton.next_power_of_2(M))


def _layer_norm_fwd_impl(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
    quant_heads: int = 0,
    *,
    use_z3d_tile: bool = False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    if N % group_size:
        raise ValueError(f"group_size {group_size} must divide the feature dim {N}")
    ngroups = N // group_size
    if x.stride(-1) != 1:
        raise ValueError("the last stride of x must be 1")

    z_is_3d = z is not None and z.ndim == 3
    if z is not None:
        if z.stride(-1) != 1:
            raise ValueError("the last stride of z must be 1")
        if z_is_3d:
            if z.shape[0] * z.shape[1] != M or z.shape[2] != N:
                raise ValueError("3D z must satisfy T*H == M and D == N")
        elif z.shape != (M, N):
            raise ValueError("2D z must have the same shape as x")
    if weight.shape != (N,) or weight.stride(-1) != 1:
        raise ValueError("weight must be a contiguous [N] tensor")
    if bias is not None and (bias.shape != (N,) or bias.stride(-1) != 1):
        raise ValueError("bias must be a contiguous [N] tensor")

    max_fused = MAX_FUSED_BYTES // x.element_size()
    # Tile exactly on the group width when rounding up would idle a large share
    # of every vector instruction (192 -> 256 keeps only 75% of the lanes busy).
    # Triton-Ascend accepts non-power-of-two block shapes on the vector path.
    exact = min(max_fused, group_size)
    block_n = exact if triton.next_power_of_2(exact) * 8 >= exact * 9 else triton.next_power_of_2(exact)
    if group_size > block_n:
        raise RuntimeError("gated normalization does not support a feature dim >= 64 KiB")
    num_warps = min(max(block_n // 256, 1), 8)
    if os.getenv(_NUM_WARPS_ENV):
        num_warps = int(os.environ[_NUM_WARPS_ENV])
    rows_per_block = calc_rows_per_block(M, block_n, ngroups)

    mean = None if is_rms_norm else torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    quant_args = {}
    if quant_heads:
        if ngroups != 1 or M % quant_heads:
            raise ValueError("quant_heads requires one normalization group and M % quant_heads == 0")
        # Two tokens per tile halve the per-tile cost of the token-wide reduction;
        # the odd tail keeps one token so the last tile stays a single token.
        token_group = 2 if M % (2 * quant_heads) == 0 else 1
        rows_per_block = quant_heads * token_group
        num_warps = max(min(num_warps * quant_heads // rows_per_block, 16), 1)
        q = torch.empty((M // quant_heads, quant_heads * N), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((M // quant_heads, 1), dtype=torch.float32, device=x.device)
        quant_args = {
            "Q": q,
            "S": s,
            "QUANT": True,
            "QUANT_TOKEN_GROUP": token_group,
            "Y_IS_BF16": x.dtype == torch.bfloat16,
        }
    else:
        if out is None:
            out = torch.empty_like(x)
        q = s = None
    # The quantised epilogue never touches Y; reuse x as its dtype carrier.
    y = x if quant_heads else out

    row_tasks = triton.cdiv(M, rows_per_block)
    # Quant tiles are short enough that queueing two blocks per vector core
    # overlaps the epilogue conversion and the FP8 stores with the next tile.
    grid = (min(vector_core_count() * (2 if quant_heads else 1), row_tasks), ngroups)
    meta = {
        "BLOCK_N": block_n,
        "ROWS_PER_BLOCK": rows_per_block,
        "HAS_BIAS": bias is not None,
        "HAS_Z": z is not None,
        "Z_IS_3D": z_is_3d,
        "Z_HEADS": z.shape[1] if z_is_3d else 1,
        "Z_3D_TILE": (use_z3d_tile and z_is_3d and rows_per_block % z.shape[1] == 0 and M * N >= Z3D_TILE_MIN_ELEMENTS),
        "NORM_BEFORE_GATE": norm_before_gate,
        "IS_RMS_NORM": is_rms_norm,
        "ACTIVATION": activation,
        "EVEN_ROWS": M % rows_per_block == 0,
    }
    _layer_norm_fwd_kernel[grid](
        x,
        y,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0) if out is not None else 0,
        z.stride(0) if z is not None and not z_is_3d else 0,
        z.stride(0) if z_is_3d else 0,
        z.stride(1) if z_is_3d else 0,
        M,
        group_size,
        eps,
        num_warps=num_warps,
        **meta,
        **quant_args,
    )
    return ((q, s) if quant_heads else out), mean, rstd
