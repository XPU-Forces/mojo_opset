"""A2 Triton implementation of Qwen3-VL vision position-embedding interpolation.

The interpolation semantics mirror upstream vLLM ``_bilinear_pos_embed_kernel``:
bilinear sampling of a ``G x G`` learned table and spatial-merge reordering are
folded into the table addresses.

For Ascend, three structural choices matter:

* the per-row scalar index math (integer div/mod, floor, int64 multiplies) is
  precomputed once on the host and cached;
* the spatial pattern is interpolated once and the resulting vector is stored
  into each of the ``t`` frames, so table traffic does not grow with ``t``;
* work is split per image with a 2-D grid.  ``h*w``, ``t``, the output base and
  the spatial-metadata base are scalar per image, so the per-frame output
  address is one vector add instead of a 2-D int64 multiply.

The kernel is persistent along spatial rows, so a large video does not create
one program per output row.
"""

from __future__ import annotations

from typing import Optional
from typing import Sequence

import torch
import triton
import triton.language as tl

# Dedicated row-indexed GM->UB intrinsic (not the generic tl.load lowering).
try:
    from triton.language.extra.cann.extension import index_select_simd as _vpe_index_select_simd
except ImportError:
    _vpe_index_select_simd = getattr(tl, "index_select_simd", None)

# Row metadata is small but shape-dependent; keep the cache bounded so ad-hoc
# sweeps in tests do not grow host state without limit.
_META_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}
_META_CACHE_LIMIT = 32


@triton.jit
def _bilinear_pos_embed_kernel(
    embed_ptr,
    out_ptr,
    index_ptr,
    weight_ptr,
    image_ptr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    TOTAL_SPATIAL,
):
    """Interpolate one image's spatial rows, then store each value into frames.

    ``image`` is int32 ``[num_images, 4]``: ``[hw, t, output_base_row,
    spatial_base]``.  ``index``/``weight`` are neighbour-major
    ``[4, TOTAL_SPATIAL]`` arrays of table element offsets and float32 weights.
    Weights are cast to the output dtype before the multiply-accumulate, as
    upstream does.
    """
    pid = tl.program_id(0)
    image = tl.program_id(1)
    out_dtype = out_ptr.dtype.element_ty

    hw = tl.load(image_ptr + image * 4 + 0)
    frames = tl.load(image_ptr + image * 4 + 1)
    out_base = tl.load(image_ptr + image * 4 + 2)
    spatial_base = tl.load(image_ptr + image * 4 + 3)

    stride = hw * HIDDEN_DIM
    for block in tl.range(pid, tl.cdiv(hw, BLOCK_M), NUM_PROGS):
        rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < hw
        srows = spatial_base + rows

        off00 = tl.load(index_ptr + srows, mask=row_mask, other=0)
        off01 = tl.load(index_ptr + TOTAL_SPATIAL + srows, mask=row_mask, other=0)
        off10 = tl.load(index_ptr + 2 * TOTAL_SPATIAL + srows, mask=row_mask, other=0)
        off11 = tl.load(index_ptr + 3 * TOTAL_SPATIAL + srows, mask=row_mask, other=0)
        w00 = tl.load(weight_ptr + srows, mask=row_mask, other=0.0)
        w01 = tl.load(weight_ptr + TOTAL_SPATIAL + srows, mask=row_mask, other=0.0)
        w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0)
        w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0)

        off0 = (out_base + rows).to(tl.int32) * HIDDEN_DIM
        for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
            cols = d + tl.arange(0, BLOCK_D)
            col_mask = cols < HIDDEN_DIM
            mask = row_mask[:, None] & col_mask[None, :]

            e00 = tl.load(embed_ptr + off00[:, None] + cols[None, :], mask=mask, other=0.0)
            e01 = tl.load(embed_ptr + off01[:, None] + cols[None, :], mask=mask, other=0.0)
            e10 = tl.load(embed_ptr + off10[:, None] + cols[None, :], mask=mask, other=0.0)
            e11 = tl.load(embed_ptr + off11[:, None] + cols[None, :], mask=mask, other=0.0)

            value = w00[:, None] * e00 + w01[:, None] * e01 + w10[:, None] * e10 + w11[:, None] * e11

            for frame in tl.range(0, frames):
                active = row_mask & (frame < frames)
                off = off0 + frame * stride
                tl.store(
                    out_ptr + off[:, None] + cols[None, :],
                    value,
                    mask=active[:, None] & col_mask[None, :],
                )


@triton.jit
def _bilinear_pos_embed_static_kernel(
    embed_ptr,
    out_ptr,
    index_ptr,
    weight_ptr,
    image_ptr,
    add_ptr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    TOTAL_SPATIAL: tl.constexpr,
    HW: tl.constexpr,
    FRAMES: tl.constexpr,
    DW_ZERO: tl.constexpr,
    CLEAN: tl.constexpr = 0,
    AFFINE_2X2: tl.constexpr = 0,
    FMA: tl.constexpr = 0,
    INDEX_SELECT: tl.constexpr = 0,
    ADD: tl.constexpr = 0,
):
    if INDEX_SELECT == 1:
        tl.static_assert(BLOCK_M == 4 and BLOCK_D == 1152 and HIDDEN_DIM == 1152)
        tl.static_assert(HW == 4800 and TOTAL_SPATIAL == 4800)
        tl.static_assert(FRAMES == 1 and CLEAN == 1 and DW_ZERO == 0)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        off00 = tl.load(index_ptr + rows)
        off01 = tl.load(index_ptr + TOTAL_SPATIAL + rows)
        off10 = tl.load(index_ptr + 2 * TOTAL_SPATIAL + rows)
        off11 = tl.load(index_ptr + 3 * TOTAL_SPATIAL + rows)
        out_dtype = out_ptr.dtype.element_ty
        w00 = tl.load(weight_ptr + rows)
        w01 = tl.load(weight_ptr + TOTAL_SPATIAL + rows)
        w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + rows)
        w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + rows)
        # Transport BF16 payloads as raw 16-bit words; no numeric cast.
        words = embed_ptr  # bf16 direct (int16 bitcast unsupported by BiShengIR)
        i00 = off00 // HIDDEN_DIM
        i01 = off01 // HIDDEN_DIM
        i10 = off10 // HIDDEN_DIM
        i11 = off11 // HIDDEN_DIM
        e00 = _vpe_index_select_simd(
            words, 0, i00, (2304, HIDDEN_DIM), (-1, 0), (-1, BLOCK_D)
          )
        e10 = _vpe_index_select_simd(
            words, 0, i10, (2304, HIDDEN_DIM), (-1, 0), (-1, BLOCK_D)
          )
        e01 = _vpe_index_select_simd(
            words, 0, i01, (2304, HIDDEN_DIM), (-1, 0), (-1, BLOCK_D)
          )
        e11 = _vpe_index_select_simd(
            words, 0, i11, (2304, HIDDEN_DIM), (-1, 0), (-1, BLOCK_D)
          )
        if FMA == 1:
            value = tl.math.fma(
                w11[:, None], e11,
                tl.math.fma(
                    w10[:, None], e10,
                    tl.math.fma(w01[:, None], e01, w00[:, None] * e00),
                ),
            )
        else:
            value = w00[:, None] * e00 + w01[:, None] * e01 + w10[:, None] * e10 + w11[:, None] * e11
        cols = tl.arange(0, BLOCK_D)
        off = rows.to(tl.int32) * HIDDEN_DIM
        tl.store(out_ptr + off[:, None] + cols[None, :], value.to(out_dtype))
    elif AFFINE_2X2 == 2:
        # Per-pixel scalar-base form (vLLM CUDA/Triton layout: one pixel per
        # program, scalar table offsets, one 1-D column vector).  BLOCK_M
        # pixels are unrolled with tl.static_range so the launch grid stays at
        # hw / BLOCK_M while every address remains scalar + [BLOCK_D].
        tl.static_assert(CLEAN == 1 and FRAMES == 1)
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        out_dtype = out_ptr.dtype.element_ty
        for _i in tl.static_range(0, BLOCK_M):
            pix = pid * BLOCK_M + _i
            o00 = tl.load(index_ptr + pix)
            o01 = tl.load(index_ptr + TOTAL_SPATIAL + pix)
            o10 = tl.load(index_ptr + 2 * TOTAL_SPATIAL + pix)
            o11 = tl.load(index_ptr + 3 * TOTAL_SPATIAL + pix)
            w00 = tl.load(weight_ptr + pix)
            w01 = tl.load(weight_ptr + TOTAL_SPATIAL + pix)
            w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + pix)
            w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + pix)
            e00 = tl.load(embed_ptr + o00 + cols)
            e10 = tl.load(embed_ptr + o10 + cols)
            if DW_ZERO == 1:
                val = w00 * e00 + w10 * e10
            else:
                e01 = tl.load(embed_ptr + o01 + cols)
                e11 = tl.load(embed_ptr + o11 + cols)
                val = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11
            tl.store(out_ptr + pix * HIDDEN_DIM + cols, val.to(out_dtype))
    elif AFFINE_2X2 == 1:
        tl.static_assert(BLOCK_M == 4 and BLOCK_D == 1152)
        tl.static_assert(HW == 4800 and FRAMES == 1 and DW_ZERO == 0 and CLEAN == 1)
        tile = tl.program_id(0)
        br = tile // 40
        bc = tile % 40
        # Scalar-only geometry. Integer floors equal the existing FP32 floors
        # for every coordinate of this particular 60x80 grid.
        hf0 = (2 * br * 47) // 59
        hf1 = ((2 * br + 1) * 47) // 59
        wf0 = (2 * bc * 47) // 79
        wf1 = ((2 * bc + 1) * 47) // 79
        hc0 = tl.minimum(hf0 + 1, 47)
        hc1 = tl.minimum(hf1 + 1, 47)
        wc0 = tl.minimum(wf0 + 1, 47)
        wc1 = tl.minimum(wf1 + 1, 47)
        shf = (hf1 - hf0) * 48 * BLOCK_D
        shc = (hc1 - hc0) * 48 * BLOCK_D
        swf = (wf1 - wf0) * BLOCK_D
        swc = (wc1 - wc0) * BLOCK_D
        # Each base below is a scalar pointer, never an address tensor.
        b00 = embed_ptr + (hf0 * 48 + wf0) * BLOCK_D
        b01 = embed_ptr + (hf0 * 48 + wc0) * BLOCK_D
        b10 = embed_ptr + (hc0 * 48 + wf0) * BLOCK_D
        b11 = embed_ptr + (hc0 * 48 + wc0) * BLOCK_D
        p00 = tl.make_block_ptr(
            base=b00, shape=(2, 2, BLOCK_D), strides=(shf, swf, 1),
            offsets=(0, 0, 0), block_shape=(2, 2, BLOCK_D), order=(2, 1, 0))
        p01 = tl.make_block_ptr(
            base=b01, shape=(2, 2, BLOCK_D), strides=(shf, swc, 1),
            offsets=(0, 0, 0), block_shape=(2, 2, BLOCK_D), order=(2, 1, 0))
        p10 = tl.make_block_ptr(
            base=b10, shape=(2, 2, BLOCK_D), strides=(shc, swf, 1),
            offsets=(0, 0, 0), block_shape=(2, 2, BLOCK_D), order=(2, 1, 0))
        p11 = tl.make_block_ptr(
            base=b11, shape=(2, 2, BLOCK_D), strides=(shc, swc, 1),
            offsets=(0, 0, 0), block_shape=(2, 2, BLOCK_D), order=(2, 1, 0))
        e00 = tl.load(p00)
        e10 = tl.load(p10)
        e01 = tl.load(p01)
        e11 = tl.load(p11)
        rr = 2 * tl.arange(0, 2)[:, None] + tl.arange(0, 2)[None, :]
        rows = tile * BLOCK_M + rr
        out_dtype = out_ptr.dtype.element_ty
        w00 = tl.load(weight_ptr + rows)
        w01 = tl.load(weight_ptr + TOTAL_SPATIAL + rows)
        w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + rows)
        w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + rows)
        if FMA == 2:
            # Explicit full-shape broadcast before the fused multiply-add.
            # ``w[:, :, None]`` alone stays a broadcasted operand, which the
            # BiShengIR vectorizer lowers to VMUL + BAR.V + VADD instead of a
            # single VFMA.  Materialising the weight tile first gives the
            # pattern matcher two same-shape tensors to fuse.
            _b00 = tl.broadcast_to(w00[:, :, None], (2, 2, BLOCK_D))
            _b01 = tl.broadcast_to(w01[:, :, None], (2, 2, BLOCK_D))
            _b10 = tl.broadcast_to(w10[:, :, None], (2, 2, BLOCK_D))
            _b11 = tl.broadcast_to(w11[:, :, None], (2, 2, BLOCK_D))
            value = tl.math.fma(
                _b11, e11,
                tl.math.fma(_b10, e10, tl.math.fma(_b01, e01, _b00 * e00)))
        elif FMA == 1:
            value = tl.math.fma(
                w11[:, :, None], e11,
                tl.math.fma(
                    w10[:, :, None], e10,
                    tl.math.fma(w01[:, :, None], e01, w00[:, :, None] * e00),
                ),
            )
        elif FMA == 3:
            # Two-level tree: (p00 + p01) + (p10 + p11) keeps the two halves
            # independent so they can share UB buffers instead of overwriting
            # a single accumulator three times.
            value = ((w00[:, :, None] * e00 + w01[:, :, None] * e01)
                     + (w10[:, :, None] * e10 + w11[:, :, None] * e11))
        else:
            value = (w00[:, :, None] * e00 + w01[:, :, None] * e01
                     + w10[:, :, None] * e10 + w11[:, :, None] * e11)
        dst = tl.make_block_ptr(
            base=out_ptr + tile * BLOCK_M * BLOCK_D,
            shape=(2, 2, BLOCK_D), strides=(2 * BLOCK_D, BLOCK_D, 1),
            offsets=(0, 0, 0), block_shape=(2, 2, BLOCK_D), order=(2, 1, 0))
        tl.store(dst, value.to(out_dtype))
    else:
        """Interpolate one image's spatial rows, then store each value into frames.

        ``image`` is int32 ``[num_images, 4]``: ``[hw, t, output_base_row,
        spatial_base]``.  ``index``/``weight`` are neighbour-major
        ``[4, TOTAL_SPATIAL]`` arrays of table element offsets and float32 weights.
        Weights are cast to the output dtype before the multiply-accumulate, as
        upstream does.
        """
        pid = tl.program_id(0)
        image = tl.program_id(1)
        out_dtype = out_ptr.dtype.element_ty

        hw = HW
        frames = FRAMES
        out_base = 0
        spatial_base = 0

        stride = hw * HIDDEN_DIM
        block = pid // tl.cdiv(HIDDEN_DIM, BLOCK_D)
        d = (pid % tl.cdiv(HIDDEN_DIM, BLOCK_D)) * BLOCK_D
        if block * BLOCK_M < hw:
            rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
            row_mask = rows < hw
            srows = spatial_base + rows

            if CLEAN == 1:
                off00 = tl.load(index_ptr + srows)
                off01 = tl.load(index_ptr + TOTAL_SPATIAL + srows)
                off10 = tl.load(index_ptr + 2 * TOTAL_SPATIAL + srows)
                off11 = tl.load(index_ptr + 3 * TOTAL_SPATIAL + srows)
                w00 = tl.load(weight_ptr + srows)
                w01 = tl.load(weight_ptr + TOTAL_SPATIAL + srows)
                w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + srows)
                w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + srows)
            else:
                off00 = tl.load(index_ptr + srows, mask=row_mask, other=0)
                off01 = tl.load(index_ptr + TOTAL_SPATIAL + srows, mask=row_mask, other=0)
                off10 = tl.load(index_ptr + 2 * TOTAL_SPATIAL + srows, mask=row_mask, other=0)
                off11 = tl.load(index_ptr + 3 * TOTAL_SPATIAL + srows, mask=row_mask, other=0)
                w00 = tl.load(weight_ptr + srows, mask=row_mask, other=0.0)
                w01 = tl.load(weight_ptr + TOTAL_SPATIAL + srows, mask=row_mask, other=0.0)
                w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0)
                w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0)

            off0 = (out_base + rows).to(tl.int32) * HIDDEN_DIM
            cols = d + tl.arange(0, BLOCK_D)
            col_mask = cols < HIDDEN_DIM
            mask = row_mask[:, None] & col_mask[None, :]

            if CLEAN == 1:
                e00 = tl.load(embed_ptr + off00[:, None] + cols[None, :])
                e10 = tl.load(embed_ptr + off10[:, None] + cols[None, :])
                if DW_ZERO == 1:
                    # dw 恒为 0 ⇒ w01 = w11 = 0，e01/e11 的 load 一并省掉
                    value = w00[:, None] * e00 + w10[:, None] * e10
                else:
                    e01 = tl.load(embed_ptr + off01[:, None] + cols[None, :])
                    e11 = tl.load(embed_ptr + off11[:, None] + cols[None, :])
                    if FMA == 1:
                        value = tl.math.fma(
                            w11[:, None], e11,
                            tl.math.fma(
                                w10[:, None], e10,
                                tl.math.fma(w01[:, None], e01, w00[:, None] * e00),
                            ),
                        )
                    else:
                        value = w00[:, None] * e00 + w01[:, None] * e01 + w10[:, None] * e10 + w11[:, None] * e11
            else:
                e00 = tl.load(embed_ptr + off00[:, None] + cols[None, :], mask=mask, other=0.0)
                e10 = tl.load(embed_ptr + off10[:, None] + cols[None, :], mask=mask, other=0.0)

                if DW_ZERO == 1:
                    # dw 恒为 0 ⇒ w01 = w11 = 0，e01/e11 乘零，整个 load 可以省掉
                    value = w00[:, None] * e00 + w10[:, None] * e10
                else:
                    e01 = tl.load(embed_ptr + off01[:, None] + cols[None, :], mask=mask, other=0.0)
                    e11 = tl.load(embed_ptr + off11[:, None] + cols[None, :], mask=mask, other=0.0)
                    p00 = w00[:, None] * e00
                    p01 = w01[:, None] * e01
                    p10 = w10[:, None] * e10
                    p11 = w11[:, None] * e11
                    value = (p00 + p01) + (p10 + p11)

            if CLEAN == 1:
                for frame in tl.static_range(0, FRAMES):
                    if frame < frames:
                        off = off0 + frame * stride
                        if ADD == 1:
                            # keep the intermediate BF16 rounding of the original
                            # non-fused composition: round interp to out dtype first
                            _av = tl.load(add_ptr + off[:, None] + cols[None, :]).to(tl.float32)
                            _rd = value.to(out_ptr.dtype.element_ty).to(tl.float32)
                            tl.store(out_ptr + off[:, None] + cols[None, :],
                                     (_rd + _av).to(out_ptr.dtype.element_ty))
                        else:
                            tl.store(out_ptr + off[:, None] + cols[None, :], value.to(out_dtype))
            else:
                for frame in tl.static_range(0, FRAMES):
                    active = row_mask & (frame < frames)
                    off = off0 + frame * stride
                    tl.store(
                        out_ptr + off[:, None] + cols[None, :],
                        value,
                        mask=active[:, None] & col_mask[None, :],
                    )


def _select_block_d(hidden_dim: int) -> int:
    # Row-per-program first: when hidden is 128-aligned use the whole row as BD
    # (single pass, zero padding).  Otherwise fall back to 512-wide tiles.
    if int(hidden_dim) % 128 == 0:
        return int(hidden_dim)
    if hidden_dim <= 256:
        return 256
    if hidden_dim <= 512:
        return 512
    return 512

def _build_row_metadata(
    embed_weight: torch.Tensor,
    grids: tuple[tuple[int, int, int], ...],
    num_grid: int,
    m_size: int,
    hidden_dim: int,
):
    """Build cached per-image and per-spatial-row metadata.

    The math is the upstream/native float32 path: ``h_scale``/``w_scale`` are
    host floats, fractional coordinates are float32, and the four weights use
    the upstream subtraction form.
    """
    key = (embed_weight.device, grids, num_grid, m_size, hidden_dim)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached

    index_parts = [[], [], [], []]
    weight_parts = [[], [], [], []]
    image_rows = []
    output_row = 0
    spatial_base = 0
    for t, h, w in grids:
        count = h * w
        spatial = torch.arange(count, dtype=torch.int64)

        blocks_w = w // m_size
        block_idx = spatial // (m_size * m_size)
        local_idx = spatial % (m_size * m_size)
        br = block_idx // blocks_w
        bc = block_idx % blocks_w
        lr = local_idx // m_size
        lc = local_idx % m_size
        grid_row = br * m_size + lr
        grid_col = bc * m_size + lc

        h_scale = torch.tensor(float(num_grid - 1) / float(h - 1) if h > 1 else 0.0, dtype=torch.float32)
        w_scale = torch.tensor(float(num_grid - 1) / float(w - 1) if w > 1 else 0.0, dtype=torch.float32)
        h_frac = grid_row.to(torch.float32) * h_scale
        w_frac = grid_col.to(torch.float32) * w_scale

        hf = torch.floor(h_frac).to(torch.int64)
        wf = torch.floor(w_frac).to(torch.int64)
        hc = torch.minimum(hf + 1, torch.tensor(num_grid - 1, dtype=torch.int64))
        wc = torch.minimum(wf + 1, torch.tensor(num_grid - 1, dtype=torch.int64))

        dh = h_frac - hf.to(torch.float32)
        dw = w_frac - wf.to(torch.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        for slot, value in enumerate(
            (
                (hf * num_grid + wf) * hidden_dim,
                (hf * num_grid + wc) * hidden_dim,
                (hc * num_grid + wf) * hidden_dim,
                (hc * num_grid + wc) * hidden_dim,
            )
        ):
            index_parts[slot].append(value.to(torch.int32))
        for slot, value in enumerate((w00, w01, w10, w11)):
            weight_parts[slot].append(value.to(torch.float32))

        image_rows.append((count, t, output_row, spatial_base))
        output_row += t * count
        spatial_base += count

    index = torch.stack([torch.cat(part, dim=0) for part in index_parts], dim=0).reshape(-1).to(embed_weight.device)
    weight = torch.stack([torch.cat(part, dim=0) for part in weight_parts], dim=0).reshape(-1).to(embed_weight.device)
    if embed_weight.dtype == torch.bfloat16:
        # Pre-round to BF16 then widen: bit-identical to the kernel-side
        # (f32 -> bf16 -> f32) vcast chain, moved out of the hot loop.
        weight = weight.to(torch.bfloat16).to(torch.float32)
    image = torch.tensor(image_rows, dtype=torch.int32, device=embed_weight.device).reshape(-1)

    if len(_META_CACHE) >= _META_CACHE_LIMIT:
        _META_CACHE.pop(next(iter(_META_CACHE)))
    _META_CACHE[key] = (index, weight, image)
    return index, weight, image


def _launch(embed_weight: torch.Tensor, grids: Sequence[Sequence[int]], num_grid: int, m_size: int, dtype: torch.dtype, add=None):
    if not grids:
        return torch.empty((0, embed_weight.shape[1]), dtype=dtype, device=embed_weight.device)
    if not embed_weight.is_contiguous():
        embed_weight = embed_weight.contiguous()

    hidden_dim = embed_weight.shape[1]
    total_rows = sum(int(t) * int(h) * int(w) for t, h, w in grids)
    total_spatial = sum(int(h) * int(w) for _, h, w in grids)
    output = torch.empty((total_rows, hidden_dim), dtype=dtype, device=embed_weight.device)

    # ---- slot 快路径：单帧、整行 BD、非 dwzero、块数足够时才启用 -----------------
    index, weight, image = _build_row_metadata(
        embed_weight,
        tuple(tuple(int(v) for v in grid) for grid in grids),
        num_grid,
        m_size,
        hidden_dim,
    )

    # 单图 (t==1) 走 950 已验证的 static kernel；w==num_grid 时 dw 恒为 0，
    # 可整段省掉 e01/e11 的 load 与两项乘加（dwzero）。
    # 单图（任意帧数）走 static kernel：读一次空间行、按 FRAMES 写 t 帧。
    # static kernel 内部本就有 static_range 帧循环，t==1 时与旧行为完全等价。
    if len(grids) == 1:
        _t, _h, _w = [int(x) for x in grids[0]]
        hw = _h * _w
        bd = (
            2048
            if __import__("os").environ.get("MOJO_VPE_BD2048", "0") == "1"
            else (
                int(hidden_dim)
                if (int(hidden_dim) % 128) == 0
                else triton.next_power_of_2(hidden_dim)
            )
        )
        _dw_zero = 1 if (int(num_grid) - 1) == (_w - 1) else 0
        # _BMEQ_ BM sweep marker
        _bm = 4
        _index_select = (
            __import__("os").environ.get("MOJO_VPE_INDEX_SELECT", "0") == "1"
            and (_t, _h, _w) == (1, 60, 80)
            and int(num_grid) == 48 and int(m_size) == 2
            and tuple(embed_weight.shape) == (2304, 1152)
            and embed_weight.dtype == torch.bfloat16 and dtype == torch.bfloat16
            and bd == 1152 and _bm == 4
        )
        if _index_select and _vpe_index_select_simd is None:
            raise RuntimeError("This Triton build does not expose index_select_simd")
        _affine = (
            __import__("os").environ.get("MOJO_VPE_AFFINE", "1") == "1"
            and (_t, _h, _w) == (1, 60, 80)
            and int(num_grid) == 48 and int(m_size) == 2
            and tuple(embed_weight.shape) == (2304, 1152)
            and embed_weight.dtype == torch.bfloat16 and dtype == torch.bfloat16
        )
        _compile_options = {}
        for _kv in __import__("os").environ.get("MOJO_VPE_OPTS", "").split(","):
            _kv = _kv.strip()
            if "=" in _kv:
                _k, _v = _kv.split("=", 1)
                _v = _v.strip()
                _compile_options[_k.strip()] = (
                    True if _v == "1" else False if _v == "0"
                    else int(_v) if _v.lstrip("-").isdigit() else _v)
        if (__import__("os").environ.get("MOJO_VPE_CAST_ALIGN_PROBE", "0") == "1"
                and (_t, _h, _w) == (1, 60, 80) and _dw_zero == 0
                and not _index_select and not _affine):
            _compile_options["disable_size_align_for_cast"] = True
        _bilinear_pos_embed_static_kernel[(triton.cdiv(hw, _bm), 1)](
            embed_weight, output, index, weight, image,
            output if add is None else add,
            hidden_dim, _bm, bd, 1, total_spatial, hw, _t, _dw_zero,
            1 if (_dw_zero == 0 and hw % _bm == 0 and hidden_dim % bd == 0) else 0,
            AFFINE_2X2=(
                2 if (__import__("os").environ.get("MOJO_VPE_SCALAR4", "0") == "1"
                      and (_t, _h, _w) == (1, 60, 80) and _dw_zero == 0
                      and hw % _bm == 0 and hidden_dim % bd == 0)
                else (1 if _affine else 0)),
            FMA=int(__import__("os").environ.get("MOJO_VPE_FMA", "1")),
            INDEX_SELECT=1 if _index_select else 0,
            ADD=1 if (add is not None and _dw_zero == 0 and hw % _bm == 0
                      and hidden_dim % bd == 0) else 0,
            **_compile_options)
        return output

    # FP32 tiles use twice the UB of BF16; 8 rows keeps the four neighbour
    # tiles plus output inside the 254 KB UB budget.
    # UB budget = BM * BD * dtype_size * 5 (four neighbours + value), A2 UB = 192KB:
    #   bf16 BD=row: BM=8 -> 92KB    fp32 BD=row: BM=4 -> 92KB
    block_d = _select_block_d(hidden_dim)
    if int(block_d) == int(hidden_dim):
        block_m = 8 if embed_weight.dtype == torch.bfloat16 else 4
    else:
        block_m = 16 if embed_weight.dtype == torch.bfloat16 else 8
    try:
        from mojo_opset.kernels._npu_triton_utils import get_num_cores

        num_progs = int(get_num_cores("vector"))
    except Exception:
        num_progs = 56
    num_progs = max(1, num_progs)

    _bilinear_pos_embed_kernel[(num_progs, len(grids))](
        embed_weight,
        output,
        index,
        weight,
        image,
        hidden_dim,
        block_m,
        block_d,
        num_progs,
        total_spatial,
    )
    return output


def vision_pos_embed_interp_fwd(
    embed_weight: torch.Tensor,
    grid_thw: Sequence[Sequence[int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype] = None,
    add=None,
) -> torch.Tensor:
    dtype = embed_weight.dtype if dtype is None else dtype
    return _launch(embed_weight, grid_thw, int(num_grid_per_side), int(m_size), dtype, add=add)
