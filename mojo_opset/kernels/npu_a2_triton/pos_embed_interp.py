"""Bilinear position-embedding interpolation with spatial-merge reorder (Ascend Triton).

Port of vLLM ``_bilinear_pos_embed_kernel`` (``vllm/model_executor/models/qwen3_vl.py``,
``_bilinear_pos_embed_kernel`` at L186-253) to triton-ascend.

Semantics kept identical to upstream:
  * the spatial-merge reorder is encoded in the address arithmetic, so the output already
    has the ``combined.permute(0, 2, 1, 3, 4)`` layout of the eager reference;
  * the four bilinear weights use the *same* subtraction chain as upstream
    (``w11 = dh*dw``, ``w10 = dh - w11``, ``w01 = dw - w11``, ``w00 = 1.0 - dh - w01``) --
    rewriting them as products changes the float rounding order;
  * the weights are cast to the output dtype *before* the multiply-accumulate, so the
    FP32 and BF16 rounding paths match the eager reference;
  * ``h_scale``/``w_scale`` are precomputed on the host exactly as upstream
    (``(G-1)/(size-1)`` for ``size > 1``, otherwise ``0.0``);
  * the temporal dimension is recomputed rather than materialised, which makes ``t > 1``
    bit-identical to ``repeat(t, 1)`` of the ``t == 1`` result.

Ascend-specific choices:
  * ``grid = t*h*w`` on GPU would be up to 268800, above the Ascend core-dimension limit.
    The grid is therefore capped at ``num_programs`` and each program strides over its
    share of rows.
  * ``h_frac``/``w_frac`` are non-negative, so ``floor`` is exactly an int32 truncation.
"""

from typing import List
from typing import Sequence
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

# Keep the 4-corner gather tile small enough to leave room for double buffering on a
# 192 KB UB core.
_MAX_BLOCK_D = 2048


def _axis_scale(num_grid: int, size: int) -> float:
    """Upstream :281-282. ``size == 1`` degenerates to scale 0."""
    return float(num_grid - 1) / float(size - 1) if size > 1 else 0.0


def _pick_block_d(hidden: int) -> int:
    return min(_MAX_BLOCK_D, triton.next_power_of_2(hidden))


@libentry()
@triton.jit
def _pos_embed_interp_kernel(
    embed_ptr,
    out_ptr,
    h,
    w,
    h_scale,
    w_scale,
    total_rows,
    num_grid: tl.constexpr,
    m_size: tl.constexpr,
    hidden: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_spatial = h * w
    num_blocks_w = w // m_size
    m_sq: tl.constexpr = m_size * m_size
    # One program handles one output row; rows are strided across programs so the grid
    # stays under the Ascend core-dimension limit.
    row_id = pid
    while row_id < total_rows:
        # Temporal dimension: decode against h*w so every frame repeats the same spatial
        # pattern (bit-identical to repeat(t, 1)).
        spatial_idx = row_id % total_spatial

        block_idx = spatial_idx // m_sq
        local_idx = spatial_idx % m_sq
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // m_size
        lc = local_idx % m_size
        row = br * m_size + lr
        col = bc * m_size + lc

        h_frac = row.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        # h_frac/w_frac are non-negative: truncation == floor.
        hf = h_frac.to(tl.int32)
        wf = w_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, num_grid - 1)
        wc = tl.minimum(wf + 1, num_grid - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * num_grid + wf) * hidden
        off01 = (hf * num_grid + wc) * hidden
        off10 = (hc * num_grid + wf) * hidden
        off11 = (hc * num_grid + wc) * hidden
        out_off = row_id * hidden

        out_dtype = out_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        for d in range(0, hidden, block_d):
            cols = d + tl.arange(0, block_d)
            mask = cols < hidden

            e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
            e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
            e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
            e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

            val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

            tl.store(out_ptr + out_off + cols, val, mask=mask)

        row_id += num_programs


def pos_embed_interp_impl(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fused bilinear pos-embed interpolation for one (t, h, w) grid.

    Returns ``(t * h * w, hidden_dim)`` in spatial-merge order.
    """
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    hidden = embed_weight.shape[1]
    total_rows = t * h * w
    out = torch.empty(total_rows, hidden, device=embed_weight.device, dtype=dtype)
    if total_rows == 0:
        return out

    h_scale = _axis_scale(num_grid_per_side, h)
    w_scale = _axis_scale(num_grid_per_side, w)

    block_d = _pick_block_d(hidden)
    num_programs = min(total_rows, get_num_cores("vector") * 4)

    _pos_embed_interp_kernel[(num_programs,)](
        embed_weight,
        out,
        h,
        w,
        h_scale,
        w_scale,
        total_rows,
        num_grid_per_side,
        m_size,
        hidden,
        block_d,
    )
    return out


# ---------------------------------------------------------------------------
# Two-kernel temporal broadcast (t > 1)
# ---------------------------------------------------------------------------
# The fused kernel above recomputes the bilinear interpolation for *every* frame,
# so with t > 1 it reads the 4-neighbour corners t times more than necessary and
# writes rows one by one. Measured on Ascend910B3 the (16,100,168) case sat at
# MBU 0.110 -- the 4-corner gather dominates.
#
# Since all t frames are bit-identical by construction (the row index is decoded
# modulo ``h*w``, so frame ``f`` reproduces the pattern of frame 0 exactly), the
# same result can be produced in two kernels:
#
#   seg1: interpolate the single frame   (h*w rows)
#   seg2: broadcast it to t frames        (pure write, t*h*w rows)
#
# seg2 is write-only, so it can use a wide contiguous store: load one block of
# source rows once, then store it into ``frame_repeat`` consecutive frames. That
# removes both the redundant 4-corner reads and the per-row store.
#
# Measured (A2, BF16, fill_ 1479.7 GB/s, (16,100,168)):
#   fused single kernel : 3834.8 us  MBU 0.110
#   two-kernel          :  650.7 us  MBU 0.649   (seg1 192.1 + seg2 458.7)
# and bit-identical to the fused path on 6 grids.
#
# The organizer explicitly allowed this shape (issue #64 answer 3: two kernels
# are fine, scored as the sum of both kernels' Duration).
#
# ``block_rows`` is platform-dependent: A2 (910B3) peaks at 8 while A5 (950PR) keeps
# improving up to 48, so the value is picked per device (see ``_broadcast_block_rows``).
_BROADCAST_BLOCK_ROWS = 8
_BROADCAST_BLOCK_ROWS_A5 = 52
_BROADCAST_FRAME_REPEAT = 16
# A2 measurement (bf16, block_rows=8):
#   (3,16,16) h*w=256   fused 15.0us -> two-kernel 10.9us  (1.38x)
#   (3,60,80) h*w=4800  fused 205.4us -> two-kernel 83.1us (2.47x)
# t == 1 grids cannot use the broadcast at all, so the gate only guards t > 1.
_BROADCAST_MIN_ROWS = 256


@triton.jit
def _pos_embed_broadcast_kernel(
    src_ptr,
    out_ptr,
    rows,
    t,
    block_rows: tl.constexpr,
    frame_repeat: tl.constexpr,
    hidden: tl.constexpr,
    block_d: tl.constexpr,
):
    """Copy one frame of ``rows`` rows into ``t`` frames of ``out_ptr``.

    ``block_rows`` source rows are loaded once and stored into ``frame_repeat``
    frames before advancing, which is what lifts the write efficiency from ~76%
    to ~91% of the ``fill_`` probe.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    blocks_per_frame = (rows + block_rows - 1) // block_rows
    groups = t // frame_repeat

    block_id = pid
    while block_id < blocks_per_frame * groups:
        group = block_id // blocks_per_frame
        k = block_id % blocks_per_frame

        row_idx = k * block_rows + tl.arange(0, block_rows)[:, None]
        cols = tl.arange(0, block_d)[None, :]
        mask = (row_idx < rows) & (cols < hidden)

        val = tl.load(src_ptr + row_idx * hidden + cols, mask=mask)

        for j in tl.static_range(0, frame_repeat):
            frame = group * frame_repeat + j
            tl.store(
                out_ptr + (frame * rows + row_idx) * hidden + cols,
                val,
                mask=mask,
            )

        block_id += num_programs


def _pick_frame_repeat(t: int) -> int:
    """Largest divisor of ``t`` that is <= the tuned repeat factor."""
    repeat = min(t, _BROADCAST_FRAME_REPEAT)
    while repeat > 1 and t % repeat:
        repeat //= 2
    return repeat


def _is_a5_device() -> bool:
    """Whether the current device is the A5 (950) family.

    A5 reuses this module, so the same code serves both families; only a couple of
    tuning constants differ. Device name is queried once and cached.
    """
    global _A5_CACHE
    if _A5_CACHE is None:
        try:
            import torch

            name = torch.npu.get_device_name(0)
            _A5_CACHE = "950" in name
        except Exception:
            _A5_CACHE = False
    return _A5_CACHE


_A5_CACHE = None


def _broadcast_block_rows(dtype=None) -> int:
    """Tuned ``block_rows`` for the current platform and element width.

    The tile lives in UB, so a wider element halves the affordable row count:
    FP32 needs 2x the bytes of BF16. The tuned values below were measured in BF16,
    which is the dtype the scoring matrix uses; FP32 is scaled down to fit.
    """
    rows = _BROADCAST_BLOCK_ROWS_A5 if _is_a5_device() else _BROADCAST_BLOCK_ROWS
    if dtype is not None and getattr(dtype, "itemsize", 2) > 2:
        rows = max(1, rows // 2)
    return rows


def _broadcast_ub_budget_bytes() -> int:
    """UB size available for the broadcast tile.

    A2/A3 have 192 KB per core; A5 (950) has 256 KB.
    """
    return 256 * 1024 if _is_a5_device() else 192 * 1024


def _use_two_kernel(t: int, h: int, w: int, hidden: int, dtype=None) -> bool:
    """Whether the two-kernel path pays off for this shape.

    Small grids emit too few rows for the second launch to amortise: the fused
    kernel is already a single launch and the extra kernel only adds overhead.

    UB budget: the broadcast tile holds ``block_rows * block_d`` elements and
    stores it ``frame_repeat`` times; the compiler needs roughly
    ``2 * block_rows * block_d`` bf16 values live at once. ``block_d`` is kept at
    the packed hidden width -- rounding 1152 up to 2048 trips "ub overflow" at
    compile time.
    """
    if t < 2:
        return False
    if h * w < _BROADCAST_MIN_ROWS:
        return False
    block_rows = _broadcast_block_rows(dtype)
    # two live copies of the tile (source + staged store), element-sized
    item = getattr(dtype, "itemsize", 2) if dtype is not None else 2
    return hidden * block_rows * 2 * item <= _broadcast_ub_budget_bytes()


def pos_embed_interp_broadcast_impl(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Two-kernel variant of :func:`pos_embed_interp_impl` for ``t > 1``."""
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    hidden = embed_weight.shape[1]
    rows = h * w
    total_rows = t * rows

    frame = torch.empty(rows, hidden, device=embed_weight.device, dtype=dtype)
    h_scale = _axis_scale(num_grid_per_side, h)
    w_scale = _axis_scale(num_grid_per_side, w)
    block_d = _pick_block_d(hidden)

    num_programs = min(rows, get_num_cores("vector") * 4)
    _pos_embed_interp_kernel[(num_programs,)](
        embed_weight,
        frame,
        h,
        w,
        h_scale,
        w_scale,
        rows,
        num_grid_per_side,
        m_size,
        hidden,
        block_d,
    )

    out = torch.empty(total_rows, hidden, device=embed_weight.device, dtype=dtype)
    frame_repeat = _pick_frame_repeat(t)
    block_rows = _broadcast_block_rows(dtype)
    blocks_per_frame = (rows + block_rows - 1) // block_rows
    groups = t // frame_repeat
    broadcast_programs = min(blocks_per_frame * groups, get_num_cores("vector") * 4)
    # Keep the broadcast tile at the packed hidden width. Rounding up to the next
    # power of two (2048 for hidden=1152) doubles the live UB footprint and makes
    # the compiler fail with "ub overflow".
    _pos_embed_broadcast_kernel[(broadcast_programs,)](
        frame,
        out,
        rows,
        t,
        block_rows,
        frame_repeat,
        hidden,
        hidden,
    )
    return out


@torch.library.custom_op("mojo_npu_triton_a2::pos_embed_interp", mutates_args=())
def pos_embed_interp(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if _use_two_kernel(t, h, w, embed_weight.shape[1], dtype):
        return pos_embed_interp_broadcast_impl(
            embed_weight, t, h, w, num_grid_per_side, m_size, dtype
        )
    return pos_embed_interp_impl(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)


@pos_embed_interp.register_fake
def _pos_embed_interp_fake(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return torch.empty(t * h * w, embed_weight.shape[1], device=embed_weight.device, dtype=dtype)


# ---------------------------------------------------------------------------
# mojo_opset provider contract
# ---------------------------------------------------------------------------
# ``mojo_opset/functions/_dispatch.py::_load_kernel_binding`` resolves an op as:
#     provider.OPS[op_id] -> module  ->  getattr(module, f"{op_id}_fwd")
# The official entry (mojo_opset#523 ``functions/vision_pos_embed_interp.py``)
# calls ``load_impl("vision_pos_embed_interp", implementation)``, so this module
# must expose ``vision_pos_embed_interp_fwd`` under exactly that op id.
def vision_pos_embed_interp_fwd(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Official-provider entry: fused single launch or two-kernel broadcast."""
    return pos_embed_interp(
        embed_weight, t, h, w, num_grid_per_side, m_size, dtype
    )

# ---------------------------------------------------------------------------
# Multi-image merged launch
# ---------------------------------------------------------------------------
# Upstream ``fast_pos_embed_interpolate`` loops over ``grid_thw`` on the host and
# concatenates the per-grid results.  The merged variant launches once.
#
# Bit-exactness: a *global* row index ``g`` runs ``0 .. total_rows-1``; the owning image is
# found by scanning the per-image row counts, and the in-image row index is
# ``g - image_out_off``.  For image j that sequence is ``0 .. rows_j-1``, i.e. exactly what a
# standalone call would produce, and every row goes through the same instruction sequence.
# The result is therefore bit-identical to ``torch.cat([single(g) for g in grids])``.
#
# **Ascend constraint**: on triton-ascend 3.6.0 a scalar produced by ``tl.load`` of a device
# tensor is *not* usable in address arithmetic inside a multi-dimension grid -- the value reads
# correctly but the resulting store address is wrong (observed: image 1 writes nothing, image 0
# correct).  The merged kernel therefore uses a **1-D grid** and derives the image index in
# scalar arithmetic instead of relying on ``program_id(1)``.


@libentry()
@triton.jit
def _pos_embed_interp_multi_kernel(
    embed_ptr,
    out_ptr,
    h_ptr,
    w_ptr,
    h_scale_ptr,
    w_scale_ptr,
    rows_ptr,
    out_off_ptr,
    num_images: tl.constexpr,
    num_grid: tl.constexpr,
    m_size: tl.constexpr,
    hidden: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Total output rows = last image offset + last image rows.
    total_rows = tl.load(out_off_ptr + (num_images - 1)) + tl.load(rows_ptr + (num_images - 1))

    g = pid
    while g < total_rows:
        # Locate the owning image.  ``num_images`` is a small constexpr, so the scan is fully
        # unrolled at compile time; a nested ``while`` here fails to lower on triton-ascend.
        image = tl.zeros((), dtype=tl.int32)
        image_off = tl.zeros((), dtype=tl.int32)
        for i in tl.static_range(num_images):
            cand_off = tl.load(out_off_ptr + i)
            image = tl.where(g >= cand_off, i, image)
            image_off = tl.where(g >= cand_off, cand_off, image_off)

        row_id = g - image_off

        h = tl.load(h_ptr + image)
        w = tl.load(w_ptr + image)
        h_scale = tl.load(h_scale_ptr + image)
        w_scale = tl.load(w_scale_ptr + image)

        total_spatial = h * w
        num_blocks_w = w // m_size
        m_sq: tl.constexpr = m_size * m_size

        spatial_idx = row_id % total_spatial

        block_idx = spatial_idx // m_sq
        local_idx = spatial_idx % m_sq
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // m_size
        lc = local_idx % m_size
        row = br * m_size + lr
        col = bc * m_size + lc

        h_frac = row.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = h_frac.to(tl.int32)
        wf = w_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, num_grid - 1)
        wc = tl.minimum(wf + 1, num_grid - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * num_grid + wf) * hidden
        off01 = (hf * num_grid + wc) * hidden
        off10 = (hc * num_grid + wf) * hidden
        off11 = (hc * num_grid + wc) * hidden
        out_off = g * hidden

        out_dtype = out_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        for d in range(0, hidden, block_d):
            cols = d + tl.arange(0, block_d)
            mask = cols < hidden

            e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
            e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
            e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
            e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

            val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

            tl.store(out_ptr + out_off + cols, val, mask=mask)

        g += num_programs


def pos_embed_interp_multi_impl(
    embed_weight: torch.Tensor,
    grid_thw: Sequence[Sequence[int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Merged single-launch bilinear pos-embed interpolation over several grids.

    Bit-identical to ``torch.cat([pos_embed_interp_impl(embed, *g, ...) for g in grid_thw])``.
    """
    grids = [tuple(int(v) for v in g) for g in grid_thw]
    hidden = embed_weight.shape[1]
    if not grids:
        return embed_weight.new_empty((0, hidden), dtype=dtype)

    for t, h, w in grids:
        assert h % m_size == 0 and w % m_size == 0, (
            f"h={h} and w={w} must be divisible by m_size={m_size}"
        )

    device = embed_weight.device
    rows = [t * h * w for t, h, w in grids]
    total_out = sum(rows)

    out = torch.empty(total_out, hidden, device=device, dtype=dtype)
    if total_out == 0:
        return out

    h_t = torch.tensor([g[1] for g in grids], device=device, dtype=torch.int32)
    w_t = torch.tensor([g[2] for g in grids], device=device, dtype=torch.int32)
    hs_t = torch.tensor(
        [_axis_scale(num_grid_per_side, g[1]) for g in grids], device=device, dtype=torch.float32
    )
    ws_t = torch.tensor(
        [_axis_scale(num_grid_per_side, g[2]) for g in grids], device=device, dtype=torch.float32
    )
    rows_t = torch.tensor(rows, device=device, dtype=torch.int32)

    offs = [0]
    for r in rows[:-1]:
        offs.append(offs[-1] + r)
    off_t = torch.tensor(offs, device=device, dtype=torch.int32)

    block_d = _pick_block_d(hidden)
    num_programs = min(total_out, get_num_cores("vector") * 4)

    _pos_embed_interp_multi_kernel[(num_programs,)](
        embed_weight,
        out,
        h_t,
        w_t,
        hs_t,
        ws_t,
        rows_t,
        off_t,
        len(grids),
        num_grid_per_side,
        m_size,
        hidden,
        block_d,
    )
    return out


def grids_to_flat(grid_thw: Sequence[Sequence[int]]) -> List[int]:
    """``[(t, h, w), ...]`` -> ``[t0, h0, w0, t1, h1, w1, ...]``."""
    flat: List[int] = []
    for g in grid_thw:
        if len(g) != 3:
            raise ValueError(f"each grid must be (t, h, w), got {tuple(g)}")
        flat.extend(int(v) for v in g)
    return flat


def flat_to_grids(grid_thw_flat: Sequence[int]) -> List[Tuple[int, int, int]]:
    """``[t0, h0, w0, t1, h1, w1, ...]`` -> ``[(t, h, w), ...]``."""
    if len(grid_thw_flat) % 3:
        raise ValueError("grid_thw_flat length must be a multiple of 3")
    return [
        (int(grid_thw_flat[i]), int(grid_thw_flat[i + 1]), int(grid_thw_flat[i + 2]))
        for i in range(0, len(grid_thw_flat), 3)
    ]


# Additional provider contracts. ``_dispatch._load_kernel_binding`` always looks up
# ``f"{op_id}_fwd"``, so every op id declared in ``OPS`` needs one of these.
def pos_embed_interp_fwd(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Alias of :func:`vision_pos_embed_interp_fwd` under the short op id."""
    return pos_embed_interp(
        embed_weight, t, h, w, num_grid_per_side, m_size, dtype
    )


def pos_embed_interp_multi_fwd(
    embed_weight: torch.Tensor,
    grid_thw: Sequence[Sequence[int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Provider entry for the merged multi-grid launch."""
    return pos_embed_interp_multi_impl(
        embed_weight, grid_thw, num_grid_per_side, m_size, dtype
    )
