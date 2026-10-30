"""Bilinear position-embedding interpolation with spatial-merge reorder (Ascend Triton).

Port of vLLM ``_bilinear_pos_embed_kernel`` (``vllm/model_executor/models/qwen3_vl.py``,
main@aa31fa0) to triton-ascend.

Design notes (Ascend specific)
------------------------------
* Upstream launches one program per output row (``grid = t*h*w``, up to 268800). On Ascend
  that is both a grid-size risk and a scalar-index-arithmetic bound: every row would pay a
  fresh ``%``/``/`` chain for the spatial-merge decode. Here one program owns a whole
  ``m_size x m_size`` merge block, which is exactly ``m_size*m_size`` **consecutive** output
  rows (``combined.reshape(h//m, m, w//m, m)`` flatten order), so the block decode costs two
  integer divisions per block, and the store covers a contiguous multi-row span.
* One kernel serves both the single-grid and the multi-grid API. ``program_id(1)`` selects
  the image and every per-image scalar is loaded from a tiny device tensor once per
  program; the per-row arithmetic is therefore literally the same instruction sequence for
  both entry points, which is what makes the merged launch bit-identical to per-image
  calls (task requirement 4.4).
* The grid is persistent (``grid = #vector cores``, grid-stride over blocks), matching
  ``rmsnorm.py`` and avoiding Ascend grid-size limits.
* ``h_frac``/``w_frac`` are non-negative, so ``floor`` and truncation agree and an explicit
  ``tl.floor`` before the int32 cast is an exact substitution, not a reordering. The explicit
  floor is also required for correctness: on Ascend 950PR (triton-ascend 3.6.0) the truncating
  f32->i32 cast lowers wrongly and feeds the corner loads garbage offsets, which zeroes whole
  output rows. Cost on A2 (X51Cm), two runs each with independent profiler dirs: +1.7% to
  +3.3% per grid (worst case 3956 us vs 3828 us on the 268800-row video grid).
* Bilinear weights are cast to the output dtype before the multiply-accumulate, mirroring
  upstream :236-240, so the FP32 and BF16 rounding paths stay identical.
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

# 2048 elements keep the 4-corner gather tile at ~32KB of UB per d-block, leaving room for
# double buffering on a 192KB UB core.
_MAX_BLOCK_D = 2048
# Per-image scalar bundles, keyed by the whole grid list. Reused across calls so neither entry
# point pays host-side tensor construction after the first call.
_SCALAR_CACHE: Dict[Tuple, Tuple[Dict[str, torch.Tensor], int]] = {}
_CACHE_LIMIT = 64


def _pick_tuning(cores: int, max_blocks: int, hidden: int) -> Tuple[int, int]:
    """Return ``(block_d, grid_width)``.

    ``block_d`` covers the whole row in one tile when it fits (fewer loop turns), and the
    persistent-grid width decides how many independent block streams are in flight.
    Ascend hides load latency with concurrent programs, not with deep software pipelining,
    so the grid is deliberately allowed to exceed the core count; the per-block work is
    independent, which keeps the extra programs cheap.
    """
    # Measured on 910B3 (A2), grid (16,100,168), hidden=1152: one wide slice beats several
    # narrow ones, because the extra loop turns and per-slice masks cost more than the padded
    # lanes -- block_d 256 -> 5480us, 512 -> 4558us, 1024 -> 4364us, 2048 -> 3786us.
    # A 2D block tile that merges several blocks into one contiguous store was 25% faster on
    # triton-ascend 3.2.x (3047us) but 170x slower on 3.6.0 (730896us) because the masked
    # gather with a per-row base offset lowers pathologically, so it is not shipped here; see
    # triton-lang/triton-ascend#2865 and revisit once that lowering is fixed.
    block_d = min(_MAX_BLOCK_D, triton.next_power_of_2(hidden))
    grid_width = min(max_blocks, max(cores, 1) * 4)
    return block_d, grid_width


def _axis_scale(num_grid: int, size: int) -> float:
    # Same expression as upstream :281-282; size == 1 degenerates to scale 0.
    return float(num_grid - 1) / float(size - 1) if size > 1 else 0.0


@libentry()
@triton.jit
def _pos_embed_interp_kernel(
    embed_ptr,
    out_ptr,
    h_ptr,
    w_ptr,
    h_scale_ptr,
    w_scale_ptr,
    total_blocks_ptr,
    blocks_per_frame_ptr,
    out_row_off_ptr,
    num_grid: tl.constexpr,
    m_size: tl.constexpr,
    hidden: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    image = tl.program_id(1)

    w = tl.load(w_ptr + image)
    h_scale = tl.load(h_scale_ptr + image)
    w_scale = tl.load(w_scale_ptr + image)
    total_blocks = tl.load(total_blocks_ptr + image)
    blocks_per_frame = tl.load(blocks_per_frame_ptr + image)
    out_row_off = tl.load(out_row_off_ptr + image)

    num_blocks_w = w // m_size

    # Contiguous chunk per program, not grid-stride. A strided walk hands each program blocks
    # from all over the output, so every iteration touches a different part of the position
    # table and the 4-neighbour reads miss cache. A contiguous chunk makes one program walk
    # along an output row, where the source rows advance by ~2*w_scale per step, so the
    # neighbour reads stay resident and HBM traffic drops towards the write-only floor.
    chunk = (total_blocks + grid_size - 1) // grid_size
    start = pid * chunk
    stop = tl.minimum(start + chunk, total_blocks)

    for blk in range(start, stop):
        spatial_blk = blk % blocks_per_frame
        br = spatial_blk // num_blocks_w
        bc = spatial_blk % num_blocks_w
        out_row = (out_row_off + blk).to(tl.int64) * (m_size * m_size)

        for lr in tl.static_range(m_size):
            row = br * m_size + lr
            h_frac = row.to(tl.float32) * h_scale
            # floor before the cast: on Ascend 950PR (triton-ascend 3.6.0) the direct
            # f32->i32 truncation lowers incorrectly and feeds the corner loads garbage
            # offsets (whole output rows become exact 0). h_frac >= 0, so truncation and
            # floor are identical here; the explicit floor keeps every backend correct.
            hf = tl.floor(h_frac).to(tl.int32)
            hc = tl.minimum(hf + 1, num_grid - 1)
            dh = h_frac - hf.to(tl.float32)

            for lc in tl.static_range(m_size):
                col = bc * m_size + lc
                w_frac = col.to(tl.float32) * w_scale
                wf = tl.floor(w_frac).to(tl.int32)
                wc = tl.minimum(wf + 1, num_grid - 1)
                dw = w_frac - wf.to(tl.float32)

                w11 = dh * dw
                w10 = dh - w11
                w01 = dw - w11
                w00 = 1.0 - dh - w01

                off00 = (hf * num_grid + wf) * hidden
                off01 = (hf * num_grid + wc) * hidden
                off10 = (hc * num_grid + wf) * hidden
                off11 = (hc * num_grid + wc) * hidden

                # Weights are cast to the output dtype before the MAC (upstream :236-240).
                out_dtype = out_ptr.dtype.element_ty
                w00_c = w00.to(out_dtype)
                w01_c = w01.to(out_dtype)
                w10_c = w10.to(out_dtype)
                w11_c = w11.to(out_dtype)

                dst_row = out_row + lr * m_size + lc
                for d0 in tl.static_range(0, hidden, block_d):
                    cols = d0 + tl.arange(0, block_d)
                    mask = cols < hidden

                    e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
                    e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
                    e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
                    e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

                    val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

                    tl.store(out_ptr + dst_row * hidden + cols, val, mask=mask)


def _check_geometry(h: int, w: int, m_size: int, num_grid: int) -> None:
    # m_size and num_grid are checked first: the divisibility test below divides by m_size, so
    # an m_size of 0 must raise the ValueError instead of a ZeroDivisionError.
    if m_size <= 0:
        raise ValueError(f"m_size={m_size} must be positive")
    if num_grid < 2:
        raise ValueError(f"num_grid_per_side={num_grid} must be >= 2")
    if h <= 0 or w <= 0:
        raise ValueError(f"h={h} and w={w} must be positive")
    if h % m_size or w % m_size:
        raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")


def _scalar_bundle(
    grids: Tuple[Tuple[int, int, int], ...], num_grid: int, m_size: int, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Per-image scalar tensors for one launch, cached by geometry.

    Both entry points go through here, so the multi-grid path (the one vllm actually hits)
    does not rebuild its tensors on every forward. ``t`` participates through
    ``total_blocks`` and the row offsets, so the full grid list is part of the key; the
    tensors are a pure function of the geometry, which is what makes caching sound.

    Returns the bundle and the largest per-image block count, which the caller needs on the
    host to size the grid without a device sync.
    """
    key = (grids, num_grid, m_size, str(device))
    cached = _SCALAR_CACHE.get(key)
    if cached is not None:
        return cached

    merge = m_size * m_size
    blocks = [(t * h * w) // merge for t, h, w in grids]
    offsets, cursor = [], 0
    for n in blocks:
        offsets.append(cursor)
        cursor += n

    i32 = torch.int32
    bundle = {
        "h": torch.tensor([h for _t, h, _w in grids], device=device, dtype=i32),
        "w": torch.tensor([w for _t, _h, w in grids], device=device, dtype=i32),
        "h_scale": torch.tensor(
            [_axis_scale(num_grid, h) for _t, h, _w in grids], device=device, dtype=torch.float32
        ),
        "w_scale": torch.tensor(
            [_axis_scale(num_grid, w) for _t, _h, w in grids], device=device, dtype=torch.float32
        ),
        "total_blocks": torch.tensor(blocks, device=device, dtype=i32),
        "blocks_per_frame": torch.tensor(
            [(h // m_size) * (w // m_size) for _t, h, w in grids], device=device, dtype=i32
        ),
        "out_row_off": torch.tensor(offsets, device=device, dtype=i32),
    }
    result = (bundle, max(blocks))
    if len(_SCALAR_CACHE) >= _CACHE_LIMIT:
        _SCALAR_CACHE.clear()
    _SCALAR_CACHE[key] = result
    return result


def _launch(
    embed_weight: torch.Tensor,
    grids: List[Tuple[int, int, int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    hidden = embed_weight.shape[1]
    out_dtype = embed_weight.dtype if dtype is None else dtype
    device = embed_weight.device

    total_rows = sum(t * h * w for t, h, w in grids)
    out = torch.empty(total_rows, hidden, device=device, dtype=out_dtype)
    if total_rows == 0:
        return out

    bundle, max_blocks = _scalar_bundle(tuple(grids), num_grid_per_side, m_size, device)

    cores = max(get_num_cores("vector"), 1)
    per_image = max(1, cores // len(grids))
    block_d, width = _pick_tuning(per_image, max_blocks, hidden)
    grid = (min(max_blocks, width), len(grids))

    _pos_embed_interp_kernel[grid](
        embed_weight,
        out,
        bundle["h"],
        bundle["w"],
        bundle["h_scale"],
        bundle["w_scale"],
        bundle["total_blocks"],
        bundle["blocks_per_frame"],
        bundle["out_row_off"],
        num_grid=num_grid_per_side,
        m_size=m_size,
        hidden=hidden,
        block_d=block_d,
    )
    return out


def _normalize_grids(grids) -> List[Tuple[int, int, int]]:
    if isinstance(grids, torch.Tensor):
        flat = grids.to(dtype=torch.int64, device="cpu").reshape(-1, 3).tolist()
        return [(int(a), int(b), int(c)) for a, b, c in flat]
    if isinstance(grids, Sequence):
        out = []
        for item in grids:
            values = [int(v) for v in item]
            if len(values) != 3:
                raise ValueError(f"each grid must be (t, h, w), got {item!r}")
            out.append((values[0], values[1], values[2]))
        return out
    raise TypeError("grids must be a sequence of (t, h, w) or an int tensor of shape [N, 3]")


def pos_embed_interp_infer_fwd(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Bilinear interpolate a ``[G*G, hidden]`` table onto ``(t, h, w)``.

    Returns ``[t*h*w, hidden]`` in spatial-merge order (``m_size x m_size`` neighbours
    adjacent), matching ``pos_embed_interpolate_native``.
    """
    if embed_weight.ndim != 2:
        raise ValueError(f"embed_weight must be 2D, got shape {tuple(embed_weight.shape)}")
    _check_geometry(h, w, m_size, num_grid_per_side)
    if (t * h * w) % (m_size * m_size):
        raise ValueError(f"t*h*w={t * h * w} must be divisible by m_size^2={m_size * m_size}")
    return _launch(embed_weight, [(int(t), int(h), int(w))], int(num_grid_per_side), int(m_size), dtype)


def pos_embed_interp_multi_infer_fwd(
    embed_weight: torch.Tensor,
    grids,
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """One launch for a list of ``(t, h, w)`` grids.

    Concatenates the per-image outputs along dim 0 in list order, bit-exactly equal to
    calling :func:`pos_embed_interp_infer_fwd` per image and ``torch.cat``-ing them.
    """
    if embed_weight.ndim != 2:
        raise ValueError(f"embed_weight must be 2D, got shape {tuple(embed_weight.shape)}")
    normalized = _normalize_grids(grids)
    if not normalized:
        raise ValueError("grids must be non-empty")
    for _t, h, w in normalized:
        _check_geometry(h, w, m_size, num_grid_per_side)
    merge = m_size * m_size
    for t, h, w in normalized:
        if (t * h * w) % merge:
            raise ValueError(f"t*h*w={t * h * w} must be divisible by m_size^2={merge}")
    return _launch(embed_weight, normalized, int(num_grid_per_side), int(m_size), dtype)
