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
# Per-image scalar bundles, keyed by geometry. Reused across calls so the single-grid entry
# point pays no host-side tensor construction after the first call.
_SCALAR_CACHE: Dict[Tuple[int, int, int, int, int, str], Dict[str, torch.Tensor]] = {}
_CACHE_LIMIT = 64


def _pick_tuning(cores: int, max_blocks: int, hidden: int) -> Tuple[int, int, str, int]:
    """Return ``(block_d, grid_width)``.

    ``block_d`` covers the whole row in one tile when it fits (fewer loop turns), and the
    persistent-grid width decides how many independent block streams are in flight.
    Ascend hides load latency with concurrent programs, not with deep software pipelining,
    so the grid is deliberately allowed to exceed the core count; the per-block work is
    independent, which keeps the extra programs cheap.
    """
    # Measured on 910B3 (A2), grid (16,100,168), hidden=1152:
    #   block_d 2048 -> 3400us, 1024 -> 3501us, 512 -> 3047us, 256 -> 3722us
    # A 512-wide tile keeps the (4, 512) corner tiles plus accumulator inside UB with room
    # for double buffering; 2048 spills and 256 needs too many loop turns. The 2D block tile
    # beats the row-wise kernel at every width (3816us at its best), so it is the default.
    # Row kernel on 3.6.0: one wide slice beats several narrow ones, because the extra
    # instructions and the per-slice masks cost more than the padded lanes. Measured on grid
    # (16,100,168): block_d 256 -> 5480us, 512 -> 4558us, 1024 -> 4364us, 2048 -> 3786us.
    block_d = min(_MAX_BLOCK_D, triton.next_power_of_2(hidden))
    grid_width = min(max_blocks, max(cores, 1) * 4)
    blocks_per_iter = 2 if max_blocks >= 2 else 1
    # Mode "row" is the default on triton-ascend 3.6.0: the 2D tile kernel's masked gather with a
    # per-row base offset lowers to something pathological on this compiler (measured 777ms vs
    # 4.6ms for the row-wise kernel on grid (16,100,168), i.e. 170x). On 3.2.x the 2D tile was the
    # faster one (3047us vs 3816us), so the mode stays selectable through _pick_tuning.
    return block_d, grid_width, "row", blocks_per_iter


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
    chunked: tl.constexpr,
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

    # Two ways to hand blocks to programs. Grid-stride spreads a program's blocks across the
    # whole output, so every iteration touches a different part of the position table and the
    # 4-neighbour reads miss cache. A contiguous chunk makes one program walk along an output
    # row, where the source rows advance by ~2*w_scale per step, so the neighbour reads stay
    # resident and HBM traffic drops towards the write-only floor.
    if chunked:
        chunk = (total_blocks + grid_size - 1) // grid_size
        start = pid * chunk
        stop = tl.minimum(start + chunk, total_blocks)
        step = 1
    else:
        start = pid
        stop = total_blocks
        step = grid_size

    for blk in range(start, stop, step):
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


@triton.jit
def _corner_tile(
    embed_ptr,
    out_ptr,
    off00,
    off01,
    off10,
    off11,
    dst,
    w00_c,
    w01_c,
    w10_c,
    w11_c,
    d0,
    row_mask,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    """One d-slice of the bilinear MAC for a ``(tile_rows, block)`` tile.

    Split out so the caller can run full-width slices plus one exact-width tail slice, instead
    of padding the tail to the full block and burning masked lanes (hidden=1152 with
    block_d=512 would otherwise issue 512 lanes for the last 128 columns).
    """
    cols = d0 + tl.arange(0, block)
    mask = row_mask & (cols < hidden)[None, :]
    e00 = tl.load(embed_ptr + off00 + cols[None, :], mask=mask)
    e01 = tl.load(embed_ptr + off01 + cols[None, :], mask=mask)
    e10 = tl.load(embed_ptr + off10 + cols[None, :], mask=mask)
    e11 = tl.load(embed_ptr + off11 + cols[None, :], mask=mask)

    val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

    tl.store(out_ptr + dst + cols[None, :], val, mask=mask)


@libentry()
@triton.jit
def _pos_embed_interp_block_kernel(
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
    blocks_per_iter: tl.constexpr,
    n_full_d: tl.constexpr,
    tail_d: tl.constexpr,
):
    """``blocks_per_iter`` merge blocks per iteration as one ``(K*m_size^2, block_d)`` tile.

    The merge decode becomes vector arithmetic over ``arange(K*m_size^2)`` and one iteration
    issues ``K*m_size^2`` contiguous output rows as a single tile. Two reasons this matters on
    Ascend (both measured, see docs/T2_*§5):

    * the 1-D row-wise form issues 2304-byte stores that only reach ~600 GB/s, i.e. 43% of the
      fill_ calibration; widening the contiguous span per store is the only lever left once the
      neighbour reads are cache-resident;
    * the block decode cost is amortised over K blocks instead of one.

    ``blocks_per_iter=1`` reproduces the previous single-block behaviour.
    """
    num_rows: tl.constexpr = m_size * m_size
    tile_rows: tl.constexpr = blocks_per_iter * num_rows
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
    # q indexes every output row of this iteration; b picks which of the K blocks it belongs to
    # and j is the row inside that block, so (lr, lc) come straight out of the flatten order.
    q = tl.arange(0, tile_rows)
    b = q // num_rows
    j = q % num_rows
    lr = j // m_size
    lc = j % m_size
    out_dtype = out_ptr.dtype.element_ty

    total_super = (total_blocks + blocks_per_iter - 1) // blocks_per_iter
    for sb in range(pid, total_super, grid_size):
        blk0 = sb * blocks_per_iter
        blk = blk0 + b
        # Tail guard: the last super-block can run past total_blocks.
        valid = blk < total_blocks
        spatial_blk = blk % blocks_per_frame
        br = spatial_blk // num_blocks_w
        bc = spatial_blk % num_blocks_w
        dst = ((out_row_off + blk0).to(tl.int64) * num_rows + q) * hidden

        row = br * m_size + lr
        col = bc * m_size + lc

        h_frac = row.to(tl.float32) * h_scale
        hf = tl.floor(h_frac).to(tl.int32)
        hc = tl.minimum(hf + 1, num_grid - 1)
        dh = h_frac - hf.to(tl.float32)

        w_frac = col.to(tl.float32) * w_scale
        wf = tl.floor(w_frac).to(tl.int32)
        wc = tl.minimum(wf + 1, num_grid - 1)
        dw = w_frac - wf.to(tl.float32)

        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01
        w00_c = w00.to(out_dtype)[:, None]
        w01_c = w01.to(out_dtype)[:, None]
        w10_c = w10.to(out_dtype)[:, None]
        w11_c = w11.to(out_dtype)[:, None]

        off00 = ((hf * num_grid + wf) * hidden)[:, None]
        off01 = ((hf * num_grid + wc) * hidden)[:, None]
        off10 = ((hc * num_grid + wf) * hidden)[:, None]
        off11 = ((hc * num_grid + wc) * hidden)[:, None]
        dst = dst[:, None]
        row_mask = valid[:, None]

        for i in tl.static_range(n_full_d):
            _corner_tile(
                embed_ptr,
                out_ptr,
                off00,
                off01,
                off10,
                off11,
                dst,
                w00_c,
                w01_c,
                w10_c,
                w11_c,
                i * block_d,
                row_mask,
                hidden=hidden,
                block=block_d,
            )
        if tail_d > 0:
            _corner_tile(
                embed_ptr,
                out_ptr,
                off00,
                off01,
                off10,
                off11,
                dst,
                w00_c,
                w01_c,
                w10_c,
                w11_c,
                n_full_d * block_d,
                row_mask,
                hidden=hidden,
                block=tail_d,
            )


def _check_geometry(h: int, w: int, m_size: int, num_grid: int) -> None:
    if h <= 0 or w <= 0:
        raise ValueError(f"h={h} and w={w} must be positive")
    if h % m_size or w % m_size:
        raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")
    if num_grid < 2:
        raise ValueError(f"num_grid_per_side={num_grid} must be >= 2")
    if m_size <= 0:
        raise ValueError(f"m_size={m_size} must be positive")


def _scalar_bundle(
    t: int, h: int, w: int, num_grid: int, m_size: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    # t participates through total_blocks, so it must be part of the key.
    key = (t, h, w, num_grid, m_size, str(device))
    cached = _SCALAR_CACHE.get(key)
    if cached is not None:
        return cached
    merge = m_size * m_size
    bundle = {
        "h": torch.tensor([h], device=device, dtype=torch.int32),
        "w": torch.tensor([w], device=device, dtype=torch.int32),
        "h_scale": torch.tensor([_axis_scale(num_grid, h)], device=device, dtype=torch.float32),
        "w_scale": torch.tensor([_axis_scale(num_grid, w)], device=device, dtype=torch.float32),
        "total_blocks": torch.tensor([(t * h * w) // merge], device=device, dtype=torch.int32),
        "blocks_per_frame": torch.tensor([(h // m_size) * (w // m_size)], device=device, dtype=torch.int32),
        "out_row_off": torch.tensor([0], device=device, dtype=torch.int32),
    }
    if len(_SCALAR_CACHE) >= _CACHE_LIMIT:
        _SCALAR_CACHE.clear()
    _SCALAR_CACHE[key] = bundle
    return bundle


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
    merge = m_size * m_size

    rows = [t * h * w for t, h, w in grids]
    total_rows = sum(rows)
    out = torch.empty(total_rows, hidden, device=device, dtype=out_dtype)
    if total_rows == 0:
        return out

    if len(grids) == 1:
        t, h, w = grids[0]
        bundle = _scalar_bundle(t, h, w, num_grid_per_side, m_size, device)
        h_t = bundle["h"]
        w_t = bundle["w"]
        hs_t = bundle["h_scale"]
        ws_t = bundle["w_scale"]
        tb_t = bundle["total_blocks"]
        bpf_t = bundle["blocks_per_frame"]
        off_t = bundle["out_row_off"]
        # Derived on the host: calling .item() here would force a device sync per call.
        max_blocks = (t * h * w) // merge
    else:
        blocks = [n // merge for n in rows]
        offsets, cursor_blocks = [], 0
        for n in blocks:
            offsets.append(cursor_blocks)
            cursor_blocks += n
        i32 = torch.int32
        h_t = torch.tensor([h for _t, h, _w in grids], device=device, dtype=i32)
        w_t = torch.tensor([w for _t, _h, w in grids], device=device, dtype=i32)
        hs_t = torch.tensor(
            [_axis_scale(num_grid_per_side, h) for _t, h, _w in grids], device=device, dtype=torch.float32
        )
        ws_t = torch.tensor(
            [_axis_scale(num_grid_per_side, w) for _t, _h, w in grids], device=device, dtype=torch.float32
        )
        tb_t = torch.tensor(blocks, device=device, dtype=i32)
        bpf_t = torch.tensor(
            [(h // m_size) * (w // m_size) for _t, h, w in grids], device=device, dtype=i32
        )
        off_t = torch.tensor(offsets, device=device, dtype=i32)
        max_blocks = max(blocks)

    cores = max(get_num_cores("vector"), 1)
    per_image = max(1, cores // max(len(grids), 1))
    block_d, width, mode, blocks_per_iter = _pick_tuning(per_image, max_blocks, hidden)
    grid = (min(max_blocks, width), len(grids))

    n_full_d = hidden // block_d
    tail = hidden - n_full_d * block_d
    tail_d = triton.next_power_of_2(tail) if tail else 0
    if mode == "block":
        _pos_embed_interp_block_kernel[grid](
            embed_weight,
            out,
            h_t,
            w_t,
            hs_t,
            ws_t,
            tb_t,
            bpf_t,
            off_t,
            num_grid=num_grid_per_side,
            m_size=m_size,
            hidden=hidden,
            block_d=block_d,
            blocks_per_iter=blocks_per_iter,
            n_full_d=n_full_d,
            tail_d=tail_d,
        )
    else:
        # "row" walks a contiguous chunk of blocks per program, "row_stride" grid-strides.
        _pos_embed_interp_kernel[grid](
            embed_weight,
            out,
            h_t,
            w_t,
            hs_t,
            ws_t,
            tb_t,
            bpf_t,
            off_t,
            num_grid=num_grid_per_side,
            m_size=m_size,
            hidden=hidden,
            block_d=block_d,
            chunked=(mode != "row_stride"),
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
