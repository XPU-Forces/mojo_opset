"""A2 (Ascend910B3) row-tiled path for ANY single-image frame count (V rows per store).

Architecture difference vs the delivered kernel (not a tile sweep).  The delivered
t=1 schedule walks one spatial row per iteration: four int32 offset loads and four
fp32 weight loads (8 scalar DMAs) feed four 2304 B gathers, then one 2304 B store.
Measured on A2 (one process, fill_ 1461.8 GB/s, 60% budget 18.66 us for (1,60,80)):

    store-only, 1 row/iteration            20.168 us  (108% of the budget)
    store-only, 4 rows/iteration            8.064 us
    delivered kernel                       37.058 us
    four rows per iteration, 32 scalar
      loads + four separate stores         38.355 us   <- grouping alone does not pay

so the cost is the per-row scalar decode chain, not the gather bytes.  This kernel
makes the *row* the second tile axis:

  * the decode is stored transposed, ``offT[c*N + row]`` / ``wtT[c*N + row]``, so the
    four corner offsets (and weights) of V consecutive rows are one contiguous V-wide
    vector load each -- 8 vector loads per V rows instead of 8*V scalar loads;
  * each corner becomes a single [V, D] gather with a per-row base address;
  * the V rows of the output are contiguous, so the whole group is written by ONE
    [V, D] store (V*2304 B) instead of V separate 2304 B stores.

Per-row arithmetic is unchanged: the four arithmetic corner terms are added in the same
order, with weights pre-cast to the output dtype, so the result is bit-identical to the
delivered kernel (verified in-process against it before timing).

``_V`` (rows per tile) and ``_EXACT`` (exact D vs padded+masked width) are module
constants so one file can sweep both without touching the kernel body.  The path is
gated to ``_ROWS_ARCH`` (A3 additionally only for t == 1, where it was measured better);
every other launch keeps the
delivered kernel untouched.
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

# Programs per vector core.  Per-row work is independent, so extra programs only hide DMA
# latency; measured insensitive between 4 and 16 waves (423 us vs 427 us on the large grid).
_WAVES = 4
# Platforms where a 2-D output tile may use an exact (non power-of-two) width.  On
# Ascend950PR + CANN 9.2.0-beta.2 such a tile lowers pathologically (hidden=1000 measured
# 30 ms), so that family keeps the padded width; on Ascend910_9382 + CANN 9.3.0 it is the
# fast path (hidden=1000 tail 55.1 -> 34.6 us, hidden=1152 video unchanged).
_EXACT_2D_ARCH = ("Ascend910_9382",)
_ARCH_CACHE = []
# Rows per contiguous store.  4 is the measured plateau: [4,1152] bf16 = 9216 B runs at
# 1558 GB/s, [8,*] / [16,*] add nothing and cost UB occupancy.
_STORE_ROWS = 4
_V = 4                      # rows per [V, D] tile in the A2 t == 1 kernel
_EXACT = True               # exact D width (no padding, no store mask)
# Platforms whose measured schedule is the row-tiled kernel.  A2 keeps it for every
# single-image launch; on A3 the shipped 4-frame tile is measured better for t > 1
# ((3,16,16) 6.90 -> 7.25 us, (3,60,80) 34.5 -> 40.0 us), so A3 takes it for t == 1 only.
_ROWS_ARCH = ("Ascend910B3", "Ascend910_9382")
_ROWS_TONE_ONLY = ("Ascend910_9382",)
_CACHE_LIMIT = 64
_DECODE_CACHE: Dict[Tuple, Tuple[Dict[str, torch.Tensor], int, int]] = {}


def _arch() -> str:
    """Backend arch string, queried once (``Ascend910_9382`` / ``Ascend950PR_9579``)."""
    if not _ARCH_CACHE:
        try:
            _ARCH_CACHE.append(triton.runtime.driver.active.get_current_target().arch)
        except Exception:  # noqa: BLE001 - a missing target must not break the op
            _ARCH_CACHE.append("")
    return _ARCH_CACHE[0]


def _axis_scale(num_grid: int, size: int) -> float:
    # Same expression as upstream :281-282; size == 1 degenerates to scale 0.
    return float(num_grid - 1) / float(size - 1) if size > 1 else 0.0


def _decode(
    grids: Tuple[Tuple[int, int, int], ...],
    num_grid: int,
    m_size: int,
    hidden: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Corner offsets and bilinear weights for every spatial row, in output order.

    ``torch.linspace`` is used for the sampling positions (not ``arange * scale``) so the
    ``floor``/``frac`` split is bit-identical to ``pos_embed_interpolate_native``.  The
    four weights are the reference's own ``w00/w01/w10/w11`` expressions evaluated in FP32
    and then cast to the output dtype, i.e. exactly upstream's ``weights.to(out_dtype)``;
    the multiply-accumulate keeps having the output dtype as its operand type.
    """
    off_parts: List[torch.Tensor] = []
    wt_parts: List[torch.Tensor] = []
    frames: List[int] = []
    spatial: List[int] = []
    bases: List[int] = []
    dspatial: List[int] = []
    cursor = 0
    decode_cursor = 0

    for t, h, w in grids:
        h_idx = torch.linspace(0, num_grid - 1, h, dtype=torch.float32)
        w_idx = torch.linspace(0, num_grid - 1, w, dtype=torch.float32)
        h_floor = h_idx.to(torch.long)
        w_floor = w_idx.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)
        dh = h_idx - h_floor.to(torch.float32)
        dw = w_idx - w_floor.to(torch.float32)
        prod = dh[:, None] * dw[None, :]
        w11 = prod
        w10 = dh[:, None] - prod
        w01 = dw[None, :] - prod
        w00 = 1 - dh[:, None] - w01

        # spatial-merge order: block (br, bc) owns the m x m rows (br*m+lr, bc*m+lc)
        br = torch.arange(h // m_size)
        bc = torch.arange(w // m_size)
        lr = torch.arange(m_size)
        lc = torch.arange(m_size)
        row = (br[:, None, None, None] * m_size + lr[None, None, :, None]).expand(
            h // m_size, w // m_size, m_size, m_size).reshape(-1)
        col = (bc[None, :, None, None] * m_size + lc[None, None, None, :]).expand(
            h // m_size, w // m_size, m_size, m_size).reshape(-1)

        idx = torch.stack([h_floor[row] * num_grid + w_floor[col],
                           h_floor[row] * num_grid + w_ceil[col],
                           h_ceil[row] * num_grid + w_floor[col],
                           h_ceil[row] * num_grid + w_ceil[col]], dim=1)
        wei = torch.stack([w00[row, col], w01[row, col],
                           w10[row, col], w11[row, col]], dim=1)
        off_parts.append((idx * hidden).to(torch.int32))
        wt_parts.append(wei.to(torch.float32))
        frames.append(int(t))
        spatial.append(int(h * w))
        bases.append(cursor)
        dspatial.append(decode_cursor)
        cursor += int(t) * int(h) * int(w)
        decode_cursor += int(h) * int(w)

    i32 = torch.int32
    bundle = {
        "t": torch.tensor(frames, device=device, dtype=i32),
        "spatial": torch.tensor(spatial, device=device, dtype=i32),
        "base": torch.tensor(bases, device=device, dtype=i32),
        "dsp": torch.tensor(dspatial, device=device, dtype=i32),
        "off": torch.cat(off_parts, 0).contiguous().to(device),
        "wt": torch.cat(wt_parts, 0).contiguous().to(device),
    }
    return bundle, max(spatial)


def _check_geometry(h: int, w: int, m_size: int, num_grid: int) -> None:
    # m_size and num_grid are checked first: the divisibility test below divides by m_size,
    # so an m_size of 0 must raise the ValueError instead of a ZeroDivisionError.
    if m_size <= 0:
        raise ValueError(f"m_size={m_size} must be positive")
    if num_grid < 2:
        raise ValueError(f"num_grid_per_side={num_grid} must be >= 2")
    if h <= 0 or w <= 0:
        raise ValueError(f"h={h} and w={w} must be positive")
    if h % m_size or w % m_size:
        raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")


@libentry()
@triton.jit
def _pos_embed_interp_kernel(
    embed_ptr,
    out_ptr,
    off_ptr,
    wt_ptr,
    t_ptr,
    sp_ptr,
    base_ptr,
    dsp_ptr,
    RS: tl.constexpr,
    D: tl.constexpr,
    BLK: tl.constexpr,
    MASKF: tl.constexpr,
    TONE: tl.constexpr,
):
    image = tl.program_id(1)
    pid = tl.program_id(0)
    npg = tl.num_programs(0)

    t = tl.load(t_ptr + image)
    spatial = tl.load(sp_ptr + image)
    out_base = tl.load(base_ptr + image)
    dec_base = tl.load(dsp_ptr + image)

    chunk = (spatial + npg - 1) // npg
    start = pid * chunk
    stop = tl.minimum(start + chunk, spatial)
    cols = tl.arange(0, BLK)
    tail = BLK != D
    step = spatial * D
    dt = out_ptr.dtype.element_ty
    dec4 = dec_base * 4
    out_d = out_base * D

    for r in range(start, stop):
        b = dec4 + r * 4
        o0 = tl.load(off_ptr + b + 0)
        o1 = tl.load(off_ptr + b + 1)
        o2 = tl.load(off_ptr + b + 2)
        o3 = tl.load(off_ptr + b + 3)
        w0 = tl.load(wt_ptr + b + 0).to(dt)
        w1 = tl.load(wt_ptr + b + 1).to(dt)
        w2 = tl.load(wt_ptr + b + 2).to(dt)
        w3 = tl.load(wt_ptr + b + 3).to(dt)

        if tail:
            dmask = cols < D
            e0 = tl.load(embed_ptr + o0 + cols, mask=dmask)
            e1 = tl.load(embed_ptr + o1 + cols, mask=dmask)
            e2 = tl.load(embed_ptr + o2 + cols, mask=dmask)
            e3 = tl.load(embed_ptr + o3 + cols, mask=dmask)
        else:
            e0 = tl.load(embed_ptr + o0 + cols)
            e1 = tl.load(embed_ptr + o1 + cols)
            e2 = tl.load(embed_ptr + o2 + cols)
            e3 = tl.load(embed_ptr + o3 + cols)

        # w00*e00 + w01*e01 + w10*e10 + w11*e11 in the output dtype, exactly upstream
        # :236-240; the operand type is the load type, the weights are pre-cast.
        val = w0 * e0 + w1 * e1 + w2 * e2 + w3 * e3

        row_off = out_d + r * D
        if RS == 1:
            # No frame *batching* here, but the temporal broadcast still applies: a mixed
            # grid list can carry t > 1 for one image while another has t == 1.  TONE is a
            # constexpr "every image has exactly one frame" so the common case keeps a
            # straight-line store (a dynamic one-iteration loop costs 13% on 1x60x80).
            if TONE:
                if tail:
                    tl.store(out_ptr + row_off + cols, val, mask=cols < D)
                else:
                    tl.store(out_ptr + row_off + cols, val)
            else:
                for f in range(t):
                    if tail:
                        tl.store(out_ptr + row_off + f * step + cols, val, mask=cols < D)
                    else:
                        tl.store(out_ptr + row_off + f * step + cols, val)
        else:
            # One row interpolated once, stored to RS frames with one contiguous block.
            rr = tl.arange(0, RS)
            tile = val[None, :] + tl.zeros((RS, 1), out_ptr.dtype.element_ty)
            chunks = (t + RS - 1) // RS
            for c in range(chunks):
                off = row_off + (c * RS + rr)[:, None] * step + cols[None, :]
                if MASKF:
                    fmask = (c * RS + rr)[:, None] < t
                    if tail:
                        tl.store(out_ptr + off, tile, mask=fmask & (cols[None, :] < D))
                    else:
                        tl.store(out_ptr + off, tile, mask=fmask)
                else:
                    if tail:
                        tl.store(out_ptr + off, tile, mask=cols[None, :] < D)
                    else:
                        tl.store(out_ptr + off, tile)


@libentry()
@triton.jit
def _t1_rows_kernel(
    embed_ptr,
    out_ptr,
    offT_ptr,
    wtT_ptr,
    t_ptr,
    sp_ptr,
    base_ptr,
    dsp_ptr,
    NR: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    BLK: tl.constexpr,
):
    """V rows per iteration: 8 vector decodes, four [V, D] gathers, one [V, D] store."""
    pid = tl.program_id(0)
    npg = tl.num_programs(0)

    spatial = tl.load(sp_ptr)
    out_base = tl.load(base_ptr)
    dec_base = tl.load(dsp_ptr)
    frames = tl.load(t_ptr)
    step = spatial * D

    cols = tl.arange(0, BLK)
    vv = tl.arange(0, V)
    tail = BLK != D
    dt = out_ptr.dtype.element_ty
    out_d = out_base * D

    units = spatial // V
    chunk = (units + npg - 1) // npg
    start = pid * chunk
    stop = tl.minimum(start + chunk, units)

    for u in range(start, stop):
        r0 = u * V
        b = dec_base + r0
        o0 = tl.load(offT_ptr + b + vv)
        o1 = tl.load(offT_ptr + NR + b + vv)
        o2 = tl.load(offT_ptr + 2 * NR + b + vv)
        o3 = tl.load(offT_ptr + 3 * NR + b + vv)
        w0 = tl.load(wtT_ptr + b + vv).to(tl.float32)
        w1 = tl.load(wtT_ptr + NR + b + vv).to(tl.float32)
        w2 = tl.load(wtT_ptr + 2 * NR + b + vv).to(tl.float32)
        w3 = tl.load(wtT_ptr + 3 * NR + b + vv).to(tl.float32)

        if tail:
            dmask = cols[None, :] < D
            e0 = tl.load(embed_ptr + o0[:, None] + cols[None, :], mask=dmask, other=0.0)
            e1 = tl.load(embed_ptr + o1[:, None] + cols[None, :], mask=dmask, other=0.0)
            e2 = tl.load(embed_ptr + o2[:, None] + cols[None, :], mask=dmask, other=0.0)
            e3 = tl.load(embed_ptr + o3[:, None] + cols[None, :], mask=dmask, other=0.0)
        else:
            e0 = tl.load(embed_ptr + o0[:, None] + cols[None, :])
            e1 = tl.load(embed_ptr + o1[:, None] + cols[None, :])
            e2 = tl.load(embed_ptr + o2[:, None] + cols[None, :])
            e3 = tl.load(embed_ptr + o3[:, None] + cols[None, :])

        # same four terms, same order, weights pre-cast: bit-identical to the delivered kernel
        acc = (w0[:, None] * e0.to(tl.float32) + w1[:, None] * e1.to(tl.float32)
               + w2[:, None] * e2.to(tl.float32) + w3[:, None] * e3.to(tl.float32)).to(dt)

        off = out_d + (r0 + vv)[:, None] * D + cols[None, :]
        for f in range(frames):
            if tail:
                tl.store(out_ptr + f * step + off, acc, mask=cols[None, :] < D)
            else:
                tl.store(out_ptr + f * step + off, acc)


def _launch(
    embed_weight: torch.Tensor,
    grids: List[Tuple[int, int, int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    hidden = int(embed_weight.shape[1])
    out_dtype = embed_weight.dtype if dtype is None else dtype
    device = embed_weight.device
    total_rows = sum(t * h * w for t, h, w in grids)
    out = torch.empty(total_rows, hidden, device=device, dtype=out_dtype)
    if total_rows == 0:
        return out

    cores = max(int(get_num_cores("vector")), 1)
    key = (tuple(grids), int(num_grid_per_side), int(m_size), hidden, out_dtype, str(device))
    cached = _DECODE_CACHE.get(key)
    if cached is None:
        bundle, max_spatial = _decode(tuple(grids), int(num_grid_per_side), int(m_size),
                                      hidden, out_dtype, device)
        cached = (bundle, max_spatial)
        if len(_DECODE_CACHE) >= _CACHE_LIMIT:
            _DECODE_CACHE.clear()
        _DECODE_CACHE[key] = cached
    bundle, max_spatial = cached

    frames = bundle["t"].tolist()
    # Frame batching pays off only when every image has more than one frame; a mixed list
    # (the task-book multi case is t = 1, 1, 3) would otherwise pay a masked 3-of-4 store
    # for the two single-frame images.
    rs = _STORE_ROWS if all(f > 1 for f in frames) else 1
    # The frame mask is only needed when some image's frame count is not a multiple of RS.
    maskf = any(f % rs for f in frames)
    tone = all(f == 1 for f in frames)
    # The one-row-per-store schedule is DMA-descriptor bound rather than bandwidth bound,
    # and a single wave measured 2.5% faster there; the batched-frame schedule keeps 4.
    waves = 1 if rs == 1 else _WAVES
    width = max(min(max_spatial, cores * waves), 1)
    # A single-row store tolerates an exact (non power-of-two) width - the shipped PR#513
    # kernel already relies on that - and exact width removes the mask and the padded
    # lanes from all four gathers.  A multi-row 2-D tile does not: a non power-of-two
    # second axis lowers pathologically (measured 30 ms for hidden=1000), so the batched
    # path keeps the padded width.
    exact2d = any(tag in _arch() for tag in _EXACT_2D_ARCH)
    blk = hidden if (rs == 1 or exact2d) else triton.next_power_of_2(hidden)

    # ---- row-tiled fast path (see module docstring) -----------------------------------
    arch = _arch()
    tone_only = any(tag in arch for tag in _ROWS_TONE_ONLY)
    if (len(grids) == 1 and any(tag in arch for tag in _ROWS_ARCH)
            and max_spatial % _V == 0 and not (tone_only and frames[0] != 1)):
        nr = int(max_spatial)
        if "offT" not in bundle:
            bundle["offT"] = bundle["off"].reshape(nr, 4).t().contiguous()
            bundle["wtT"] = bundle["wt"].reshape(nr, 4).t().contiguous()
        # exact width only where the backend lowers it well: hidden=1152 is fine,
        # hidden=1000 lowers pathologically (measured 3.0 ms for a [4,1000] tile on A2)
        blk = hidden if (_EXACT and hidden % 128 == 0) else triton.next_power_of_2(hidden)
        units = nr // _V
        width = max(min(units, cores), 1)
        _t1_rows_kernel[(width,)](
            embed_weight, out, bundle["offT"], bundle["wtT"], bundle["t"], bundle["spatial"],
            bundle["base"], bundle["dsp"], NR=nr, V=_V, D=hidden, BLK=blk)
        return out

    _pos_embed_interp_kernel[(width, len(grids))](
        embed_weight,
        out,
        bundle["off"],
        bundle["wt"],
        bundle["t"],
        bundle["spatial"],
        bundle["base"],
        bundle["dsp"],
        RS=rs,
        D=hidden,
        BLK=blk,
        MASKF=maskf,
        TONE=tone,
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
    return _launch(embed_weight, [(int(t), int(h), int(w))], int(num_grid_per_side),
                   int(m_size), dtype)


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


# The acceptance suite marks cases with the public op id, which needs its own binding;
# the signature is identical to the inference entry, so the kernel entry point is shared.
vision_pos_embed_interp_fwd = pos_embed_interp_infer_fwd
