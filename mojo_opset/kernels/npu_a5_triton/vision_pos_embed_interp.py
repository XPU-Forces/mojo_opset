"""A5 Triton implementation of Qwen3-VL vision position-embedding interpolation.

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
        w00 = tl.load(weight_ptr + srows, mask=row_mask, other=0.0).to(out_dtype)
        w01 = tl.load(weight_ptr + TOTAL_SPATIAL + srows, mask=row_mask, other=0.0).to(out_dtype)
        w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0).to(out_dtype)
        w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0).to(out_dtype)

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
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    TOTAL_SPATIAL: tl.constexpr,
    HW: tl.constexpr,
    FRAMES: tl.constexpr,
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

        off00 = tl.load(index_ptr + srows, mask=row_mask, other=0)
        off01 = tl.load(index_ptr + TOTAL_SPATIAL + srows, mask=row_mask, other=0)
        off10 = tl.load(index_ptr + 2 * TOTAL_SPATIAL + srows, mask=row_mask, other=0)
        off11 = tl.load(index_ptr + 3 * TOTAL_SPATIAL + srows, mask=row_mask, other=0)
        w00 = tl.load(weight_ptr + srows, mask=row_mask, other=0.0).to(out_dtype)
        w01 = tl.load(weight_ptr + TOTAL_SPATIAL + srows, mask=row_mask, other=0.0).to(out_dtype)
        w10 = tl.load(weight_ptr + 2 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0).to(out_dtype)
        w11 = tl.load(weight_ptr + 3 * TOTAL_SPATIAL + srows, mask=row_mask, other=0.0).to(out_dtype)

        off0 = (out_base + rows).to(tl.int32) * HIDDEN_DIM
        cols = d + tl.arange(0, BLOCK_D)
        col_mask = cols < HIDDEN_DIM
        mask = row_mask[:, None] & col_mask[None, :]

        e00 = tl.load(embed_ptr + off00[:, None] + cols[None, :], mask=mask, other=0.0)
        e01 = tl.load(embed_ptr + off01[:, None] + cols[None, :], mask=mask, other=0.0)
        e10 = tl.load(embed_ptr + off10[:, None] + cols[None, :], mask=mask, other=0.0)
        e11 = tl.load(embed_ptr + off11[:, None] + cols[None, :], mask=mask, other=0.0)

        value = w00[:, None] * e00 + w01[:, None] * e01 + w10[:, None] * e10 + w11[:, None] * e11

        for frame in tl.static_range(0, FRAMES):
            active = row_mask & (frame < frames)
            off = off0 + frame * stride
            tl.store(
                out_ptr + off[:, None] + cols[None, :],
                value,
                mask=active[:, None] & col_mask[None, :],
            )

def _select_block_d(hidden_dim: int) -> int:
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
    image = torch.tensor(image_rows, dtype=torch.int32, device=embed_weight.device).reshape(-1)

    if len(_META_CACHE) >= _META_CACHE_LIMIT:
        _META_CACHE.pop(next(iter(_META_CACHE)))
    _META_CACHE[key] = (index, weight, image)
    return index, weight, image


def _launch(embed_weight: torch.Tensor, grids: Sequence[Sequence[int]], num_grid: int, m_size: int, dtype: torch.dtype):
    if not grids:
        return torch.empty((0, embed_weight.shape[1]), dtype=dtype, device=embed_weight.device)
    if not embed_weight.is_contiguous():
        embed_weight = embed_weight.contiguous()

    hidden_dim = embed_weight.shape[1]
    total_rows = sum(int(t) * int(h) * int(w) for t, h, w in grids)
    total_spatial = sum(int(h) * int(w) for _, h, w in grids)
    output = torch.empty((total_rows, hidden_dim), dtype=dtype, device=embed_weight.device)

    index, weight, image = _build_row_metadata(
        embed_weight,
        tuple(tuple(int(v) for v in grid) for grid in grids),
        num_grid,
        m_size,
        hidden_dim,
    )

    if len(grids) == 1 and int(grids[0][0]) == 1:
        hw = int(grids[0][1]) * int(grids[0][2])
        bd = triton.next_power_of_2(hidden_dim)
        _bilinear_pos_embed_static_kernel[(triton.cdiv(hw, 4), 1)](embed_weight,output,index,weight,image,hidden_dim,4,bd,1,total_spatial,hw,1)
        return output

    # FP32 tiles use twice the UB of BF16; 8 rows keeps the four neighbour
    # tiles plus output inside the 254 KB UB budget.
    block_m = 16 if embed_weight.dtype == torch.bfloat16 else 8
    block_d = _select_block_d(hidden_dim)
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
) -> torch.Tensor:
    dtype = embed_weight.dtype if dtype is None else dtype
    return _launch(embed_weight, grid_thw, int(num_grid_per_side), int(m_size), dtype)
