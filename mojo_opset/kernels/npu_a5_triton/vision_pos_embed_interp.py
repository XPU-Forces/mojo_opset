"""a5 Triton provider for learned position-table interpolation."""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_vision_pos_embed_interp import _bilinear_value
from mojo_opset.kernels._npu_triton_utils import get_num_cores


@triton.jit
def _interpolate_flat_block(
    embed_ptr,
    output_ptr,
    block,
    out_start: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HS: tl.constexpr,
    WS: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
):
    elements = block * B + tl.arange(0, B)
    pixel = elements // D
    feature = elements % D
    merge = pixel // (M * M)
    local = pixel % (M * M)
    row = merge // (W // M) * M + local // M
    col = merge % (W // M) * M + local % M
    h_frac = row.to(tl.float32) * HS
    w_frac = col.to(tl.float32) * WS
    # Coordinates are nonnegative, so conversion implements floor directly.
    hf = h_frac.to(tl.int32)
    wf = w_frac.to(tl.int32)
    hc = tl.minimum(hf + 1, G - 1)
    wc = tl.minimum(wf + 1, G - 1)
    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01
    mask = elements < H * W * D
    e00 = tl.load(embed_ptr + (hf * G + wf) * S0 + feature * S1, mask, other=0)
    e01 = tl.load(embed_ptr + (hf * G + wc) * S0 + feature * S1, mask, other=0)
    e10 = tl.load(embed_ptr + (hc * G + wf) * S0 + feature * S1, mask, other=0)
    e11 = tl.load(embed_ptr + (hc * G + wc) * S0 + feature * S1, mask, other=0)
    dtype = output_ptr.dtype.element_ty
    value = _bilinear_value(e00, e01, e10, e11, w00, w01, w10, w11, dtype)
    if T <= 16:
        for frame in tl.static_range(T):
            tl.store(output_ptr + (out_start + frame * H * W) * D + elements, value, mask)
    else:
        for frame in range(T):
            tl.store(output_ptr + (out_start + frame * H * W) * D + elements, value, mask)


@triton.jit
def _dispatch_images(
    embed_ptr,
    output_ptr,
    pid,
    GRIDS: tl.constexpr,
    ROW_STARTS: tl.constexpr,
    TILE_STARTS: tl.constexpr,
    SCALES_H: tl.constexpr,
    SCALES_W: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
):
    if HI - LO == 1:
        _interpolate_flat_block(
            embed_ptr,
            output_ptr,
            pid - TILE_STARTS[LO],
            ROW_STARTS[LO],
            GRIDS[LO][0],
            GRIDS[LO][1],
            GRIDS[LO][2],
            SCALES_H[LO],
            SCALES_W[LO],
            G,
            M,
            D,
            S0,
            S1,
            B,
        )
    else:
        MID: tl.constexpr = (LO + HI) // 2
        if pid < TILE_STARTS[MID]:
            _dispatch_images(
                embed_ptr,
                output_ptr,
                pid,
                GRIDS,
                ROW_STARTS,
                TILE_STARTS,
                SCALES_H,
                SCALES_W,
                G,
                M,
                D,
                S0,
                S1,
                B,
                LO,
                MID,
            )
        else:
            _dispatch_images(
                embed_ptr,
                output_ptr,
                pid,
                GRIDS,
                ROW_STARTS,
                TILE_STARTS,
                SCALES_H,
                SCALES_W,
                G,
                M,
                D,
                S0,
                S1,
                B,
                MID,
                HI,
            )


@triton.jit
def _bilinear_pos_embed_kernel(
    embed_ptr,
    output_ptr,
    GRIDS: tl.constexpr,
    ROW_STARTS: tl.constexpr,
    TILE_STARTS: tl.constexpr,
    SCALES_H: tl.constexpr,
    SCALES_W: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
):
    _dispatch_images(
        embed_ptr,
        output_ptr,
        tl.program_id(0),
        GRIDS,
        ROW_STARTS,
        TILE_STARTS,
        SCALES_H,
        SCALES_W,
        G,
        M,
        D,
        S0,
        S1,
        B,
        0,
        len(GRIDS),
    )


@triton.jit
def _interpolate_spatial_tile(
    embed_ptr, output_ptr, block, out_start: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HS: tl.constexpr, WS: tl.constexpr, G: tl.constexpr, M: tl.constexpr,
    D: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
    R: tl.constexpr, C: tl.constexpr, GROUP: tl.constexpr,
    PIXEL_START: tl.constexpr,
    LOAD_CACHE: tl.constexpr, STORE_CACHE: tl.constexpr,
):
    groups: tl.constexpr = triton.cdiv(triton.cdiv(D, C), GROUP)
    block = block.to(tl.uint32)
    pixel = PIXEL_START + block // groups * R + tl.arange(0, R).to(tl.uint32)
    feature_group = block % groups
    merge = pixel // (M * M)
    local = pixel % (M * M)
    row = merge // (W // M) * M + local // M
    col = merge % (W // M) * M + local % M
    h_frac = row.to(tl.float32) * HS
    w_frac = col.to(tl.float32) * WS
    hf, wf = h_frac.to(tl.uint32), w_frac.to(tl.uint32)
    hc, wc = tl.minimum(hf + 1, G - 1), tl.minimum(wf + 1, G - 1)
    dh, dw = h_frac - hf.to(tl.float32), w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10, w01 = dh - w11, dw - w11
    w00 = 1.0 - dh - w01
    dtype: tl.constexpr = output_ptr.dtype.element_ty
    # Reuse coordinate arithmetic and rounded coefficients over features.
    w00 = w00.to(dtype).to(tl.float32)[:, None]
    w01 = w01.to(dtype).to(tl.float32)[:, None]
    w10 = w10.to(dtype).to(tl.float32)[:, None]
    w11 = w11.to(dtype).to(tl.float32)[:, None]
    p00 = embed_ptr + (hf * G + wf)[:, None] * S0
    p01 = embed_ptr + (hf * G + wc)[:, None] * S0
    p10 = embed_ptr + (hc * G + wf)[:, None] * S0
    p11 = embed_ptr + (hc * G + wc)[:, None] * S0
    for part in tl.range(GROUP, loop_unroll_factor=GROUP):
        feature = (feature_group * GROUP + part) * C + tl.arange(0, C).to(tl.uint32)
        mask = tl.full((R, C), True, tl.int1)
        if H * W % R:
            mask = mask & (pixel[:, None] < H * W)
        if D % (GROUP * C):
            mask = mask & (feature[None, :] < D)
        columns = feature[None, :] * S1
        e00 = tl.load(p00 + columns, mask, 0, cache_modifier=LOAD_CACHE)
        e01 = tl.load(p01 + columns, mask, 0, cache_modifier=LOAD_CACHE)
        e10 = tl.load(p10 + columns, mask, 0, cache_modifier=LOAD_CACHE)
        e11 = tl.load(p11 + columns, mask, 0, cache_modifier=LOAD_CACHE)
        value = _bilinear_value(e00, e01, e10, e11, w00, w01, w10, w11, dtype)
        for frame in tl.static_range(T):
            tl.store(output_ptr + (out_start + frame * H * W + pixel[:, None]) * D + feature[None, :], value, mask, cache_modifier=STORE_CACHE)


@triton.jit
def _balanced_spatial_tiles_kernel(
    embed_ptr, output_ptr, H: tl.constexpr, W: tl.constexpr,
    HS: tl.constexpr, WS: tl.constexpr, G: tl.constexpr, M: tl.constexpr,
    D: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
    CORES: tl.constexpr, FULL_TILES: tl.constexpr, TAIL_ROWS: tl.constexpr,
):
    for block in range(tl.program_id(0), FULL_TILES, CORES):
        _interpolate_spatial_tile(
            embed_ptr, output_ptr, block, 0, 1, H, W, HS, WS,
            G, M, D, S0, S1, 16, 128, D // 128, 0, ".ca", ".cg",
        )
    if tl.program_id(0) < triton.cdiv(H * W - FULL_TILES * 16, TAIL_ROWS):
        _interpolate_spatial_tile(
            embed_ptr, output_ptr, tl.program_id(0), 0, 1, H, W, HS, WS,
            G, M, D, S0, S1, TAIL_ROWS, 128, D // 128,
            FULL_TILES * 16, ".ca", ".cg",
        )


@triton.jit
def _dispatch_spatial_tiles(
    embed_ptr, output_ptr, pid, GRIDS: tl.constexpr, ROW_STARTS: tl.constexpr,
    TILE_STARTS: tl.constexpr, SCALES_H: tl.constexpr, SCALES_W: tl.constexpr,
    TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr, TILE_GROUPS: tl.constexpr,
    G: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr,
):
    if HI - LO == 1:
        _interpolate_spatial_tile(
            embed_ptr, output_ptr, pid - TILE_STARTS[LO], ROW_STARTS[LO],
            GRIDS[LO][0], GRIDS[LO][1], GRIDS[LO][2], SCALES_H[LO], SCALES_W[LO],
            G, M, D, S0, S1, TILE_ROWS[LO], TILE_COLS[LO], TILE_GROUPS[LO], 0, "", "",
        )
    else:
        MID: tl.constexpr = (LO + HI) // 2
        if pid < TILE_STARTS[MID]:
            _dispatch_spatial_tiles(
                embed_ptr, output_ptr, pid, GRIDS, ROW_STARTS, TILE_STARTS,
                SCALES_H, SCALES_W, TILE_ROWS, TILE_COLS, TILE_GROUPS, G, M, D, S0, S1, LO, MID,
            )
        else:
            _dispatch_spatial_tiles(
                embed_ptr, output_ptr, pid, GRIDS, ROW_STARTS, TILE_STARTS,
                SCALES_H, SCALES_W, TILE_ROWS, TILE_COLS, TILE_GROUPS, G, M, D, S0, S1, MID, HI,
            )


@triton.jit
def _spatial_tiles_kernel(
    embed_ptr, output_ptr, GRIDS: tl.constexpr, ROW_STARTS: tl.constexpr,
    TILE_STARTS: tl.constexpr, SCALES_H: tl.constexpr, SCALES_W: tl.constexpr,
    TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr, TILE_GROUPS: tl.constexpr,
    G: tl.constexpr, M: tl.constexpr, D: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
):
    for pid in range(tl.program_id(0), TILE_STARTS[-1], tl.num_programs(0)):
        _dispatch_spatial_tiles(
            embed_ptr, output_ptr, pid, GRIDS, ROW_STARTS, TILE_STARTS,
            SCALES_H, SCALES_W, TILE_ROWS, TILE_COLS, TILE_GROUPS, G, M, D, S0, S1, 0, len(GRIDS),
        )


def _launch(embed_weight, flat_grids, g, m, dtype):
    grids = tuple(tuple(flat_grids[i : i + 3]) for i in range(0, len(flat_grids), 3))
    d = embed_weight.shape[1]
    block = 4096
    # Long videos keep the flat time-reuse path. Short inputs amortize index
    # arithmetic across a two-dimensional pixel/feature tile.
    use_spatial_tiles = bool(grids) and all(t <= 3 for t, _, _ in grids) and embed_weight.stride(1) == 1
    tile_rows = tuple(16 for _ in grids)
    tile_cols = tuple(128 for _ in grids)
    tile_groups = tuple(min(triton.cdiv(d, 128), 9) if h * w >= 512 else 1 for _, h, w in grids)
    rows, tiles = [0], [0]
    hs, ws = [], []
    for index, (t, h, w) in enumerate(grids):
        rows.append(rows[-1] + t * h * w)
        count = (triton.cdiv(h * w, tile_rows[index]) * triton.cdiv(triton.cdiv(d, tile_cols[index]), tile_groups[index])
                 if use_spatial_tiles else triton.cdiv(h * w * d, block))
        tiles.append(tiles[-1] + count)
        hs.append(float(g - 1) / float(h - 1) if h > 1 else 0.0)
        ws.append(float(g - 1) / float(w - 1) if w > 1 else 0.0)
    output = torch.empty((rows[-1], d), device=embed_weight.device, dtype=dtype)
    if grids:
        with torch.npu.device(embed_weight.device):
            cores = get_num_cores()
            # Complete waves retain the regular tile. Divide the remaining
            # rows among all cores with a smaller power-of-two tile, avoiding
            # a whole 16-row final tile on a subset of the cores.
            if (len(grids) == 1 and grids[0][0] == 1
                    and embed_weight.dtype == dtype == torch.bfloat16
                    and embed_weight.is_contiguous() and d % 128 == 0
                    and 128 <= d <= 9 * 128 and grids[0][1] * grids[0][2] > cores * 64):
                _, h, w = grids[0]
                if w > g and 16 % m == 0:
                    from mojo_opset.kernels._npu_vision_pos_embed_interp_vf import launch_vf_span

                    launch_vf_span(embed_weight, output, h, w, g, m)
                    return output
                full_tiles = h * w // (cores * 16) * cores
                remaining = h * w - full_tiles * 16
                tail_rows = triton.next_power_of_2(max(1, triton.cdiv(remaining, cores)))
                _balanced_spatial_tiles_kernel[(cores,)](
                    embed_weight, output, h, w, hs[0], ws[0], g, m,
                    d, *embed_weight.stride(), cores, full_tiles, tail_rows,
                    compile_mode="simt_only", num_warps=32,
                    simt_optimization_mode=1910101,
                    disable_fma=True, enable_fp_fusion=False,
                )
                return output
            if use_spatial_tiles:
                _spatial_tiles_kernel[(min(tiles[-1], get_num_cores()),)](
                    embed_weight, output, grids, tuple(rows), tuple(tiles),
                    tuple(hs), tuple(ws), tile_rows, tile_cols, tile_groups,
                    g, m, d, *embed_weight.stride(),
                    compile_mode="simt_only", num_warps=16,
                    disable_fma=True, enable_fp_fusion=False,
                )
                return output
            _bilinear_pos_embed_kernel[(tiles[-1],)](
                embed_weight,
                output,
                grids,
                tuple(rows),
                tuple(tiles),
                tuple(hs),
                tuple(ws),
                g,
                m,
                d,
                *embed_weight.stride(),
                block,
                compile_mode="simt_only",
                num_warps=16,
                disable_fma=True,
                enable_fp_fusion=False,
            )
    return output


@torch.library.custom_op("mojo_npu_triton_a5::vision_pos_embed_interp_batch_fwd", mutates_args=())
def vision_pos_embed_interp_batch_fwd(
    embed_weight: torch.Tensor,
    grids: list[int],
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _launch(embed_weight, grids, num_grid_per_side, m_size, dtype)


@vision_pos_embed_interp_batch_fwd.register_fake
def _batch_fake(embed_weight, grids, num_grid_per_side, m_size, dtype):
    total = sum(grids[i] * grids[i + 1] * grids[i + 2] for i in range(0, len(grids), 3))
    return embed_weight.new_empty((total, embed_weight.shape[1]), dtype=dtype)


def vision_pos_embed_interp_fwd(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return vision_pos_embed_interp_batch_fwd(embed_weight, [t, h, w], num_grid_per_side, m_size, dtype)
