# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2/A3 row scheduling and dtype-preserving position interpolation."""

import torch
import triton
import triton.language as tl

from ._npu_triton_utils import get_num_cores


@triton.jit
def _coefficient_fp32(value):
    if len(value.shape) == 0:
        # A2/A3 promote scalar bf16 casts back to f32 without narrowing.
        # Coefficients are finite, so integer round-to-nearest-even preserves
        # the BF16 value without relying on a scalar bf16 representation.
        bits = value.to(tl.uint32, bitcast=True)
        bits = (bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000
        return bits.to(tl.float32, bitcast=True)
    else:
        return value.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _bilinear_value(
    e00, e01, e10, e11, w00, w01, w10, w11, DTYPE: tl.constexpr,
    ZERO_HORIZONTAL: tl.constexpr = False,
):
    if DTYPE == tl.bfloat16 and e00.dtype == tl.bfloat16:
        if ZERO_HORIZONTAL:
            # Integral width coordinates give exactly zero w01 and w11.
            # Their products, and p00 + p01, are already representable in
            # BF16. The last addend is also zero or NaN, so rounding the
            # preceding sum can wait until the final cast. A sum of BF16
            # values cannot underflow below the smallest BF16 subnormal.
            # Keep every arithmetic operation, including signed-zero/NaN
            # propagation, while avoiding redundant conversion pairs.
            p00 = (_coefficient_fp32(w00) * e00.to(tl.float32)).to(DTYPE).to(tl.float32)
            p01 = w01 * e01.to(tl.float32)
            p10 = (_coefficient_fp32(w10) * e10.to(tl.float32)).to(DTYPE).to(tl.float32)
            p11 = w11 * e11.to(tl.float32)
            partial = (p00 + p01) + p10
            return (partial + p11).to(DTYPE)
        # Explicit narrowing retains four BF16 products and three additions;
        # implicit bf16 arithmetic is otherwise promoted by the backend.
        p00 = (_coefficient_fp32(w00) * e00.to(tl.float32)).to(DTYPE).to(tl.float32)
        p01 = (_coefficient_fp32(w01) * e01.to(tl.float32)).to(DTYPE).to(tl.float32)
        p10 = (_coefficient_fp32(w10) * e10.to(tl.float32)).to(DTYPE).to(tl.float32)
        p11 = (_coefficient_fp32(w11) * e11.to(tl.float32)).to(DTYPE).to(tl.float32)
        partial = (p00 + p01).to(DTYPE).to(tl.float32)
        partial = (partial + p10).to(DTYPE).to(tl.float32)
        return (partial + p11).to(DTYPE)
    elif DTYPE == tl.bfloat16:
        # FP32 tables retain FP32 products, but weights still narrow to BF16.
        return (_coefficient_fp32(w00) * e00 + _coefficient_fp32(w01) * e01
                + _coefficient_fp32(w10) * e10 + _coefficient_fp32(w11) * e11)
    else:
        return w00.to(DTYPE) * e00 + w01.to(DTYPE) * e01 + w10.to(DTYPE) * e10 + w11.to(DTYPE) * e11


@triton.jit
def _interpolate_features(
    embed_ptr,
    output_ptr,
    pid,
    out_start: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    hf,
    hc,
    wf,
    wc,
    w00,
    w01,
    w10,
    w11,
    OFFSET,
    C: tl.constexpr,
    REUSE_TIME: tl.constexpr,
    ZERO_HORIZONTAL: tl.constexpr,
):
    cols = OFFSET + tl.arange(0, C)
    mask = cols < D
    e00 = tl.load(embed_ptr + (hf * G + wf) * S0 + cols * S1, mask, other=0)
    e01 = tl.load(embed_ptr + (hf * G + wc) * S0 + cols * S1, mask, other=0)
    e10 = tl.load(embed_ptr + (hc * G + wf) * S0 + cols * S1, mask, other=0)
    e11 = tl.load(embed_ptr + (hc * G + wc) * S0 + cols * S1, mask, other=0)
    dtype = output_ptr.dtype.element_ty
    value = _bilinear_value(e00, e01, e10, e11, w00, w01, w10, w11, dtype, ZERO_HORIZONTAL)
    if REUSE_TIME:
        # Broadcast the already-rounded spatial result in UB. A single 2D
        # store avoids a separate synchronization sequence for every frame.
        if (T > 1 and T <= 16) and triton.next_power_of_2(T) * C * dtype.primitive_bitwidth // 8 <= 65536:
            frames = tl.arange(0, triton.next_power_of_2(T))
            offset = (out_start + frames[:, None] * H * W + pid) * D + cols[None, :]
            repeated = tl.broadcast_to(value[None, :], (triton.next_power_of_2(T), C))
            tl.store(output_ptr + offset, repeated, (frames[:, None] < T) & mask[None, :])
        elif T <= 16:
            for frame in tl.static_range(T):
                tl.store(output_ptr + (out_start + frame * H * W + pid) * D + cols, value, mask)
        else:
            for frame in range(T):
                tl.store(output_ptr + (out_start + frame * H * W + pid) * D + cols, value, mask)
    else:
        tl.store(output_ptr + (out_start + pid) * D + cols, value, mask)


@triton.jit
def _interpolate_row(
    embed_ptr,
    output_ptr,
    pid,
    col_block,
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
    BD: tl.constexpr,
    REUSE_TIME: tl.constexpr,
):
    spatial = pid % (H * W)
    block = spatial // (M * M)
    local = spatial % (M * M)
    row = block // (W // M) * M + local // M
    col = block % (W // M) * M + local % M
    h_frac = row.to(tl.float32) * HS
    hf = h_frac.to(tl.int32)
    if WS == 1.0:
        wf = col
        w_frac = col.to(tl.float32)
    else:
        w_frac = col.to(tl.float32) * WS
        wf = w_frac.to(tl.int32)
    hc = tl.minimum(hf + 1, G - 1)
    wc = tl.minimum(wf + 1, G - 1)
    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    if WS == 1.0:
        # Integral width coordinates have exactly zero horizontal fractions.
        # Retain all four products below, including their IEEE NaN behavior.
        w11 = tl.full((), 0.0, tl.float32)
        w01 = tl.full((), 0.0, tl.float32)
        w10 = dh
        w00 = 1.0 - dh
    else:
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01
    # Avoid computing 2048 features for a 1152-column row. Both segments
    # retain the same BF16 casts and expression order as the unsplit path.
    if (
        (T == 1 and H * W >= 512)
        and ((BD >= D and BD >= 128) and (D < BD and D - BD // 2 <= BD // 4))
    ):
        MAIN: tl.constexpr = BD // 2
        TAIL: tl.constexpr = triton.next_power_of_2(D - MAIN)
        _interpolate_features(
            embed_ptr,
            output_ptr,
            pid,
            out_start,
            T,
            H,
            W,
            G,
            D,
            S0,
            S1,
            hf,
            hc,
            wf,
            wc,
            w00,
            w01,
            w10,
            w11,
            0,
            MAIN,
            REUSE_TIME,
            WS == 1.0,
        )
        _interpolate_features(
            embed_ptr,
            output_ptr,
            pid,
            out_start,
            T,
            H,
            W,
            G,
            D,
            S0,
            S1,
            hf,
            hc,
            wf,
            wc,
            w00,
            w01,
            w10,
            w11,
            MAIN,
            TAIL,
            REUSE_TIME,
            WS == 1.0,
        )
    else:
        _interpolate_features(
            embed_ptr,
            output_ptr,
            pid,
            out_start,
            T,
            H,
            W,
            G,
            D,
            S0,
            S1,
            hf,
            hc,
            wf,
            wc,
            w00,
            w01,
            w10,
            w11,
            col_block * BD,
            BD,
            REUSE_TIME,
            WS == 1.0,
        )


@triton.jit
def _interpolate_width_span(
    embed_ptr, output_ptr, task, out_start: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr, HS: tl.constexpr,
    G: tl.constexpr, M: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
):
    # When W == G, adjacent output columns read contiguous source spans.
    # Keep all four neighbors: zero-weight terms still carry IEEE behavior.
    row = task // tl.cdiv(W, R)
    col = task % tl.cdiv(W, R) * R
    hr = row.to(tl.float32) * HS
    hf = hr.to(tl.int32)
    hc = tl.minimum(hf + 1, G - 1)
    dh = hr - hf.to(tl.float32)
    feature = tl.arange(0, R * D)
    valid = feature < (W - col) * D
    right_valid = feature < (W - col - 1) * D
    a = tl.load(embed_ptr + (hf * G + col) * D + feature, valid, 0)
    c = tl.load(embed_ptr + (hc * G + col) * D + feature, valid, 0)
    b = tl.load(embed_ptr + (hf * G + col + 1) * D + feature, right_valid, 0)
    d = tl.load(embed_ptr + (hc * G + col + 1) * D + feature, right_valid, 0)
    b = tl.where(right_valid, b, a)
    d = tl.where(right_valid, d, c)
    zero = tl.full((), 0.0, tl.float32)
    value = _bilinear_value(a, b, c, d, 1.0 - dh, zero, dh, zero, tl.bfloat16, True)
    group = col // M + tl.arange(0, R // M)
    within = tl.arange(0, M * D)
    offset = (out_start + (row // M * (W // M) + group[:, None]) * M * M
              + row % M * M) * D + within[None, :]
    tl.store(output_ptr + offset, value.reshape(R // M, M * D), group[:, None] < W // M)


@triton.jit
def _batch_body(
    embed_ptr,
    output_ptr,
    GRIDS: tl.constexpr,
    ROW_STARTS: tl.constexpr,
    TILE_STARTS: tl.constexpr,
    SCALES_H: tl.constexpr,
    SCALES_W: tl.constexpr,
    USE_WIDTH_SPAN: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    R: tl.constexpr = (8 // M) * M
    # Triton 3.2 needs list() to index constexpr tuples.
    for image in tl.static_range(len(GRIDS)):
        count = list(TILE_STARTS)[image + 1] - list(TILE_STARTS)[image]
        if list(USE_WIDTH_SPAN)[image]:
            for local in range(pid, list(GRIDS)[image][1] * tl.cdiv(list(GRIDS)[image][2], R), tl.num_programs(0)):
                _interpolate_width_span(
                    embed_ptr, output_ptr, local, list(ROW_STARTS)[image],
                    list(GRIDS)[image][1], list(GRIDS)[image][2], list(SCALES_H)[image], G, M, D, R,
                )
        else:
            for local in range(pid, count, tl.num_programs(0)):
                _interpolate_row(
                    embed_ptr,
                    output_ptr,
                    local // tl.cdiv(D, BD),
                    local % tl.cdiv(D, BD),
                    list(ROW_STARTS)[image],
                    list(GRIDS)[image][0],
                    list(GRIDS)[image][1],
                    list(GRIDS)[image][2],
                    list(SCALES_H)[image],
                    list(SCALES_W)[image],
                    G,
                    M,
                    D,
                    S0,
                    S1,
                    BD,
                    True,
                )


def launch(kernel, embed_weight, flat_grids, g, m, dtype):
    """One allocation/launch; grid metadata contains host integers and floats."""
    grids = tuple(tuple(flat_grids[i : i + 3]) for i in range(0, len(flat_grids), 3))
    d = embed_weight.shape[1]
    # SIMD accepts exact (non-power-of-two) feature widths. Avoid padded
    # arithmetic and a second segment for contiguous rows. Strided gathers
    # still need smaller UB tiles to avoid an extra copy kernel.
    limit = 128 if embed_weight.stride(1) != 1 else 2048
    bd = min(limit, triton.next_power_of_2(d) if embed_weight.stride(1) != 1 else d)
    s0, s1 = embed_weight.stride()
    width_span_supported = (
        embed_weight.dtype == dtype == torch.bfloat16 and (s0, s1) == (d, 1)
        and m <= 8 and d <= 1152 and d % 16 == 0
    )
    rows, tiles = [0], [0]
    hs, ws, use_width_span = [], [], []
    for t, h, w in grids:
        rows.append(rows[-1] + t * h * w)
        tiles.append(tiles[-1] + h * w * triton.cdiv(d, bd))
        hs.append(float(g - 1) / float(h - 1) if h > 1 else 0.0)
        ws.append(float(g - 1) / float(w - 1) if w > 1 else 0.0)
        # Resolve the specialization on the host for Triton 3.2.
        use_width_span.append(width_span_supported and t == 1 and ws[-1] == 1.0)
    output = torch.empty((rows[-1], d), device=embed_weight.device, dtype=dtype)
    # The 2D store lowering does not preserve signed zero for unaligned
    # spans. Keep those widths on the existing row path.
    only_width_spans = width_span_supported and all(t == 1 and w == g and w > 1 for t, _, w in grids)
    if grids:
        with torch.npu.device(embed_weight.device):
            kernel[(min(tiles[-1], get_num_cores()),)](
                embed_weight,
                output,
                grids,
                tuple(rows),
                tuple(tiles),
                tuple(hs),
                tuple(ws),
                tuple(use_width_span),
                g,
                m,
                d,
                s0,
                s1,
                bd,
                enable_fp_fusion=False,
                multibuffer=not only_width_spans,
            )
    return output
