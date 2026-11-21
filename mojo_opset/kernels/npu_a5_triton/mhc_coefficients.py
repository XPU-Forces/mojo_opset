"""Fused four-stream manifold-constrained Hyper-Connections on Ascend A5."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.runtime.driver as driver

HC_MULT = 4
MIX_WIDTH = 24
BLOCK_N = 32
HC_EPS = 1e-6
NORM_EPS = 1e-5
BLOCK_M = 64
BLOCK_K = 512
BLOCK_C = 1024


@triton.jit
def mhc_gemm_kernel(
    x_ptr,
    phi_ptr,
    mixes_ptr,
    tokens,
    k_size: tl.constexpr,
    mix_width: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    norm_eps: tl.constexpr,
    dot_hf32: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    tile_count = tl.cdiv(tokens, block_m)
    for tile in tl.range(pid, tile_count, nprog):
        rows = tile * block_m + tl.arange(0, block_m)
        row_mask = rows < tokens
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        sumsq = tl.zeros((block_m,), dtype=tl.float32)
        k_tiles = k_size // block_k
        for ki in tl.range(0, k_tiles):
            k0 = ki * block_k
            offs_k = k0 + tl.arange(0, block_k)
            offs_n = tl.arange(0, block_n)
            a = tl.load(
                x_ptr + rows[:, None] * k_size + offs_k[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            sumsq += tl.sum(a * a, axis=1)
            b = tl.load(
                phi_ptr + offs_n[None, :] * k_size + offs_k[:, None],
                mask=offs_n[None, :] < mix_width,
                other=0.0,
            )
            if dot_hf32:
                acc = tl.dot(a, b, acc, input_precision="hf32")
            else:
                acc = tl.dot(a, b, acc)
        inv = tl.rsqrt(sumsq / k_size + norm_eps)
        acc = acc * inv[:, None]
        offs_n = tl.arange(0, block_n)
        tl.store(
            mixes_ptr + rows[:, None] * mix_width + offs_n[None, :],
            acc,
            mask=row_mask[:, None] & (offs_n[None, :] < mix_width),
        )


@triton.jit
def mhc_pack_phi_pair_kernel(
    phi_ptr, packed_ptr, k_size: tl.constexpr, block_n: tl.constexpr, mix_width: tl.constexpr, pack_block: tl.constexpr
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    kk = tl.arange(0, pack_block)
    total_tiles = block_n * k_size // pack_block
    valid_tiles = mix_width * k_size // pack_block
    for tile in tl.range(pid, total_tiles, nprog):
        off = tile * pack_block + kk
        if tile < valid_tiles:
            value = tl.load(phi_ptr + off)
        else:
            value = tl.full((pack_block,), 0.0, tl.float32)
        hi = value.to(packed_ptr.dtype.element_ty)
        lo = (value - hi.to(tl.float32)).to(packed_ptr.dtype.element_ty)
        tl.store(packed_ptr + off, hi)
        tl.store(packed_ptr + block_n * k_size + off, lo)


@triton.jit
def mhc_gemm_part_kernel(
    x_ptr,
    phi_ptr,
    partial_ptr,
    sumsq_ptr,
    tokens: tl.constexpr,
    k_parts: tl.constexpr,
    tile_begin: tl.constexpr,
    k_size: tl.constexpr,
    mix_width: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    dot_hf32: tl.constexpr,
    packed_phi: tl.constexpr = False,
):
    tile = tl.program_id(0) + tile_begin
    part = tl.program_id(1)
    rows = tile * block_m + tl.arange(0, block_m)
    row_mask = rows < tokens
    offs_n = tl.arange(0, block_n)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    if packed_phi:
        acc_low = tl.zeros((block_m, block_n), dtype=tl.float32)
        gram = tl.zeros((block_m, block_m), dtype=tl.float32)
    sumsq = tl.zeros((block_m,), dtype=tl.float32)
    k_tiles = k_size // block_k
    k_begin = part * k_tiles // k_parts
    k_end = (part + 1) * k_tiles // k_parts
    for ki in tl.range(k_begin, k_end):
        k0 = ki * block_k
        offs_k = k0 + tl.arange(0, block_k)
        a = tl.load(
            x_ptr + rows[:, None] * k_size + offs_k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        if not packed_phi:
            sumsq += tl.sum(a * a, axis=1)
        if packed_phi:
            bp = offs_n[None, :] * k_size + offs_k[:, None]
            hi = tl.load(phi_ptr + bp)
            lo = tl.load(phi_ptr + block_n * k_size + bp)
            aa = a.to(phi_ptr.dtype.element_ty)
            acc = tl.dot(aa, hi, acc)
            acc_low = tl.dot(aa, lo, acc_low)
            gram = tl.dot(aa, tl.trans(aa), gram)
        else:
            b = tl.load(
                phi_ptr + offs_n[None, :] * k_size + offs_k[:, None],
                mask=offs_n[None, :] < mix_width,
                other=0.0,
            )
            if dot_hf32:
                acc = tl.dot(a, b, acc, input_precision="hf32")
            else:
                acc = tl.dot(a, b, acc)
    if packed_phi:
        acc = acc + acc_low
        diag = tl.arange(0, block_m)
        sumsq = tl.sum(tl.where(diag[:, None] == diag[None, :], gram, 0.0), axis=1)
    base = (part * tokens + rows) * block_n
    tl.store(partial_ptr + base[:, None] + offs_n[None, :], acc, mask=row_mask[:, None])
    tl.store(sumsq_ptr + part * tokens + rows, sumsq, mask=row_mask)


@triton.jit
def mhc_epilogue_kernel(
    mixes_ptr,
    alpha_ptr,
    bias_ptr,
    post_ptr,
    res_ptr,
    pre_ptr,
    x_ptr,
    hin_ptr,
    tokens,
    mix_width: tl.constexpr,
    streams: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    hc_eps: tl.constexpr,
    hidden: tl.constexpr,
    block_c: tl.constexpr,
    store_bf16: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, streams)
    c_tiles = hidden // block_c
    offs = tl.arange(0, block_c)
    a0 = tl.load(alpha_ptr)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    pre_bias = tl.load(bias_ptr + cols)
    post_bias = tl.load(bias_ptr + streams + cols)
    for row in tl.range(pid, tokens, nprog):
        base = mixes_ptr + row * mix_width
        pre_logit = tl.load(base + cols) * a0 + pre_bias
        post_logit = tl.load(base + streams + cols) * a1 + post_bias
        pre = tl.sigmoid(pre_logit) + hc_eps
        post = 2.0 * tl.sigmoid(post_logit)
        row_idx = tl.arange(0, streams)
        col_idx = tl.arange(0, streams)
        logit_idx = 2 * streams + row_idx[:, None] * streams + col_idx[None, :]
        logits = tl.load(base + logit_idx) * a2 + tl.load(bias_ptr + logit_idx)
        row_max = tl.max(logits, axis=1)
        expo = tl.exp(logits - row_max[:, None])
        normed = expo / tl.sum(expo, axis=1)[:, None] + hc_eps
        col_sum = tl.sum(normed, axis=0) + hc_eps
        normed = normed / col_sum[None, :]
        for _step in tl.static_range(19):
            row_sum = tl.sum(normed, axis=1) + hc_eps
            normed = normed / row_sum[:, None]
            col_sum = tl.sum(normed, axis=0) + hc_eps
            normed = normed / col_sum[None, :]
        pre_base = pre_ptr + row * streams
        tl.store(pre_base + cols, pre)
        tl.store(post_ptr + row * streams + cols, post)
        tl.store(
            res_ptr + row * (streams * streams) + logit_idx - 2 * streams,
            normed,
        )
        w0 = tl.load(pre_base)
        w1 = tl.load(pre_base + 1)
        w2 = tl.load(pre_base + 2)
        w3 = tl.load(pre_base + 3)
        x_base = x_ptr + row * streams * hidden
        for tile in tl.range(0, c_tiles):
            ccols = tile * block_c + offs
            v0 = tl.load(x_base + ccols).to(tl.float32)
            v1 = tl.load(x_base + hidden + ccols).to(tl.float32)
            v2 = tl.load(x_base + 2 * hidden + ccols).to(tl.float32)
            v3 = tl.load(x_base + 3 * hidden + ccols).to(tl.float32)
            hin = v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3
            if store_bf16:
                stored = hin.to(tl.bfloat16)
            else:
                stored = hin.to(tl.float16)
            tl.store(hin_ptr + row * hidden + ccols, stored)


@triton.jit
def mhc_split_epilogue_small_kernel(
    partial_ptr,
    sumsq_ptr,
    mixes_ptr,
    alpha_ptr,
    bias_ptr,
    post_ptr,
    res_ptr,
    pre_ptr,
    x_ptr,
    hin_ptr,
    tokens: tl.constexpr,
    k_parts: tl.constexpr,
    row_begin: tl.constexpr,
    row_end: tl.constexpr,
    k_size: tl.constexpr,
    mix_width: tl.constexpr,
    block_n: tl.constexpr,
    norm_eps: tl.constexpr,
    streams: tl.constexpr,
    hc_eps: tl.constexpr,
    hidden: tl.constexpr,
    block_c: tl.constexpr,
    store_bf16: tl.constexpr,
    batch_rows: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rr = tl.arange(0, batch_rows)
    nn = tl.arange(0, block_n)
    cols = tl.arange(0, streams)
    coffs = tl.arange(0, block_c)
    groups = tl.cdiv(row_end - row_begin, batch_rows)
    a0 = tl.load(alpha_ptr)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    tl.load(bias_ptr + cols)
    tl.load(bias_ptr + streams + cols)
    bias_0 = tl.load(bias_ptr + 0)
    bias_1 = tl.load(bias_ptr + 1)
    bias_2 = tl.load(bias_ptr + 2)
    bias_3 = tl.load(bias_ptr + 3)
    bias_4 = tl.load(bias_ptr + 4)
    bias_5 = tl.load(bias_ptr + 5)
    bias_6 = tl.load(bias_ptr + 6)
    bias_7 = tl.load(bias_ptr + 7)
    bias_8 = tl.load(bias_ptr + 8)
    bias_9 = tl.load(bias_ptr + 9)
    bias_10 = tl.load(bias_ptr + 10)
    bias_11 = tl.load(bias_ptr + 11)
    bias_12 = tl.load(bias_ptr + 12)
    bias_13 = tl.load(bias_ptr + 13)
    bias_14 = tl.load(bias_ptr + 14)
    bias_15 = tl.load(bias_ptr + 15)
    bias_16 = tl.load(bias_ptr + 16)
    bias_17 = tl.load(bias_ptr + 17)
    bias_18 = tl.load(bias_ptr + 18)
    bias_19 = tl.load(bias_ptr + 19)
    bias_20 = tl.load(bias_ptr + 20)
    bias_21 = tl.load(bias_ptr + 21)
    bias_22 = tl.load(bias_ptr + 22)
    bias_23 = tl.load(bias_ptr + 23)
    for group in tl.range(pid, groups, nprog):
        rows = row_begin + group * batch_rows + rr
        mask = rows < row_end
        pp = tl.arange(0, triton.next_power_of_2(k_parts))
        partial_values = tl.load(
            partial_ptr + (pp[:, None, None] * tokens + rows[None, :, None]) * block_n + nn[None, None, :],
            mask=(pp[:, None, None] < k_parts) & mask[None, :, None],
            other=0.0,
        )
        acc = tl.sum(partial_values, axis=0)
        sq_values = tl.load(
            sumsq_ptr + pp[:, None] * tokens + rows[None, :], mask=(pp[:, None] < k_parts) & mask[None, :], other=0.0
        )
        sq = tl.sum(sq_values, axis=0)
        acc *= tl.rsqrt(sq / k_size + norm_eps)[:, None]
        w0 = (
            tl.sigmoid(al.extract_slice(acc, (0, 0), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_0)
            + hc_eps
        )
        p0 = 2.0 * tl.sigmoid(
            al.extract_slice(acc, (0, 4), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_4
        )
        w1 = (
            tl.sigmoid(al.extract_slice(acc, (0, 1), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_1)
            + hc_eps
        )
        p1 = 2.0 * tl.sigmoid(
            al.extract_slice(acc, (0, 5), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_5
        )
        w2 = (
            tl.sigmoid(al.extract_slice(acc, (0, 2), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_2)
            + hc_eps
        )
        p2 = 2.0 * tl.sigmoid(
            al.extract_slice(acc, (0, 6), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_6
        )
        w3 = (
            tl.sigmoid(al.extract_slice(acc, (0, 3), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_3)
            + hc_eps
        )
        p3 = 2.0 * tl.sigmoid(
            al.extract_slice(acc, (0, 7), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_7
        )
        v00 = al.extract_slice(acc, (0, 8), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_8
        v01 = al.extract_slice(acc, (0, 9), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_9
        v02 = al.extract_slice(acc, (0, 10), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_10
        v03 = al.extract_slice(acc, (0, 11), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_11
        m0 = tl.maximum(tl.maximum(v00, v01), tl.maximum(v02, v03))
        v00 = tl.exp(v00 - m0)
        v01 = tl.exp(v01 - m0)
        v02 = tl.exp(v02 - m0)
        v03 = tl.exp(v03 - m0)
        d0 = v00 + v01 + v02 + v03
        v00 = v00 / d0 + hc_eps
        v01 = v01 / d0 + hc_eps
        v02 = v02 / d0 + hc_eps
        v03 = v03 / d0 + hc_eps
        v10 = al.extract_slice(acc, (0, 12), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_12
        v11 = al.extract_slice(acc, (0, 13), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_13
        v12 = al.extract_slice(acc, (0, 14), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_14
        v13 = al.extract_slice(acc, (0, 15), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_15
        m1 = tl.maximum(tl.maximum(v10, v11), tl.maximum(v12, v13))
        v10 = tl.exp(v10 - m1)
        v11 = tl.exp(v11 - m1)
        v12 = tl.exp(v12 - m1)
        v13 = tl.exp(v13 - m1)
        d1 = v10 + v11 + v12 + v13
        v10 = v10 / d1 + hc_eps
        v11 = v11 / d1 + hc_eps
        v12 = v12 / d1 + hc_eps
        v13 = v13 / d1 + hc_eps
        v20 = al.extract_slice(acc, (0, 16), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_16
        v21 = al.extract_slice(acc, (0, 17), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_17
        v22 = al.extract_slice(acc, (0, 18), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_18
        v23 = al.extract_slice(acc, (0, 19), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_19
        m2 = tl.maximum(tl.maximum(v20, v21), tl.maximum(v22, v23))
        v20 = tl.exp(v20 - m2)
        v21 = tl.exp(v21 - m2)
        v22 = tl.exp(v22 - m2)
        v23 = tl.exp(v23 - m2)
        d2 = v20 + v21 + v22 + v23
        v20 = v20 / d2 + hc_eps
        v21 = v21 / d2 + hc_eps
        v22 = v22 / d2 + hc_eps
        v23 = v23 / d2 + hc_eps
        v30 = al.extract_slice(acc, (0, 20), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_20
        v31 = al.extract_slice(acc, (0, 21), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_21
        v32 = al.extract_slice(acc, (0, 22), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_22
        v33 = al.extract_slice(acc, (0, 23), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_23
        m3 = tl.maximum(tl.maximum(v30, v31), tl.maximum(v32, v33))
        v30 = tl.exp(v30 - m3)
        v31 = tl.exp(v31 - m3)
        v32 = tl.exp(v32 - m3)
        v33 = tl.exp(v33 - m3)
        d3 = v30 + v31 + v32 + v33
        v30 = v30 / d3 + hc_eps
        v31 = v31 / d3 + hc_eps
        v32 = v32 / d3 + hc_eps
        v33 = v33 / d3 + hc_eps
        d0 = v00 + v10 + v20 + v30 + hc_eps
        v00 = v00 / d0
        v10 = v10 / d0
        v20 = v20 / d0
        v30 = v30 / d0
        d1 = v01 + v11 + v21 + v31 + hc_eps
        v01 = v01 / d1
        v11 = v11 / d1
        v21 = v21 / d1
        v31 = v31 / d1
        d2 = v02 + v12 + v22 + v32 + hc_eps
        v02 = v02 / d2
        v12 = v12 / d2
        v22 = v22 / d2
        v32 = v32 / d2
        d3 = v03 + v13 + v23 + v33 + hc_eps
        v03 = v03 / d3
        v13 = v13 / d3
        v23 = v23 / d3
        v33 = v33 / d3
        for _step in tl.range(19):
            d0 = v00 + v01 + v02 + v03 + hc_eps
            v00 = v00 / d0
            v01 = v01 / d0
            v02 = v02 / d0
            v03 = v03 / d0
            d1 = v10 + v11 + v12 + v13 + hc_eps
            v10 = v10 / d1
            v11 = v11 / d1
            v12 = v12 / d1
            v13 = v13 / d1
            d2 = v20 + v21 + v22 + v23 + hc_eps
            v20 = v20 / d2
            v21 = v21 / d2
            v22 = v22 / d2
            v23 = v23 / d2
            d3 = v30 + v31 + v32 + v33 + hc_eps
            v30 = v30 / d3
            v31 = v31 / d3
            v32 = v32 / d3
            v33 = v33 / d3
            d0 = v00 + v10 + v20 + v30 + hc_eps
            v00 = v00 / d0
            v10 = v10 / d0
            v20 = v20 / d0
            v30 = v30 / d0
            d1 = v01 + v11 + v21 + v31 + hc_eps
            v01 = v01 / d1
            v11 = v11 / d1
            v21 = v21 / d1
            v31 = v31 / d1
            d2 = v02 + v12 + v22 + v32 + hc_eps
            v02 = v02 / d2
            v12 = v12 / d2
            v22 = v22 / d2
            v32 = v32 / d2
            d3 = v03 + v13 + v23 + v33 + hc_eps
            v03 = v03 / d3
            v13 = v13 / d3
            v23 = v23 / d3
            v33 = v33 / d3
        pre_join_1 = tl.join(w0, w2).reshape((batch_rows, 2))
        pre_join_2 = tl.join(w1, w3).reshape((batch_rows, 2))
        pre_join_3 = tl.join(pre_join_1, pre_join_2).reshape((batch_rows, 4))
        post_join_4 = tl.join(p0, p2).reshape((batch_rows, 2))
        post_join_5 = tl.join(p1, p3).reshape((batch_rows, 2))
        post_join_6 = tl.join(post_join_4, post_join_5).reshape((batch_rows, 4))
        res_join_7 = tl.join(v00, v20).reshape((batch_rows, 2))
        res_join_8 = tl.join(v10, v30).reshape((batch_rows, 2))
        res_join_9 = tl.join(res_join_7, res_join_8).reshape((batch_rows, 4))
        res_join_10 = tl.join(v02, v22).reshape((batch_rows, 2))
        res_join_11 = tl.join(v12, v32).reshape((batch_rows, 2))
        res_join_12 = tl.join(res_join_10, res_join_11).reshape((batch_rows, 4))
        res_join_13 = tl.join(res_join_9, res_join_12).reshape((batch_rows, 8))
        res_join_14 = tl.join(v01, v21).reshape((batch_rows, 2))
        res_join_15 = tl.join(v11, v31).reshape((batch_rows, 2))
        res_join_16 = tl.join(res_join_14, res_join_15).reshape((batch_rows, 4))
        res_join_17 = tl.join(v03, v23).reshape((batch_rows, 2))
        res_join_18 = tl.join(v13, v33).reshape((batch_rows, 2))
        res_join_19 = tl.join(res_join_17, res_join_18).reshape((batch_rows, 4))
        res_join_20 = tl.join(res_join_16, res_join_19).reshape((batch_rows, 8))
        res_join_21 = tl.join(res_join_13, res_join_20).reshape((batch_rows, 16))
        tl.store(pre_ptr + rows[:, None] * streams + cols[None, :], pre_join_3, mask=mask[:, None])
        tl.store(post_ptr + rows[:, None] * streams + cols[None, :], post_join_6, mask=mask[:, None])
        tl.store(res_ptr + rows[:, None] * 16 + tl.arange(0, 16)[None, :], res_join_21, mask=mask[:, None])
        for tile in tl.range(0, hidden // block_c):
            cc = tile * block_c + coffs
            xb = x_ptr + rows[:, None] * streams * hidden + cc[None, :]
            v0 = tl.load(xb, mask=mask[:, None], other=0.0).to(tl.float32)
            v1 = tl.load(xb + hidden, mask=mask[:, None], other=0.0).to(tl.float32)
            v2 = tl.load(xb + 2 * hidden, mask=mask[:, None], other=0.0).to(tl.float32)
            v3 = tl.load(xb + 3 * hidden, mask=mask[:, None], other=0.0).to(tl.float32)
            hin = v0 * w0[:, None] + v1 * w1[:, None] + v2 * w2[:, None] + v3 * w3[:, None]
            tl.store(hin_ptr + rows[:, None] * hidden + cc[None, :], hin, mask=mask[:, None])


@triton.autotune(
    configs=[triton.Config({"block_c": 256}), triton.Config({"block_c": 512}), triton.Config({"block_c": 1024})],
    key=["tokens", "hidden", "k_parts"],
    cache_results=True,
)
@triton.jit
def mhc_split_epilogue_kernel(
    partial_ptr,
    sumsq_ptr,
    mixes_ptr,
    alpha_ptr,
    bias_ptr,
    post_ptr,
    res_ptr,
    pre_ptr,
    x_ptr,
    hin_ptr,
    tokens: tl.constexpr,
    k_parts: tl.constexpr,
    row_begin: tl.constexpr,
    row_end: tl.constexpr,
    k_size: tl.constexpr,
    mix_width: tl.constexpr,
    block_n: tl.constexpr,
    norm_eps: tl.constexpr,
    streams: tl.constexpr,
    hc_eps: tl.constexpr,
    hidden: tl.constexpr,
    block_c: tl.constexpr,
    store_bf16: tl.constexpr,
    batch_rows: tl.constexpr = 8,
):
    if tokens == 128:
        pid = tl.program_id(0)
        coeff_progs: tl.constexpr = tl.cdiv(tokens, batch_rows)
        if pid < coeff_progs:
            pid = tl.program_id(0)
            nprog = tl.num_programs(0)
            rr = tl.arange(0, batch_rows)
            nn = tl.arange(0, block_n)
            cols = tl.arange(0, streams)
            coffs = tl.arange(0, block_c)
            groups = tl.cdiv(row_end - row_begin, batch_rows)
            a0 = tl.load(alpha_ptr)
            a1 = tl.load(alpha_ptr + 1)
            a2 = tl.load(alpha_ptr + 2)
            tl.load(bias_ptr + cols)
            tl.load(bias_ptr + streams + cols)
            bias_0 = tl.load(bias_ptr + 0)
            bias_1 = tl.load(bias_ptr + 1)
            bias_2 = tl.load(bias_ptr + 2)
            bias_3 = tl.load(bias_ptr + 3)
            bias_4 = tl.load(bias_ptr + 4)
            bias_5 = tl.load(bias_ptr + 5)
            bias_6 = tl.load(bias_ptr + 6)
            bias_7 = tl.load(bias_ptr + 7)
            bias_8 = tl.load(bias_ptr + 8)
            bias_9 = tl.load(bias_ptr + 9)
            bias_10 = tl.load(bias_ptr + 10)
            bias_11 = tl.load(bias_ptr + 11)
            bias_12 = tl.load(bias_ptr + 12)
            bias_13 = tl.load(bias_ptr + 13)
            bias_14 = tl.load(bias_ptr + 14)
            bias_15 = tl.load(bias_ptr + 15)
            bias_16 = tl.load(bias_ptr + 16)
            bias_17 = tl.load(bias_ptr + 17)
            bias_18 = tl.load(bias_ptr + 18)
            bias_19 = tl.load(bias_ptr + 19)
            bias_20 = tl.load(bias_ptr + 20)
            bias_21 = tl.load(bias_ptr + 21)
            bias_22 = tl.load(bias_ptr + 22)
            bias_23 = tl.load(bias_ptr + 23)
            for group in tl.range(pid, groups, nprog):
                rows = row_begin + group * batch_rows + rr
                mask = rows < row_end
                pp = tl.arange(0, triton.next_power_of_2(k_parts))
                partial_values = tl.load(
                    partial_ptr + (pp[:, None, None] * tokens + rows[None, :, None]) * block_n + nn[None, None, :],
                    mask=(pp[:, None, None] < k_parts) & mask[None, :, None],
                    other=0.0,
                )
                acc = tl.sum(partial_values, axis=0)
                sq_values = tl.load(
                    sumsq_ptr + pp[:, None] * tokens + rows[None, :],
                    mask=(pp[:, None] < k_parts) & mask[None, :],
                    other=0.0,
                )
                sq = tl.sum(sq_values, axis=0)
                acc *= tl.rsqrt(sq / k_size + norm_eps)[:, None]
                w0 = (
                    tl.sigmoid(
                        al.extract_slice(acc, (0, 0), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_0
                    )
                    + hc_eps
                )
                p0 = 2.0 * tl.sigmoid(
                    al.extract_slice(acc, (0, 4), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_4
                )
                w1 = (
                    tl.sigmoid(
                        al.extract_slice(acc, (0, 1), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_1
                    )
                    + hc_eps
                )
                p1 = 2.0 * tl.sigmoid(
                    al.extract_slice(acc, (0, 5), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_5
                )
                w2 = (
                    tl.sigmoid(
                        al.extract_slice(acc, (0, 2), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_2
                    )
                    + hc_eps
                )
                p2 = 2.0 * tl.sigmoid(
                    al.extract_slice(acc, (0, 6), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_6
                )
                w3 = (
                    tl.sigmoid(
                        al.extract_slice(acc, (0, 3), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_3
                    )
                    + hc_eps
                )
                p3 = 2.0 * tl.sigmoid(
                    al.extract_slice(acc, (0, 7), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_7
                )
                v00 = al.extract_slice(acc, (0, 8), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_8
                v01 = al.extract_slice(acc, (0, 9), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_9
                v02 = al.extract_slice(acc, (0, 10), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_10
                v03 = al.extract_slice(acc, (0, 11), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_11
                m0 = tl.maximum(tl.maximum(v00, v01), tl.maximum(v02, v03))
                v00 = tl.exp(v00 - m0)
                v01 = tl.exp(v01 - m0)
                v02 = tl.exp(v02 - m0)
                v03 = tl.exp(v03 - m0)
                d0 = v00 + v01 + v02 + v03
                v00 = v00 / d0 + hc_eps
                v01 = v01 / d0 + hc_eps
                v02 = v02 / d0 + hc_eps
                v03 = v03 / d0 + hc_eps
                v10 = al.extract_slice(acc, (0, 12), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_12
                v11 = al.extract_slice(acc, (0, 13), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_13
                v12 = al.extract_slice(acc, (0, 14), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_14
                v13 = al.extract_slice(acc, (0, 15), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_15
                m1 = tl.maximum(tl.maximum(v10, v11), tl.maximum(v12, v13))
                v10 = tl.exp(v10 - m1)
                v11 = tl.exp(v11 - m1)
                v12 = tl.exp(v12 - m1)
                v13 = tl.exp(v13 - m1)
                d1 = v10 + v11 + v12 + v13
                v10 = v10 / d1 + hc_eps
                v11 = v11 / d1 + hc_eps
                v12 = v12 / d1 + hc_eps
                v13 = v13 / d1 + hc_eps
                v20 = al.extract_slice(acc, (0, 16), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_16
                v21 = al.extract_slice(acc, (0, 17), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_17
                v22 = al.extract_slice(acc, (0, 18), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_18
                v23 = al.extract_slice(acc, (0, 19), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_19
                m2 = tl.maximum(tl.maximum(v20, v21), tl.maximum(v22, v23))
                v20 = tl.exp(v20 - m2)
                v21 = tl.exp(v21 - m2)
                v22 = tl.exp(v22 - m2)
                v23 = tl.exp(v23 - m2)
                d2 = v20 + v21 + v22 + v23
                v20 = v20 / d2 + hc_eps
                v21 = v21 / d2 + hc_eps
                v22 = v22 / d2 + hc_eps
                v23 = v23 / d2 + hc_eps
                v30 = al.extract_slice(acc, (0, 20), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_20
                v31 = al.extract_slice(acc, (0, 21), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_21
                v32 = al.extract_slice(acc, (0, 22), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_22
                v33 = al.extract_slice(acc, (0, 23), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_23
                m3 = tl.maximum(tl.maximum(v30, v31), tl.maximum(v32, v33))
                v30 = tl.exp(v30 - m3)
                v31 = tl.exp(v31 - m3)
                v32 = tl.exp(v32 - m3)
                v33 = tl.exp(v33 - m3)
                d3 = v30 + v31 + v32 + v33
                v30 = v30 / d3 + hc_eps
                v31 = v31 / d3 + hc_eps
                v32 = v32 / d3 + hc_eps
                v33 = v33 / d3 + hc_eps
                d0 = v00 + v10 + v20 + v30 + hc_eps
                v00 = v00 / d0
                v10 = v10 / d0
                v20 = v20 / d0
                v30 = v30 / d0
                d1 = v01 + v11 + v21 + v31 + hc_eps
                v01 = v01 / d1
                v11 = v11 / d1
                v21 = v21 / d1
                v31 = v31 / d1
                d2 = v02 + v12 + v22 + v32 + hc_eps
                v02 = v02 / d2
                v12 = v12 / d2
                v22 = v22 / d2
                v32 = v32 / d2
                d3 = v03 + v13 + v23 + v33 + hc_eps
                v03 = v03 / d3
                v13 = v13 / d3
                v23 = v23 / d3
                v33 = v33 / d3
                for _step in tl.range(19):
                    d0 = v00 + v01 + v02 + v03 + hc_eps
                    v00 = v00 / d0
                    v01 = v01 / d0
                    v02 = v02 / d0
                    v03 = v03 / d0
                    d1 = v10 + v11 + v12 + v13 + hc_eps
                    v10 = v10 / d1
                    v11 = v11 / d1
                    v12 = v12 / d1
                    v13 = v13 / d1
                    d2 = v20 + v21 + v22 + v23 + hc_eps
                    v20 = v20 / d2
                    v21 = v21 / d2
                    v22 = v22 / d2
                    v23 = v23 / d2
                    d3 = v30 + v31 + v32 + v33 + hc_eps
                    v30 = v30 / d3
                    v31 = v31 / d3
                    v32 = v32 / d3
                    v33 = v33 / d3
                    d0 = v00 + v10 + v20 + v30 + hc_eps
                    v00 = v00 / d0
                    v10 = v10 / d0
                    v20 = v20 / d0
                    v30 = v30 / d0
                    d1 = v01 + v11 + v21 + v31 + hc_eps
                    v01 = v01 / d1
                    v11 = v11 / d1
                    v21 = v21 / d1
                    v31 = v31 / d1
                    d2 = v02 + v12 + v22 + v32 + hc_eps
                    v02 = v02 / d2
                    v12 = v12 / d2
                    v22 = v22 / d2
                    v32 = v32 / d2
                    d3 = v03 + v13 + v23 + v33 + hc_eps
                    v03 = v03 / d3
                    v13 = v13 / d3
                    v23 = v23 / d3
                    v33 = v33 / d3
                pre_join_1 = tl.join(w0, w2).reshape((batch_rows, 2))
                pre_join_2 = tl.join(w1, w3).reshape((batch_rows, 2))
                pre_join_3 = tl.join(pre_join_1, pre_join_2).reshape((batch_rows, 4))
                post_join_4 = tl.join(p0, p2).reshape((batch_rows, 2))
                post_join_5 = tl.join(p1, p3).reshape((batch_rows, 2))
                post_join_6 = tl.join(post_join_4, post_join_5).reshape((batch_rows, 4))
                res_join_7 = tl.join(v00, v20).reshape((batch_rows, 2))
                res_join_8 = tl.join(v10, v30).reshape((batch_rows, 2))
                res_join_9 = tl.join(res_join_7, res_join_8).reshape((batch_rows, 4))
                res_join_10 = tl.join(v02, v22).reshape((batch_rows, 2))
                res_join_11 = tl.join(v12, v32).reshape((batch_rows, 2))
                res_join_12 = tl.join(res_join_10, res_join_11).reshape((batch_rows, 4))
                res_join_13 = tl.join(res_join_9, res_join_12).reshape((batch_rows, 8))
                res_join_14 = tl.join(v01, v21).reshape((batch_rows, 2))
                res_join_15 = tl.join(v11, v31).reshape((batch_rows, 2))
                res_join_16 = tl.join(res_join_14, res_join_15).reshape((batch_rows, 4))
                res_join_17 = tl.join(v03, v23).reshape((batch_rows, 2))
                res_join_18 = tl.join(v13, v33).reshape((batch_rows, 2))
                res_join_19 = tl.join(res_join_17, res_join_18).reshape((batch_rows, 4))
                res_join_20 = tl.join(res_join_16, res_join_19).reshape((batch_rows, 8))
                res_join_21 = tl.join(res_join_13, res_join_20).reshape((batch_rows, 16))
                tl.store(pre_ptr + rows[:, None] * streams + cols[None, :], pre_join_3, mask=mask[:, None])
                tl.store(post_ptr + rows[:, None] * streams + cols[None, :], post_join_6, mask=mask[:, None])
                tl.store(res_ptr + rows[:, None] * 16 + tl.arange(0, 16)[None, :], res_join_21, mask=mask[:, None])
        else:
            hin_hpid = pid - coeff_progs
            hin_hnprog = tl.num_programs(0) - coeff_progs
            hin_pp = tl.arange(0, triton.next_power_of_2(k_parts))
            hin_nn = tl.arange(0, block_n)
            hin_rr = tl.arange(0, 4)
            hin_cc = tl.arange(0, block_c)
            hin_a0 = tl.load(alpha_ptr)
            hin_b0 = tl.load(bias_ptr)
            hin_b1 = tl.load(bias_ptr + 1)
            hin_b2 = tl.load(bias_ptr + 2)
            hin_b3 = tl.load(bias_ptr + 3)
            hin_ct: tl.constexpr = hidden // block_c
            for hin_row_group in tl.range(hin_hpid, tl.cdiv(tokens, 4), hin_hnprog):
                hin_rows = hin_row_group * 4 + hin_rr
                hin_mask = hin_rows < tokens
                hin_vals = tl.load(
                    partial_ptr
                    + (hin_pp[:, None, None] * tokens + hin_rows[None, :, None]) * block_n
                    + hin_nn[None, None, :],
                    mask=(hin_pp[:, None, None] < k_parts) & hin_mask[None, :, None],
                    other=0.0,
                )
                hin_acc = tl.sum(hin_vals, axis=0)
                hin_sqs = tl.load(
                    sumsq_ptr + hin_pp[:, None] * tokens + hin_rows[None, :],
                    mask=(hin_pp[:, None] < k_parts) & hin_mask[None, :],
                    other=0.0,
                )
                hin_sq = tl.sum(hin_sqs, axis=0)
                hin_acc *= tl.rsqrt(hin_sq / k_size + norm_eps)[:, None]
                hin_w0 = (
                    tl.sigmoid(al.extract_slice(hin_acc, (0, 0), (4, 1), (1, 1)).reshape((4,)) * hin_a0 + hin_b0)
                    + hc_eps
                )
                hin_w1 = (
                    tl.sigmoid(al.extract_slice(hin_acc, (0, 1), (4, 1), (1, 1)).reshape((4,)) * hin_a0 + hin_b1)
                    + hc_eps
                )
                hin_w2 = (
                    tl.sigmoid(al.extract_slice(hin_acc, (0, 2), (4, 1), (1, 1)).reshape((4,)) * hin_a0 + hin_b2)
                    + hc_eps
                )
                hin_w3 = (
                    tl.sigmoid(al.extract_slice(hin_acc, (0, 3), (4, 1), (1, 1)).reshape((4,)) * hin_a0 + hin_b3)
                    + hc_eps
                )
                for hin_c_tile in tl.range(0, hin_ct):
                    hin_cols = hin_c_tile * block_c + hin_cc
                    hin_base = x_ptr + hin_rows[:, None] * streams * hidden + hin_cols[None, :]
                    hin_v0 = tl.load(hin_base, mask=hin_mask[:, None], other=0.0).to(tl.float32)
                    hin_v1 = tl.load(hin_base + hidden, mask=hin_mask[:, None], other=0.0).to(tl.float32)
                    hin_v2 = tl.load(hin_base + 2 * hidden, mask=hin_mask[:, None], other=0.0).to(tl.float32)
                    hin_v3 = tl.load(hin_base + 3 * hidden, mask=hin_mask[:, None], other=0.0).to(tl.float32)
                    hin_hin = (
                        hin_v0 * hin_w0[:, None]
                        + hin_v1 * hin_w1[:, None]
                        + hin_v2 * hin_w2[:, None]
                        + hin_v3 * hin_w3[:, None]
                    )
                    tl.store(hin_ptr + hin_rows[:, None] * hidden + hin_cols[None, :], hin_hin, mask=hin_mask[:, None])
    else:
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        rr = tl.arange(0, batch_rows)
        nn = tl.arange(0, block_n)
        cols = tl.arange(0, streams)
        coffs = tl.arange(0, block_c)
        groups = tl.cdiv(row_end - row_begin, batch_rows)
        a0 = tl.load(alpha_ptr)
        a1 = tl.load(alpha_ptr + 1)
        a2 = tl.load(alpha_ptr + 2)
        tl.load(bias_ptr + cols)
        tl.load(bias_ptr + streams + cols)
        bias_0 = tl.load(bias_ptr + 0)
        bias_1 = tl.load(bias_ptr + 1)
        bias_2 = tl.load(bias_ptr + 2)
        bias_3 = tl.load(bias_ptr + 3)
        bias_4 = tl.load(bias_ptr + 4)
        bias_5 = tl.load(bias_ptr + 5)
        bias_6 = tl.load(bias_ptr + 6)
        bias_7 = tl.load(bias_ptr + 7)
        bias_8 = tl.load(bias_ptr + 8)
        bias_9 = tl.load(bias_ptr + 9)
        bias_10 = tl.load(bias_ptr + 10)
        bias_11 = tl.load(bias_ptr + 11)
        bias_12 = tl.load(bias_ptr + 12)
        bias_13 = tl.load(bias_ptr + 13)
        bias_14 = tl.load(bias_ptr + 14)
        bias_15 = tl.load(bias_ptr + 15)
        bias_16 = tl.load(bias_ptr + 16)
        bias_17 = tl.load(bias_ptr + 17)
        bias_18 = tl.load(bias_ptr + 18)
        bias_19 = tl.load(bias_ptr + 19)
        bias_20 = tl.load(bias_ptr + 20)
        bias_21 = tl.load(bias_ptr + 21)
        bias_22 = tl.load(bias_ptr + 22)
        bias_23 = tl.load(bias_ptr + 23)
        for group in tl.range(pid, groups, nprog):
            rows = row_begin + group * batch_rows + rr
            mask = rows < row_end
            pp = tl.arange(0, triton.next_power_of_2(k_parts))
            partial_values = tl.load(
                partial_ptr + (pp[:, None, None] * tokens + rows[None, :, None]) * block_n + nn[None, None, :],
                mask=(pp[:, None, None] < k_parts) & mask[None, :, None],
                other=0.0,
            )
            acc = tl.sum(partial_values, axis=0)
            sq_values = tl.load(
                sumsq_ptr + pp[:, None] * tokens + rows[None, :],
                mask=(pp[:, None] < k_parts) & mask[None, :],
                other=0.0,
            )
            sq = tl.sum(sq_values, axis=0)
            acc *= tl.rsqrt(sq / k_size + norm_eps)[:, None]
            w0 = (
                tl.sigmoid(al.extract_slice(acc, (0, 0), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_0)
                + hc_eps
            )
            p0 = 2.0 * tl.sigmoid(
                al.extract_slice(acc, (0, 4), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_4
            )
            w1 = (
                tl.sigmoid(al.extract_slice(acc, (0, 1), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_1)
                + hc_eps
            )
            p1 = 2.0 * tl.sigmoid(
                al.extract_slice(acc, (0, 5), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_5
            )
            w2 = (
                tl.sigmoid(al.extract_slice(acc, (0, 2), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_2)
                + hc_eps
            )
            p2 = 2.0 * tl.sigmoid(
                al.extract_slice(acc, (0, 6), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_6
            )
            w3 = (
                tl.sigmoid(al.extract_slice(acc, (0, 3), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a0 + bias_3)
                + hc_eps
            )
            p3 = 2.0 * tl.sigmoid(
                al.extract_slice(acc, (0, 7), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a1 + bias_7
            )
            v00 = al.extract_slice(acc, (0, 8), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_8
            v01 = al.extract_slice(acc, (0, 9), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_9
            v02 = al.extract_slice(acc, (0, 10), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_10
            v03 = al.extract_slice(acc, (0, 11), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_11
            m0 = tl.maximum(tl.maximum(v00, v01), tl.maximum(v02, v03))
            v00 = tl.exp(v00 - m0)
            v01 = tl.exp(v01 - m0)
            v02 = tl.exp(v02 - m0)
            v03 = tl.exp(v03 - m0)
            d0 = v00 + v01 + v02 + v03
            v00 = v00 / d0 + hc_eps
            v01 = v01 / d0 + hc_eps
            v02 = v02 / d0 + hc_eps
            v03 = v03 / d0 + hc_eps
            v10 = al.extract_slice(acc, (0, 12), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_12
            v11 = al.extract_slice(acc, (0, 13), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_13
            v12 = al.extract_slice(acc, (0, 14), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_14
            v13 = al.extract_slice(acc, (0, 15), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_15
            m1 = tl.maximum(tl.maximum(v10, v11), tl.maximum(v12, v13))
            v10 = tl.exp(v10 - m1)
            v11 = tl.exp(v11 - m1)
            v12 = tl.exp(v12 - m1)
            v13 = tl.exp(v13 - m1)
            d1 = v10 + v11 + v12 + v13
            v10 = v10 / d1 + hc_eps
            v11 = v11 / d1 + hc_eps
            v12 = v12 / d1 + hc_eps
            v13 = v13 / d1 + hc_eps
            v20 = al.extract_slice(acc, (0, 16), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_16
            v21 = al.extract_slice(acc, (0, 17), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_17
            v22 = al.extract_slice(acc, (0, 18), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_18
            v23 = al.extract_slice(acc, (0, 19), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_19
            m2 = tl.maximum(tl.maximum(v20, v21), tl.maximum(v22, v23))
            v20 = tl.exp(v20 - m2)
            v21 = tl.exp(v21 - m2)
            v22 = tl.exp(v22 - m2)
            v23 = tl.exp(v23 - m2)
            d2 = v20 + v21 + v22 + v23
            v20 = v20 / d2 + hc_eps
            v21 = v21 / d2 + hc_eps
            v22 = v22 / d2 + hc_eps
            v23 = v23 / d2 + hc_eps
            v30 = al.extract_slice(acc, (0, 20), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_20
            v31 = al.extract_slice(acc, (0, 21), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_21
            v32 = al.extract_slice(acc, (0, 22), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_22
            v33 = al.extract_slice(acc, (0, 23), (batch_rows, 1), (1, 1)).reshape((batch_rows,)) * a2 + bias_23
            m3 = tl.maximum(tl.maximum(v30, v31), tl.maximum(v32, v33))
            v30 = tl.exp(v30 - m3)
            v31 = tl.exp(v31 - m3)
            v32 = tl.exp(v32 - m3)
            v33 = tl.exp(v33 - m3)
            d3 = v30 + v31 + v32 + v33
            v30 = v30 / d3 + hc_eps
            v31 = v31 / d3 + hc_eps
            v32 = v32 / d3 + hc_eps
            v33 = v33 / d3 + hc_eps
            d0 = v00 + v10 + v20 + v30 + hc_eps
            v00 = v00 / d0
            v10 = v10 / d0
            v20 = v20 / d0
            v30 = v30 / d0
            d1 = v01 + v11 + v21 + v31 + hc_eps
            v01 = v01 / d1
            v11 = v11 / d1
            v21 = v21 / d1
            v31 = v31 / d1
            d2 = v02 + v12 + v22 + v32 + hc_eps
            v02 = v02 / d2
            v12 = v12 / d2
            v22 = v22 / d2
            v32 = v32 / d2
            d3 = v03 + v13 + v23 + v33 + hc_eps
            v03 = v03 / d3
            v13 = v13 / d3
            v23 = v23 / d3
            v33 = v33 / d3
            for _step in tl.range(19):
                d0 = v00 + v01 + v02 + v03 + hc_eps
                v00 = v00 / d0
                v01 = v01 / d0
                v02 = v02 / d0
                v03 = v03 / d0
                d1 = v10 + v11 + v12 + v13 + hc_eps
                v10 = v10 / d1
                v11 = v11 / d1
                v12 = v12 / d1
                v13 = v13 / d1
                d2 = v20 + v21 + v22 + v23 + hc_eps
                v20 = v20 / d2
                v21 = v21 / d2
                v22 = v22 / d2
                v23 = v23 / d2
                d3 = v30 + v31 + v32 + v33 + hc_eps
                v30 = v30 / d3
                v31 = v31 / d3
                v32 = v32 / d3
                v33 = v33 / d3
                d0 = v00 + v10 + v20 + v30 + hc_eps
                v00 = v00 / d0
                v10 = v10 / d0
                v20 = v20 / d0
                v30 = v30 / d0
                d1 = v01 + v11 + v21 + v31 + hc_eps
                v01 = v01 / d1
                v11 = v11 / d1
                v21 = v21 / d1
                v31 = v31 / d1
                d2 = v02 + v12 + v22 + v32 + hc_eps
                v02 = v02 / d2
                v12 = v12 / d2
                v22 = v22 / d2
                v32 = v32 / d2
                d3 = v03 + v13 + v23 + v33 + hc_eps
                v03 = v03 / d3
                v13 = v13 / d3
                v23 = v23 / d3
                v33 = v33 / d3
            pre_join_1 = tl.join(w0, w2).reshape((batch_rows, 2))
            pre_join_2 = tl.join(w1, w3).reshape((batch_rows, 2))
            pre_join_3 = tl.join(pre_join_1, pre_join_2).reshape((batch_rows, 4))
            post_join_4 = tl.join(p0, p2).reshape((batch_rows, 2))
            post_join_5 = tl.join(p1, p3).reshape((batch_rows, 2))
            post_join_6 = tl.join(post_join_4, post_join_5).reshape((batch_rows, 4))
            res_join_7 = tl.join(v00, v20).reshape((batch_rows, 2))
            res_join_8 = tl.join(v10, v30).reshape((batch_rows, 2))
            res_join_9 = tl.join(res_join_7, res_join_8).reshape((batch_rows, 4))
            res_join_10 = tl.join(v02, v22).reshape((batch_rows, 2))
            res_join_11 = tl.join(v12, v32).reshape((batch_rows, 2))
            res_join_12 = tl.join(res_join_10, res_join_11).reshape((batch_rows, 4))
            res_join_13 = tl.join(res_join_9, res_join_12).reshape((batch_rows, 8))
            res_join_14 = tl.join(v01, v21).reshape((batch_rows, 2))
            res_join_15 = tl.join(v11, v31).reshape((batch_rows, 2))
            res_join_16 = tl.join(res_join_14, res_join_15).reshape((batch_rows, 4))
            res_join_17 = tl.join(v03, v23).reshape((batch_rows, 2))
            res_join_18 = tl.join(v13, v33).reshape((batch_rows, 2))
            res_join_19 = tl.join(res_join_17, res_join_18).reshape((batch_rows, 4))
            res_join_20 = tl.join(res_join_16, res_join_19).reshape((batch_rows, 8))
            res_join_21 = tl.join(res_join_13, res_join_20).reshape((batch_rows, 16))
            tl.store(pre_ptr + rows[:, None] * streams + cols[None, :], pre_join_3, mask=mask[:, None])
            tl.store(post_ptr + rows[:, None] * streams + cols[None, :], post_join_6, mask=mask[:, None])
            tl.store(res_ptr + rows[:, None] * 16 + tl.arange(0, 16)[None, :], res_join_21, mask=mask[:, None])
            for tile in tl.range(0, hidden // block_c):
                cc = tile * block_c + coffs
                xb = x_ptr + rows[:, None] * streams * hidden + cc[None, :]
                v0 = tl.load(xb, mask=mask[:, None], other=0.0).to(tl.float32)
                v1 = tl.load(xb + hidden, mask=mask[:, None], other=0.0).to(tl.float32)
                v2 = tl.load(xb + 2 * hidden, mask=mask[:, None], other=0.0).to(tl.float32)
                v3 = tl.load(xb + 3 * hidden, mask=mask[:, None], other=0.0).to(tl.float32)
                hin = v0 * w0[:, None] + v1 * w1[:, None] + v2 * w2[:, None] + v3 * w3[:, None]
                tl.store(hin_ptr + rows[:, None] * hidden + cc[None, :], hin, mask=mask[:, None])


@lru_cache(maxsize=None)
def _core_counts(device_index):
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties["num_aicore"]), int(properties["num_vectorcore"])


def _split_count(cube_cores, tiles, k_tiles):
    capacity = max(1, cube_cores // tiles)
    return next(parts for parts in range(min(capacity, k_tiles), 0, -1) if k_tiles % parts == 0)


def _launch(x, phi, alpha, bias):
    cube_cores, vector_cores = _core_counts(x.device.index)
    batches, seq, streams, hidden = x.shape
    tokens = batches * seq
    k_size = streams * hidden
    flat = x.view(tokens, k_size)
    block_m = 32 if tokens <= 17 else (128 if tokens == 512 else BLOCK_M)
    block_k = 1024 if tokens <= 17 else (256 if tokens == 512 else BLOCK_K)
    block_c = BLOCK_C
    gemm_tiles = triton.cdiv(tokens, block_m)
    k_parts = _split_count(cube_cores, gemm_tiles, k_size // block_k)
    if tokens == 512:
        k_parts = min(k_size // block_k, max(1, cube_cores // gemm_tiles))
    parts = 4 if tokens == 1024 else k_parts
    mixes = torch.empty((tokens, MIX_WIDTH), dtype=torch.float32, device=x.device)
    pre = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
    h_post = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
    h_res = torch.empty((tokens, streams * streams), dtype=torch.float32, device=x.device)
    hin = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if parts > 1:
        partial = torch.empty((parts, tokens, BLOCK_N), dtype=torch.float32, device=x.device)
        partial_sq = torch.empty((parts, tokens), dtype=torch.float32, device=x.device)
    # Preserve the compensated projection for the 7168-wide 1024-token path.
    use_packed = tokens == 1024 and hidden == 7168
    use_hf32 = not use_packed
    packed_phi_buf = torch.empty((2, BLOCK_N, k_size), dtype=x.dtype, device=x.device) if use_packed else phi
    store_bf16 = x.dtype == torch.bfloat16
    if use_packed:
        mhc_pack_phi_pair_kernel[vector_cores,](
            phi, packed_phi_buf, k_size=k_size, block_n=BLOCK_N, mix_width=MIX_WIDTH, pack_block=2048
        )
    used_parts = 0
    if tokens == 1024:
        wide_m = 32 if use_packed else 64
        wide_k = 1024 if use_packed else 512
        wide_parts = 4
        wide_tiles = triton.cdiv(tokens, wide_m)
        mhc_gemm_part_kernel[wide_tiles, wide_parts](
            flat,
            packed_phi_buf,
            partial,
            partial_sq,
            tokens,
            wide_parts,
            0,
            k_size=k_size,
            mix_width=MIX_WIDTH,
            block_m=wide_m,
            block_n=BLOCK_N,
            block_k=wide_k,
            dot_hf32=use_hf32,
            packed_phi=use_packed,
        )
        used_parts = wide_parts
    elif k_parts == 1:
        gemm_grid = (min(gemm_tiles, cube_cores),)
        mhc_gemm_kernel[gemm_grid](
            flat,
            phi,
            mixes,
            tokens,
            k_size=k_size,
            mix_width=MIX_WIDTH,
            block_m=block_m,
            block_n=BLOCK_N,
            block_k=block_k,
            dot_hf32=use_hf32,
            norm_eps=NORM_EPS,
        )
    else:
        mhc_gemm_part_kernel[gemm_tiles, k_parts](
            flat,
            packed_phi_buf,
            partial,
            partial_sq,
            tokens,
            k_parts,
            0,
            k_size=k_size,
            mix_width=MIX_WIDTH,
            block_m=block_m,
            block_n=BLOCK_N,
            block_k=block_k,
            dot_hf32=use_hf32,
            packed_phi=use_packed,
        )
        used_parts = k_parts
    if used_parts and tokens <= 17:
        mhc_split_epilogue_small_kernel[min(tokens, vector_cores),](
            partial,
            partial_sq,
            mixes,
            alpha,
            bias,
            h_post,
            h_res,
            pre,
            flat,
            hin,
            tokens,
            used_parts,
            0,
            tokens,
            k_size=k_size,
            mix_width=MIX_WIDTH,
            block_n=BLOCK_N,
            norm_eps=NORM_EPS,
            streams=HC_MULT,
            hc_eps=HC_EPS,
            hidden=hidden,
            block_c=block_c,
            store_bf16=store_bf16,
        )
    elif used_parts:
        mhc_split_epilogue_kernel[
            min(vector_cores, triton.cdiv(tokens, 8) + triton.cdiv(tokens, 4))
            if tokens == 128
            else min(triton.cdiv(tokens, 8), vector_cores),
        ](
            partial,
            partial_sq,
            mixes,
            alpha,
            bias,
            h_post,
            h_res,
            pre,
            flat,
            hin,
            tokens,
            used_parts,
            0,
            tokens,
            k_size=k_size,
            mix_width=MIX_WIDTH,
            block_n=BLOCK_N,
            norm_eps=NORM_EPS,
            streams=HC_MULT,
            hc_eps=HC_EPS,
            hidden=hidden,
            store_bf16=store_bf16,
        )
    else:
        mhc_epilogue_kernel[min(tokens, vector_cores),](
            mixes,
            alpha,
            bias,
            h_post,
            h_res,
            pre,
            flat,
            hin,
            tokens,
            mix_width=MIX_WIDTH,
            streams=HC_MULT,
            sinkhorn_iters=20,
            hc_eps=HC_EPS,
            hidden=hidden,
            block_c=block_c,
            store_bf16=store_bf16,
        )

    return (
        hin.view(batches, seq, hidden),
        h_post.view(batches, seq, streams),
        h_res.view(batches, seq, streams * streams),
    )


@torch.library.custom_op("mojo_npu_triton_a5::mhc_coefficients", mutates_args=(), device_types="npu")
def mhc_coefficients_fwd(
    x: torch.Tensor, phi: torch.Tensor, alpha: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.numel() == 0:
        return _mhc_coefficients_fake(x, phi, alpha, bias)
    with torch.npu.device(x.device):
        return _launch(x, phi, alpha, bias)


@mhc_coefficients_fwd.register_fake
def _mhc_coefficients_fake(x, phi, alpha, bias):
    batch, seq, streams, hidden = x.shape
    return (
        x.new_empty((batch, seq, hidden)),
        x.new_empty((batch, seq, streams), dtype=torch.float32),
        x.new_empty((batch, seq, streams * streams), dtype=torch.float32),
    )
