"""Fused four-stream manifold-constrained Hyper-Connections on Ascend A2."""

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
BLOCK_T = 64


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
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    tile_count = tl.cdiv(tokens, block_m)
    offs_n = tl.arange(0, block_n)
    n_mask = offs_n < mix_width
    for tile in tl.range(pid, tile_count, nprog):
        rows = tile * block_m + tl.arange(0, block_m)
        row_mask = rows < tokens
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        sumsq = tl.zeros((block_m,), dtype=tl.float32)
        k_tiles = k_size // block_k
        for ki in tl.range(0, k_tiles):
            offs_k = ki * block_k + tl.arange(0, block_k)
            values = tl.load(x_ptr + rows[:, None] * k_size + offs_k[None, :], mask=row_mask[:, None], other=0.0).to(
                tl.float32
            )
            sumsq += tl.sum(values * values, axis=1)
            weight = tl.load(phi_ptr + offs_n[None, :] * k_size + offs_k[:, None], mask=n_mask[None, :], other=0.0)
            acc = tl.dot(values, weight, acc)
        inv = tl.rsqrt(sumsq / k_size + norm_eps)
        acc = acc * inv[:, None]
        tl.store(mixes_ptr + rows[:, None] * mix_width + offs_n[None, :], acc, mask=row_mask[:, None] & n_mask[None, :])


@triton.jit
def mhc_gate_hin_small_kernel(
    mixes_ptr,
    alpha_ptr,
    bias_ptr,
    post_ptr,
    res_ptr,
    x_ptr,
    hin_ptr,
    tokens,
    hidden: tl.constexpr,
    block_h: tl.constexpr,
    store_bf16: tl.constexpr,
    mix_width: tl.constexpr,
    streams: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    hc_eps: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, streams)
    for row in tl.range(pid, tokens, nprog):
        base = mixes_ptr + row * mix_width
        scale_pre = tl.load(alpha_ptr)
        scale_post = tl.load(alpha_ptr + 1)
        scale_res = tl.load(alpha_ptr + 2)
        pre = tl.sigmoid(tl.load(base + cols) * scale_pre + tl.load(bias_ptr + cols)) + hc_eps
        post = 2.0 * tl.sigmoid(tl.load(base + streams + cols) * scale_post + tl.load(bias_ptr + streams + cols))
        row_idx = tl.arange(0, streams)
        col_idx = tl.arange(0, streams)
        logit_idx = 2 * streams + row_idx[:, None] * streams + col_idx[None, :]
        logits = tl.load(base + logit_idx) * scale_res + tl.load(bias_ptr + logit_idx)
        row_max = tl.max(logits, axis=1)
        expo = tl.exp(logits - row_max[:, None])
        normed = expo / tl.sum(expo, axis=1)[:, None] + hc_eps
        col_sum = tl.sum(normed, axis=0) + hc_eps
        normed = normed / col_sum[None, :]
        for _step in tl.static_range(sinkhorn_iters - 1):
            row_sum = tl.sum(normed, axis=1) + hc_eps
            normed = normed / row_sum[:, None]
            col_sum = tl.sum(normed, axis=0) + hc_eps
            normed = normed / col_sum[None, :]
        tl.store(post_ptr + row * streams + cols, post)
        tl.store(res_ptr + row * (streams * streams) + logit_idx - 2 * streams, normed)
        channels = tl.arange(0, block_h)
        hacc = tl.zeros((block_h,), dtype=tl.float32)
        for stream in tl.static_range(streams):
            values = tl.load(
                x_ptr + row * streams * hidden + stream * hidden + channels, mask=channels < hidden, other=0.0
            ).to(tl.float32)
            weight = tl.sum(tl.where(cols == stream, pre, 0.0), axis=0)
            hacc += values * weight
        if store_bf16:
            hval = hacc.to(tl.bfloat16)
        else:
            hval = hacc.to(tl.float16)
        tl.store(hin_ptr + row * hidden + channels, hval, mask=channels < hidden)


@triton.jit
def mhc_gate_kernel(
    mixes_ptr,
    alpha_ptr,
    bias_ptr,
    post_ptr,
    res_ptr,
    pre_ptr,
    tokens,
    mix_width: tl.constexpr,
    streams: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    hc_eps: tl.constexpr,
    block_t: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    tile_count = tl.cdiv(tokens, block_t)
    token_off = tl.arange(0, block_t)
    stream_off = tl.arange(0, streams)
    res_off = tl.arange(0, streams * streams)
    scale_pre = tl.load(alpha_ptr)
    scale_post = tl.load(alpha_ptr + 1)
    scale_res = tl.load(alpha_ptr + 2)
    bias_pre = tl.load(bias_ptr + stream_off)
    bias_post = tl.load(bias_ptr + streams + stream_off)
    bias_res = tl.load(bias_ptr + 2 * streams + res_off)
    for tile in tl.range(pid, tile_count, nprog):
        rows = tile * block_t + token_off
        row_mask = rows < tokens
        base = rows[:, None] * mix_width
        pre = (
            tl.sigmoid(
                tl.load(mixes_ptr + base + stream_off[None, :], mask=row_mask[:, None], other=0.0) * scale_pre
                + bias_pre[None, :]
            )
            + hc_eps
        )
        post = 2.0 * tl.sigmoid(
            tl.load(mixes_ptr + base + streams + stream_off[None, :], mask=row_mask[:, None], other=0.0) * scale_post
            + bias_post[None, :]
        )
        logits = tl.load(mixes_ptr + base + 2 * streams + res_off[None, :], mask=row_mask[:, None], other=0.0)
        logits = (logits * scale_res + bias_res[None, :]).reshape(block_t, streams, streams)
        row_max = tl.max(logits, axis=2)
        expo = tl.exp(logits - row_max[:, :, None])
        normed = expo / tl.sum(expo, axis=2)[:, :, None] + hc_eps
        col_sum = tl.sum(normed, axis=1) + hc_eps
        normed = normed / col_sum[:, None, :]
        for _step in tl.static_range(sinkhorn_iters - 1):
            row_sum = tl.sum(normed, axis=2) + hc_eps
            normed = normed / row_sum[:, :, None]
            col_sum = tl.sum(normed, axis=1) + hc_eps
            normed = normed / col_sum[:, None, :]
        tl.store(pre_ptr + rows[:, None] * streams + stream_off[None, :], pre, mask=row_mask[:, None])
        tl.store(post_ptr + rows[:, None] * streams + stream_off[None, :], post, mask=row_mask[:, None])
        tl.store(
            res_ptr + rows[:, None] * (streams * streams) + res_off[None, :],
            normed.reshape(block_t, streams * streams),
            mask=row_mask[:, None],
        )


@triton.jit
def mhc_hin_kernel(
    x_ptr,
    pre_ptr,
    hin_ptr,
    tokens,
    hidden: tl.constexpr,
    streams: tl.constexpr,
    block_c: tl.constexpr,
    store_bf16: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    c_tiles = hidden // block_c
    total = tokens * c_tiles
    offs = tl.arange(0, block_c)
    for work in tl.range(pid, total, nprog):
        row = work // c_tiles
        tile = work - row * c_tiles
        cols = tile * block_c + offs
        acc = tl.zeros((block_c,), dtype=tl.float32)
        for stream in tl.static_range(streams):
            values = tl.load(x_ptr + row * streams * hidden + stream * hidden + cols).to(tl.float32)
            weight = tl.load(pre_ptr + row * streams + stream)
            acc += values * weight
        if store_bf16:
            stored = acc.to(tl.bfloat16)
        else:
            stored = acc.to(tl.float16)
        tl.store(hin_ptr + row * hidden + cols, stored)


@triton.jit
def split_pack_phi(phi_ptr, packed_ptr, k_size: tl.constexpr, mix_width: tl.constexpr, block_k: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, 32)
    valid = lanes < mix_width
    for tile in tl.range(pid, k_size // block_k, nprog):
        for part in tl.static_range(block_k // 256):
            k_idx = tile * block_k + part * 256 + tl.arange(0, 256)
            value = tl.load(phi_ptr + lanes[:, None] * k_size + k_idx[None, :], mask=valid[:, None], other=0.0)
            hi = value.to(tl.bfloat16)
            lo = (value - hi.to(tl.float32)).to(tl.bfloat16)
            tl.store(packed_ptr + lanes[:, None] * k_size + k_idx[None, :], hi)
            tl.store(packed_ptr + (32 + lanes[:, None]) * k_size + k_idx[None, :], lo)


@triton.jit
def fused_logits_gram_tile(
    x_ptr,
    packed_ptr,
    post_ptr,
    res_ptr,
    hin_ptr,
    tokens,
    tile,
    a0,
    a1,
    a2,
    b0,
    b1,
    b2,
    b3,
    bpost,
    bres,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
    norm_eps: tl.constexpr,
):
    rows_off = tl.arange(0, block_m)
    local_k = tl.arange(0, block_k)
    offs_n = tl.arange(0, 64)
    store_n = tl.arange(0, 32)
    c_off = tl.arange(0, block_c)
    lane4 = tl.arange(0, 4)
    eps = 1e-06
    rows = tile * block_m + rows_off
    acc = tl.zeros((block_m, 64), dtype=tl.float32)
    gram = tl.zeros((block_m, block_m), dtype=tl.float32)
    for ki in tl.range(0, k_size // block_k):
        offs_k = ki * block_k + local_k
        loaded = tl.load(x_ptr + rows[:, None] * k_size + offs_k[None, :])
        weight = tl.load(packed_ptr + offs_n[:, None] * k_size + offs_k[None, :])
        acc = tl.dot(loaded, tl.trans(weight), acc)
        gram = tl.dot(loaded, tl.trans(loaded), gram)
    sumsq = tl.sum(tl.where(rows_off[:, None] == rows_off[None, :], gram, 0.0), axis=1)
    inv = tl.rsqrt(sumsq / k_size + norm_eps)
    hi = al.extract_slice(acc, (0, 0), (block_m, 32), (1, 1))
    lo = al.extract_slice(acc, (0, 32), (block_m, 32), (1, 1))
    mix = (hi + lo) * inv[:, None]
    p0 = tl.sigmoid(tl.sum(mix * (store_n == 0), axis=1) * a0 + b0) + eps
    p1 = tl.sigmoid(tl.sum(mix * (store_n == 1), axis=1) * a0 + b1) + eps
    p2 = tl.sigmoid(tl.sum(mix * (store_n == 2), axis=1) * a0 + b2) + eps
    p3 = tl.sigmoid(tl.sum(mix * (store_n == 3), axis=1) * a0 + b3) + eps
    post = 2.0 * tl.sigmoid(al.extract_slice(mix, (0, 4), (block_m, 4), (1, 1)) * a1 + bpost[None, :])
    tl.store(post_ptr + rows[:, None] * 4 + lane4[None, :], post)
    logits = al.extract_slice(mix, (0, 8), (block_m, 16), (1, 1)) * a2 + bres[None, :]
    for ct in tl.range(0, hidden // block_c):
        cols = ct * block_c + c_off
        base = x_ptr + rows[:, None] * k_size + cols[None, :]
        h = tl.load(base).to(tl.float32) * p0[:, None]
        h += tl.load(base + hidden).to(tl.float32) * p1[:, None]
        h += tl.load(base + 2 * hidden).to(tl.float32) * p2[:, None]
        h += tl.load(base + 3 * hidden).to(tl.float32) * p3[:, None]
        tl.store(hin_ptr + rows[:, None] * hidden + cols[None, :], h.to(tl.bfloat16))
    tl.store(res_ptr + rows[:, None] * 16 + tl.arange(0, 16)[None, :], logits)


@triton.jit
def fused_logits_gram_balanced(
    x_ptr,
    packed_ptr,
    post_ptr,
    res_ptr,
    hin_ptr,
    alpha_ptr,
    bias_ptr,
    tokens,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
    norm_eps: tl.constexpr,
    regular_tiles: tl.constexpr,
    tail_tiles: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane4 = tl.arange(0, 4)
    a0 = tl.load(alpha_ptr)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    b0 = tl.load(bias_ptr)
    b1 = tl.load(bias_ptr + 1)
    b2 = tl.load(bias_ptr + 2)
    b3 = tl.load(bias_ptr + 3)
    bpost = tl.load(bias_ptr + 4 + lane4)
    bres = tl.load(bias_ptr + 8 + tl.arange(0, 16))
    for tile in tl.range(pid, regular_tiles, nprog):
        fused_logits_gram_tile(
            x_ptr,
            packed_ptr,
            post_ptr,
            res_ptr,
            hin_ptr,
            tokens,
            tile,
            a0,
            a1,
            a2,
            b0,
            b1,
            b2,
            b3,
            bpost,
            bres,
            k_size,
            hidden,
            64,
            block_k,
            block_c,
            norm_eps,
        )
    if pid < tail_tiles:
        fused_logits_gram_tile(
            x_ptr,
            packed_ptr,
            post_ptr,
            res_ptr,
            hin_ptr,
            tokens,
            regular_tiles * 2 + pid,
            a0,
            a1,
            a2,
            b0,
            b1,
            b2,
            b3,
            bpost,
            bres,
            k_size,
            hidden,
            32,
            block_k,
            512,
            norm_eps,
        )


@triton.jit
def fused_logits(
    x_ptr,
    packed_ptr,
    post_ptr,
    res_ptr,
    hin_ptr,
    alpha_ptr,
    bias_ptr,
    tokens,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
    norm_eps: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rows_off = tl.arange(0, block_m)
    local_k = tl.arange(0, block_k)
    offs_n = tl.arange(0, 64)
    store_n = tl.arange(0, 32)
    c_off = tl.arange(0, block_c)
    lane4 = tl.arange(0, 4)
    a0 = tl.load(alpha_ptr)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    b0 = tl.load(bias_ptr)
    b1 = tl.load(bias_ptr + 1)
    b2 = tl.load(bias_ptr + 2)
    b3 = tl.load(bias_ptr + 3)
    bpost = tl.load(bias_ptr + 4 + lane4)
    bres = tl.load(bias_ptr + 8 + tl.arange(0, 16))
    eps = 1e-06
    for tile in tl.range(pid, tokens // block_m, nprog):
        rows = tile * block_m + rows_off
        acc = tl.zeros((block_m, 64), dtype=tl.float32)
        gram = tl.zeros((block_m, block_m), dtype=tl.float32)
        for ki in tl.range(0, k_size // block_k):
            offs_k = ki * block_k + local_k
            loaded = tl.load(x_ptr + rows[:, None] * k_size + offs_k[None, :])
            weight = tl.load(packed_ptr + offs_n[:, None] * k_size + offs_k[None, :])
            acc = tl.dot(loaded, tl.trans(weight), acc)
            gram = tl.dot(loaded, tl.trans(loaded), gram)
        sumsq = tl.sum(tl.where(rows_off[:, None] == rows_off[None, :], gram, 0.0), axis=1)
        inv = tl.rsqrt(sumsq / k_size + norm_eps)
        hi = al.extract_slice(acc, (0, 0), (block_m, 32), (1, 1))
        lo = al.extract_slice(acc, (0, 32), (block_m, 32), (1, 1))
        mix = (hi + lo) * inv[:, None]
        p0 = tl.sigmoid(tl.sum(mix * (store_n == 0), axis=1) * a0 + b0) + eps
        p1 = tl.sigmoid(tl.sum(mix * (store_n == 1), axis=1) * a0 + b1) + eps
        p2 = tl.sigmoid(tl.sum(mix * (store_n == 2), axis=1) * a0 + b2) + eps
        p3 = tl.sigmoid(tl.sum(mix * (store_n == 3), axis=1) * a0 + b3) + eps
        post = 2.0 * tl.sigmoid(al.extract_slice(mix, (0, 4), (block_m, 4), (1, 1)) * a1 + bpost[None, :])
        tl.store(post_ptr + rows[:, None] * 4 + lane4[None, :], post)
        logits = al.extract_slice(mix, (0, 8), (block_m, 16), (1, 1)) * a2 + bres[None, :]
        for ct in tl.range(0, hidden // block_c):
            cols = ct * block_c + c_off
            base = x_ptr + rows[:, None] * k_size + cols[None, :]
            h = tl.load(base).to(tl.float32) * p0[:, None]
            h += tl.load(base + hidden).to(tl.float32) * p1[:, None]
            h += tl.load(base + 2 * hidden).to(tl.float32) * p2[:, None]
            h += tl.load(base + 3 * hidden).to(tl.float32) * p3[:, None]
            tl.store(hin_ptr + rows[:, None] * hidden + cols[None, :], h.to(tl.bfloat16))
        tl.store(res_ptr + rows[:, None] * 16 + tl.arange(0, 16)[None, :], logits)


@triton.jit
def sinkhorn20(src, dst, tokens, block: tl.constexpr):
    offsets = tl.program_id(0) * block * 16 + tl.arange(0, block * 16)
    packed = tl.load(src + offsets, mask=offsets < tokens * 16, other=0.0)
    q0a, q0b = tl.split(tl.reshape(packed, (block * 8, 2), can_reorder=False))
    q1a, q1b = tl.split(tl.reshape(q0a, (block * 4, 2), can_reorder=False))
    q2a, q2b = tl.split(tl.reshape(q1a, (block * 2, 2), can_reorder=False))
    q3a, q3b = tl.split(tl.reshape(q2a, (block * 1, 2), can_reorder=False))
    r0c0 = q3a
    r2c0 = q3b
    q4a, q4b = tl.split(tl.reshape(q2b, (block * 1, 2), can_reorder=False))
    r1c0 = q4a
    r3c0 = q4b
    q5a, q5b = tl.split(tl.reshape(q1b, (block * 2, 2), can_reorder=False))
    q6a, q6b = tl.split(tl.reshape(q5a, (block * 1, 2), can_reorder=False))
    r0c2 = q6a
    r2c2 = q6b
    q7a, q7b = tl.split(tl.reshape(q5b, (block * 1, 2), can_reorder=False))
    r1c2 = q7a
    r3c2 = q7b
    q8a, q8b = tl.split(tl.reshape(q0b, (block * 4, 2), can_reorder=False))
    q9a, q9b = tl.split(tl.reshape(q8a, (block * 2, 2), can_reorder=False))
    q10a, q10b = tl.split(tl.reshape(q9a, (block * 1, 2), can_reorder=False))
    r0c1 = q10a
    r2c1 = q10b
    q11a, q11b = tl.split(tl.reshape(q9b, (block * 1, 2), can_reorder=False))
    r1c1 = q11a
    r3c1 = q11b
    q12a, q12b = tl.split(tl.reshape(q8b, (block * 2, 2), can_reorder=False))
    q13a, q13b = tl.split(tl.reshape(q12a, (block * 1, 2), can_reorder=False))
    r0c3 = q13a
    r2c3 = q13b
    q14a, q14b = tl.split(tl.reshape(q12b, (block * 1, 2), can_reorder=False))
    r1c3 = q14a
    r3c3 = q14b
    peak = tl.maximum(tl.maximum(r0c0, r0c1), tl.maximum(r0c2, r0c3))
    r0c0 = tl.exp(r0c0 - peak)
    r0c1 = tl.exp(r0c1 - peak)
    r0c2 = tl.exp(r0c2 - peak)
    r0c3 = tl.exp(r0c3 - peak)
    denom = r0c0 + r0c1 + r0c2 + r0c3
    r0c0 = r0c0 / denom + 1e-06
    r0c1 = r0c1 / denom + 1e-06
    r0c2 = r0c2 / denom + 1e-06
    r0c3 = r0c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r1c0, r1c1), tl.maximum(r1c2, r1c3))
    r1c0 = tl.exp(r1c0 - peak)
    r1c1 = tl.exp(r1c1 - peak)
    r1c2 = tl.exp(r1c2 - peak)
    r1c3 = tl.exp(r1c3 - peak)
    denom = r1c0 + r1c1 + r1c2 + r1c3
    r1c0 = r1c0 / denom + 1e-06
    r1c1 = r1c1 / denom + 1e-06
    r1c2 = r1c2 / denom + 1e-06
    r1c3 = r1c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r2c0, r2c1), tl.maximum(r2c2, r2c3))
    r2c0 = tl.exp(r2c0 - peak)
    r2c1 = tl.exp(r2c1 - peak)
    r2c2 = tl.exp(r2c2 - peak)
    r2c3 = tl.exp(r2c3 - peak)
    denom = r2c0 + r2c1 + r2c2 + r2c3
    r2c0 = r2c0 / denom + 1e-06
    r2c1 = r2c1 / denom + 1e-06
    r2c2 = r2c2 / denom + 1e-06
    r2c3 = r2c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r3c0, r3c1), tl.maximum(r3c2, r3c3))
    r3c0 = tl.exp(r3c0 - peak)
    r3c1 = tl.exp(r3c1 - peak)
    r3c2 = tl.exp(r3c2 - peak)
    r3c3 = tl.exp(r3c3 - peak)
    denom = r3c0 + r3c1 + r3c2 + r3c3
    r3c0 = r3c0 / denom + 1e-06
    r3c1 = r3c1 / denom + 1e-06
    r3c2 = r3c2 / denom + 1e-06
    r3c3 = r3c3 / denom + 1e-06
    denom = r0c0 + r1c0 + r2c0 + r3c0 + 1e-06
    r0c0 = r0c0 / denom
    r1c0 = r1c0 / denom
    r2c0 = r2c0 / denom
    r3c0 = r3c0 / denom
    denom = r0c1 + r1c1 + r2c1 + r3c1 + 1e-06
    r0c1 = r0c1 / denom
    r1c1 = r1c1 / denom
    r2c1 = r2c1 / denom
    r3c1 = r3c1 / denom
    denom = r0c2 + r1c2 + r2c2 + r3c2 + 1e-06
    r0c2 = r0c2 / denom
    r1c2 = r1c2 / denom
    r2c2 = r2c2 / denom
    r3c2 = r3c2 / denom
    denom = r0c3 + r1c3 + r2c3 + r3c3 + 1e-06
    r0c3 = r0c3 / denom
    r1c3 = r1c3 / denom
    r2c3 = r2c3 / denom
    r3c3 = r3c3 / denom
    for norm_step in tl.range(19):
        row_d0 = r0c0 + r0c1
        row_d1 = r1c0 + r1c1
        row_d2 = r2c0 + r2c1
        row_d3 = r3c0 + r3c1
        row_d0 = row_d0 + r0c2
        row_d1 = row_d1 + r1c2
        row_d2 = row_d2 + r2c2
        row_d3 = row_d3 + r3c2
        row_d0 = row_d0 + r0c3
        row_d1 = row_d1 + r1c3
        row_d2 = row_d2 + r2c3
        row_d3 = row_d3 + r3c3
        row_d0 = row_d0 + 1e-6
        row_d1 = row_d1 + 1e-6
        row_d2 = row_d2 + 1e-6
        row_d3 = row_d3 + 1e-6
        r0c0 = r0c0 / row_d0
        r1c0 = r1c0 / row_d1
        r2c0 = r2c0 / row_d2
        r3c0 = r3c0 / row_d3
        r0c1 = r0c1 / row_d0
        r1c1 = r1c1 / row_d1
        r2c1 = r2c1 / row_d2
        r3c1 = r3c1 / row_d3
        r0c2 = r0c2 / row_d0
        r1c2 = r1c2 / row_d1
        r2c2 = r2c2 / row_d2
        r3c2 = r3c2 / row_d3
        r0c3 = r0c3 / row_d0
        r1c3 = r1c3 / row_d1
        r2c3 = r2c3 / row_d2
        r3c3 = r3c3 / row_d3
        col_d0 = r0c0 + r1c0
        col_d1 = r0c1 + r1c1
        col_d2 = r0c2 + r1c2
        col_d3 = r0c3 + r1c3
        col_d0 = col_d0 + r2c0
        col_d1 = col_d1 + r2c1
        col_d2 = col_d2 + r2c2
        col_d3 = col_d3 + r2c3
        col_d0 = col_d0 + r3c0
        col_d1 = col_d1 + r3c1
        col_d2 = col_d2 + r3c2
        col_d3 = col_d3 + r3c3
        col_d0 = col_d0 + 1e-6
        col_d1 = col_d1 + 1e-6
        col_d2 = col_d2 + 1e-6
        col_d3 = col_d3 + 1e-6
        r0c0 = r0c0 / col_d0
        r0c1 = r0c1 / col_d1
        r0c2 = r0c2 / col_d2
        r0c3 = r0c3 / col_d3
        r1c0 = r1c0 / col_d0
        r1c1 = r1c1 / col_d1
        r1c2 = r1c2 / col_d2
        r1c3 = r1c3 / col_d3
        r2c0 = r2c0 / col_d0
        r2c1 = r2c1 / col_d1
        r2c2 = r2c2 / col_d2
        r2c3 = r2c3 / col_d3
        r3c0 = r3c0 / col_d0
        r3c1 = r3c1 / col_d1
        r3c2 = r3c2 / col_d2
        r3c3 = r3c3 / col_d3
    joined0 = tl.reshape(tl.join(r0c0, r2c0), (block * 2,), can_reorder=False)
    joined1 = tl.reshape(tl.join(r1c0, r3c0), (block * 2,), can_reorder=False)
    joined2 = tl.reshape(tl.join(joined0, joined1), (block * 4,), can_reorder=False)
    joined3 = tl.reshape(tl.join(r0c2, r2c2), (block * 2,), can_reorder=False)
    joined4 = tl.reshape(tl.join(r1c2, r3c2), (block * 2,), can_reorder=False)
    joined5 = tl.reshape(tl.join(joined3, joined4), (block * 4,), can_reorder=False)
    joined6 = tl.reshape(tl.join(joined2, joined5), (block * 8,), can_reorder=False)
    joined7 = tl.reshape(tl.join(r0c1, r2c1), (block * 2,), can_reorder=False)
    joined8 = tl.reshape(tl.join(r1c1, r3c1), (block * 2,), can_reorder=False)
    joined9 = tl.reshape(tl.join(joined7, joined8), (block * 4,), can_reorder=False)
    joined10 = tl.reshape(tl.join(r0c3, r2c3), (block * 2,), can_reorder=False)
    joined11 = tl.reshape(tl.join(r1c3, r3c3), (block * 2,), can_reorder=False)
    joined12 = tl.reshape(tl.join(joined10, joined11), (block * 4,), can_reorder=False)
    joined13 = tl.reshape(tl.join(joined9, joined12), (block * 8,), can_reorder=False)
    joined14 = tl.reshape(tl.join(joined6, joined13), (block * 16,), can_reorder=False)
    tl.store(dst + offsets, joined14, mask=offsets < tokens * 16)


@triton.jit
def split_pack_phi_flat48(phi_ptr, packed_ptr, k_size: tl.constexpr, mix_width: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offsets = tl.arange(0, block)
    for tile in tl.range(pid, mix_width * k_size // block, nprog):
        idx = tile * block + offsets
        value = tl.load(phi_ptr + idx)
        hi = value.to(tl.bfloat16)
        lo = (value - hi.to(tl.float32)).to(tl.bfloat16)
        tl.store(packed_ptr + idx, hi)
        tl.store(packed_ptr + mix_width * k_size + idx, lo)


@triton.jit
def fused_m64_gram_balanced_cv(
    x_ptr,
    packed_ptr,
    post_ptr,
    res_ptr,
    hin_ptr,
    alpha_ptr,
    bias_ptr,
    workspace,
    tokens,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
    norm_eps: tl.constexpr,
    prefetch_hin: tl.constexpr,
    external_sink: tl.constexpr,
    regular_tiles: tl.constexpr,
    tail_tiles: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    with al.scope(core_mode="cube"):
        cr = tl.arange(0, block_m)
        cn = tl.arange(0, 48)
        cg = tl.arange(0, block_m)
        for ctile in tl.range(pid, regular_tiles, nprog):
            cacc, cgram = _cvhin_producer(x_ptr, packed_ptr, ctile, k_size, block_m, block_k)
            al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            cbase = workspace + pid * (block_m * 48 + block_m * block_m)
            tl.store(cbase + cr[:, None] * 48 + cn[None, :], cacc)
            tl.store(cbase + block_m * 48 + cr[:, None] * block_m + cg[None, :], cgram)
            al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
        if pid < tail_tiles:
            tail_tile = regular_tiles * 2 + pid
            cacc, cgram = _cvhin_producer(x_ptr, packed_ptr, tail_tile, k_size, 32, block_k)
            al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            cbase = workspace + pid * (block_m * 48 + block_m * block_m)
            tr = tl.arange(0, 32)
            tg = tl.arange(0, 32)
            tl.store(cbase + tr[:, None] * 48 + cn[None, :], cacc)
            tl.store(cbase + 32 * 48 + tr[:, None] * 32 + tg[None, :], cgram)
            al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
        al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
    with al.scope(core_mode="vector"):
        vsub = al.sub_vec_id()
        vr = vsub * (block_m // 2) + tl.arange(0, block_m // 2)
        vn = tl.arange(0, 48)
        vg = tl.arange(0, block_m)
        vc = tl.arange(0, block_c)
        al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
        for vtile in tl.range(pid, regular_tiles, nprog):
            if prefetch_hin:
                vrows = vtile * block_m + vr
                vptr = x_ptr + vrows[:, None] * k_size + vc[None, :]
                ph0 = tl.load(vptr)
                ph1 = tl.load(vptr + hidden)
                ph2 = tl.load(vptr + 2 * hidden)
                ph3 = tl.load(vptr + 3 * hidden)
            else:
                ph0 = tl.full((block_m // 2, block_c), 0, tl.bfloat16)
                ph1 = ph0
                ph2 = ph0
                ph3 = ph0
            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
            vbase = workspace + pid * (block_m * 48 + block_m * block_m)
            vacc = tl.load(vbase + vr[:, None] * 48 + vn[None, :])
            vgram = tl.load(vbase + block_m * 48 + vr[:, None] * block_m + vg[None, :])
            al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            vss = tl.sum(tl.where(vr.to(tl.float32)[:, None] == vg.to(tl.float32)[None, :], vgram, 0.0), axis=1)
            _cvhin_consumer_res_trans(
                vacc,
                vss,
                ph0,
                ph1,
                ph2,
                ph3,
                x_ptr,
                post_ptr,
                res_ptr,
                hin_ptr,
                alpha_ptr,
                bias_ptr,
                vtile,
                vsub,
                k_size,
                hidden,
                block_m // 2,
                block_k,
                block_c,
                norm_eps,
                prefetch_hin,
                external_sink,
            )

        if pid < tail_tiles:
            tail_tile = regular_tiles * 2 + pid
            tr = vsub * 16 + tl.arange(0, 16)
            tg = tl.arange(0, 32)
            tail_rows = tail_tile * 32 + tr
            tail_ptr = x_ptr + tail_rows[:, None] * k_size + vc[None, :]
            ph0 = tl.load(tail_ptr)
            ph1 = tl.load(tail_ptr + hidden)
            ph2 = tl.load(tail_ptr + 2 * hidden)
            ph3 = tl.load(tail_ptr + 3 * hidden)
            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
            vbase = workspace + pid * (block_m * 48 + block_m * block_m)
            vacc = tl.load(vbase + tr[:, None] * 48 + vn[None, :])
            vgram = tl.load(vbase + 32 * 48 + tr[:, None] * 32 + tg[None, :])
            al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            vss = tl.sum(tl.where(tr.to(tl.float32)[:, None] == tg.to(tl.float32)[None, :], vgram, 0.0), axis=1)
            _cvhin_consumer_res_trans(
                vacc,
                vss,
                ph0,
                ph1,
                ph2,
                ph3,
                x_ptr,
                post_ptr,
                res_ptr,
                hin_ptr,
                alpha_ptr,
                bias_ptr,
                tail_tile,
                vsub,
                k_size,
                hidden,
                16,
                block_k,
                block_c,
                norm_eps,
                True,
                external_sink,
            )


@triton.jit
def _cvhin_producer(x_ptr, packed_ptr, tile, k_size: tl.constexpr, block_m: tl.constexpr, block_k: tl.constexpr):
    rows_off = tl.arange(0, block_m)
    local_k = tl.arange(0, block_k)
    offs_n = tl.arange(0, 48)
    rows = tile * block_m + rows_off
    acc = tl.zeros((block_m, 48), dtype=tl.float32)
    gram = tl.zeros((block_m, block_m), dtype=tl.float32)
    for ki in tl.range(0, k_size // block_k):
        offs_k = ki * block_k + local_k
        loaded = tl.load(x_ptr + rows[:, None] * k_size + offs_k[None, :])
        weight = tl.load(packed_ptr + offs_n[:, None] * k_size + offs_k[None, :])
        acc = tl.dot(loaded, tl.trans(weight), acc)
        gram = tl.dot(loaded, tl.trans(loaded), gram)
    return acc, gram


@triton.jit
def _cvhin_consumer_res_trans(
    acc,
    sumsq,
    ph0,
    ph1,
    ph2,
    ph3,
    x_ptr,
    post_ptr,
    res_ptr,
    hin_ptr,
    alpha_ptr,
    bias_ptr,
    tile,
    subvec,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
    norm_eps: tl.constexpr,
    prefetch_hin: tl.constexpr,
    external_sink: tl.constexpr,
):
    rows_off = tl.arange(0, block_m)
    tl.arange(0, block_k)
    tl.arange(0, 48)
    store_n = tl.arange(0, 32)
    c_off = tl.arange(0, block_c)
    lane4 = tl.arange(0, 4)
    a0 = tl.load(alpha_ptr)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    b0 = tl.load(bias_ptr)
    b1 = tl.load(bias_ptr + 1)
    b2 = tl.load(bias_ptr + 2)
    b3 = tl.load(bias_ptr + 3)
    bpost = tl.load(bias_ptr + 4 + lane4)
    bres = tl.load(bias_ptr + 8 + tl.arange(0, 16))
    eps = 1e-06
    rows = tile * (2 * block_m) + subvec * block_m + rows_off
    inv = tl.rsqrt(sumsq / k_size + norm_eps)
    first_hi = al.extract_slice(acc, (0, 0), (block_m, 16), (1, 1))
    first_lo = al.extract_slice(acc, (0, 24), (block_m, 16), (1, 1))
    first = first_hi + first_lo
    tail_hi = al.extract_slice(acc, (0, 16), (block_m, 8), (1, 1))
    tail_lo = al.extract_slice(acc, (0, 40), (block_m, 8), (1, 1))
    tail = tail_hi + tail_lo
    mix32 = tl.zeros((block_m, 32), dtype=tl.float32)
    mix32 = al.insert_slice(mix32, first, (0, 0), (block_m, 16), (1, 1))
    mix32 = al.insert_slice(mix32, tail, (0, 16), (block_m, 8), (1, 1))
    mix = mix32 * inv[:, None]
    p0 = tl.sigmoid(tl.sum(mix * (store_n == 0), axis=1) * a0 + b0) + eps
    p1 = tl.sigmoid(tl.sum(mix * (store_n == 1), axis=1) * a0 + b1) + eps
    p2 = tl.sigmoid(tl.sum(mix * (store_n == 2), axis=1) * a0 + b2) + eps
    p3 = tl.sigmoid(tl.sum(mix * (store_n == 3), axis=1) * a0 + b3) + eps
    post = 2.0 * tl.sigmoid(al.extract_slice(mix, (0, 4), (block_m, 4), (1, 1)) * a1 + bpost[None, :])
    tl.store(post_ptr + rows[:, None] * 4 + lane4[None, :], post)
    logits = al.extract_slice(mix, (0, 8), (block_m, 16), (1, 1)) * a2 + bres[None, :]
    if prefetch_hin:
        hp = ph0.to(tl.float32) * p0[:, None]
        hp += ph1.to(tl.float32) * p1[:, None]
        hp += ph2.to(tl.float32) * p2[:, None]
        hp += ph3.to(tl.float32) * p3[:, None]
        tl.store(hin_ptr + rows[:, None] * hidden + c_off[None, :], hp.to(tl.bfloat16))
    for ct in tl.range(1 if prefetch_hin else 0, hidden // block_c):
        cols = ct * block_c + c_off
        base = x_ptr + rows[:, None] * k_size + cols[None, :]
        h = tl.load(base).to(tl.float32) * p0[:, None]
        h += tl.load(base + hidden).to(tl.float32) * p1[:, None]
        h += tl.load(base + 2 * hidden).to(tl.float32) * p2[:, None]
        h += tl.load(base + 3 * hidden).to(tl.float32) * p3[:, None]
        tl.store(hin_ptr + rows[:, None] * hidden + cols[None, :], h.to(tl.bfloat16))
    if external_sink:
        tl.store(res_ptr + rows[:, None] * 16 + tl.arange(0, 16)[None, :], logits)
    else:
        cols16 = tl.arange(0, 16)
        logits_columns = tl.trans(logits).reshape(block_m * 16).reshape(16, block_m)
        r0c0 = al.extract_slice(logits_columns, (0, 0), (1, block_m), (1, 1)).reshape(block_m)
        r0c1 = al.extract_slice(logits_columns, (1, 0), (1, block_m), (1, 1)).reshape(block_m)
        r0c2 = al.extract_slice(logits_columns, (2, 0), (1, block_m), (1, 1)).reshape(block_m)
        r0c3 = al.extract_slice(logits_columns, (3, 0), (1, block_m), (1, 1)).reshape(block_m)
        r1c0 = al.extract_slice(logits_columns, (4, 0), (1, block_m), (1, 1)).reshape(block_m)
        r1c1 = al.extract_slice(logits_columns, (5, 0), (1, block_m), (1, 1)).reshape(block_m)
        r1c2 = al.extract_slice(logits_columns, (6, 0), (1, block_m), (1, 1)).reshape(block_m)
        r1c3 = al.extract_slice(logits_columns, (7, 0), (1, block_m), (1, 1)).reshape(block_m)
        r2c0 = al.extract_slice(logits_columns, (8, 0), (1, block_m), (1, 1)).reshape(block_m)
        r2c1 = al.extract_slice(logits_columns, (9, 0), (1, block_m), (1, 1)).reshape(block_m)
        r2c2 = al.extract_slice(logits_columns, (10, 0), (1, block_m), (1, 1)).reshape(block_m)
        r2c3 = al.extract_slice(logits_columns, (11, 0), (1, block_m), (1, 1)).reshape(block_m)
        r3c0 = al.extract_slice(logits_columns, (12, 0), (1, block_m), (1, 1)).reshape(block_m)
        r3c1 = al.extract_slice(logits_columns, (13, 0), (1, block_m), (1, 1)).reshape(block_m)
        r3c2 = al.extract_slice(logits_columns, (14, 0), (1, block_m), (1, 1)).reshape(block_m)
        r3c3 = al.extract_slice(logits_columns, (15, 0), (1, block_m), (1, 1)).reshape(block_m)
        peak = tl.maximum(tl.maximum(r0c0, r0c1), tl.maximum(r0c2, r0c3))
        r0c0 = tl.exp(r0c0 - peak)
        r0c1 = tl.exp(r0c1 - peak)
        r0c2 = tl.exp(r0c2 - peak)
        r0c3 = tl.exp(r0c3 - peak)
        denom = r0c0 + r0c1 + r0c2 + r0c3
        r0c0 = r0c0 / denom + 1e-06
        r0c1 = r0c1 / denom + 1e-06
        r0c2 = r0c2 / denom + 1e-06
        r0c3 = r0c3 / denom + 1e-06
        peak = tl.maximum(tl.maximum(r1c0, r1c1), tl.maximum(r1c2, r1c3))
        r1c0 = tl.exp(r1c0 - peak)
        r1c1 = tl.exp(r1c1 - peak)
        r1c2 = tl.exp(r1c2 - peak)
        r1c3 = tl.exp(r1c3 - peak)
        denom = r1c0 + r1c1 + r1c2 + r1c3
        r1c0 = r1c0 / denom + 1e-06
        r1c1 = r1c1 / denom + 1e-06
        r1c2 = r1c2 / denom + 1e-06
        r1c3 = r1c3 / denom + 1e-06
        peak = tl.maximum(tl.maximum(r2c0, r2c1), tl.maximum(r2c2, r2c3))
        r2c0 = tl.exp(r2c0 - peak)
        r2c1 = tl.exp(r2c1 - peak)
        r2c2 = tl.exp(r2c2 - peak)
        r2c3 = tl.exp(r2c3 - peak)
        denom = r2c0 + r2c1 + r2c2 + r2c3
        r2c0 = r2c0 / denom + 1e-06
        r2c1 = r2c1 / denom + 1e-06
        r2c2 = r2c2 / denom + 1e-06
        r2c3 = r2c3 / denom + 1e-06
        peak = tl.maximum(tl.maximum(r3c0, r3c1), tl.maximum(r3c2, r3c3))
        r3c0 = tl.exp(r3c0 - peak)
        r3c1 = tl.exp(r3c1 - peak)
        r3c2 = tl.exp(r3c2 - peak)
        r3c3 = tl.exp(r3c3 - peak)
        denom = r3c0 + r3c1 + r3c2 + r3c3
        r3c0 = r3c0 / denom + 1e-06
        r3c1 = r3c1 / denom + 1e-06
        r3c2 = r3c2 / denom + 1e-06
        r3c3 = r3c3 / denom + 1e-06
        denom = r0c0 + r1c0 + r2c0 + r3c0 + 1e-06
        r0c0 = r0c0 / denom
        r1c0 = r1c0 / denom
        r2c0 = r2c0 / denom
        r3c0 = r3c0 / denom
        denom = r0c1 + r1c1 + r2c1 + r3c1 + 1e-06
        r0c1 = r0c1 / denom
        r1c1 = r1c1 / denom
        r2c1 = r2c1 / denom
        r3c1 = r3c1 / denom
        denom = r0c2 + r1c2 + r2c2 + r3c2 + 1e-06
        r0c2 = r0c2 / denom
        r1c2 = r1c2 / denom
        r2c2 = r2c2 / denom
        r3c2 = r3c2 / denom
        denom = r0c3 + r1c3 + r2c3 + r3c3 + 1e-06
        r0c3 = r0c3 / denom
        r1c3 = r1c3 / denom
        r2c3 = r2c3 / denom
        r3c3 = r3c3 / denom
        for norm_step in tl.range(19):
            row_d0 = r0c0 + r0c1
            row_d1 = r1c0 + r1c1
            row_d2 = r2c0 + r2c1
            row_d3 = r3c0 + r3c1
            row_d0 = row_d0 + r0c2
            row_d1 = row_d1 + r1c2
            row_d2 = row_d2 + r2c2
            row_d3 = row_d3 + r3c2
            row_d0 = row_d0 + r0c3
            row_d1 = row_d1 + r1c3
            row_d2 = row_d2 + r2c3
            row_d3 = row_d3 + r3c3
            row_d0 = row_d0 + 1e-6
            row_d1 = row_d1 + 1e-6
            row_d2 = row_d2 + 1e-6
            row_d3 = row_d3 + 1e-6
            r0c0 = r0c0 / row_d0
            r1c0 = r1c0 / row_d1
            r2c0 = r2c0 / row_d2
            r3c0 = r3c0 / row_d3
            r0c1 = r0c1 / row_d0
            r1c1 = r1c1 / row_d1
            r2c1 = r2c1 / row_d2
            r3c1 = r3c1 / row_d3
            r0c2 = r0c2 / row_d0
            r1c2 = r1c2 / row_d1
            r2c2 = r2c2 / row_d2
            r3c2 = r3c2 / row_d3
            r0c3 = r0c3 / row_d0
            r1c3 = r1c3 / row_d1
            r2c3 = r2c3 / row_d2
            r3c3 = r3c3 / row_d3
            col_d0 = r0c0 + r1c0
            col_d1 = r0c1 + r1c1
            col_d2 = r0c2 + r1c2
            col_d3 = r0c3 + r1c3
            col_d0 = col_d0 + r2c0
            col_d1 = col_d1 + r2c1
            col_d2 = col_d2 + r2c2
            col_d3 = col_d3 + r2c3
            col_d0 = col_d0 + r3c0
            col_d1 = col_d1 + r3c1
            col_d2 = col_d2 + r3c2
            col_d3 = col_d3 + r3c3
            col_d0 = col_d0 + 1e-6
            col_d1 = col_d1 + 1e-6
            col_d2 = col_d2 + 1e-6
            col_d3 = col_d3 + 1e-6
            r0c0 = r0c0 / col_d0
            r0c1 = r0c1 / col_d1
            r0c2 = r0c2 / col_d2
            r0c3 = r0c3 / col_d3
            r1c0 = r1c0 / col_d0
            r1c1 = r1c1 / col_d1
            r1c2 = r1c2 / col_d2
            r1c3 = r1c3 / col_d3
            r2c0 = r2c0 / col_d0
            r2c1 = r2c1 / col_d1
            r2c2 = r2c2 / col_d2
            r2c3 = r2c3 / col_d3
            r3c0 = r3c0 / col_d0
            r3c1 = r3c1 / col_d1
            r3c2 = r3c2 / col_d2
            r3c3 = r3c3 / col_d3
        packed_columns = tl.zeros((16, block_m), tl.float32)
        packed_columns = al.insert_slice(packed_columns, r0c0[None, :], (0, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r0c1[None, :], (1, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r0c2[None, :], (2, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r0c3[None, :], (3, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r1c0[None, :], (4, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r1c1[None, :], (5, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r1c2[None, :], (6, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r1c3[None, :], (7, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r2c0[None, :], (8, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r2c1[None, :], (9, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r2c2[None, :], (10, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r2c3[None, :], (11, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r3c0[None, :], (12, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r3c1[None, :], (13, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r3c2[None, :], (14, 0), (1, block_m), (1, 1))
        packed_columns = al.insert_slice(packed_columns, r3c3[None, :], (15, 0), (1, block_m), (1, 1))
        tl.store(res_ptr + rows[:, None] * 16 + cols16[None, :], tl.trans(packed_columns))


@triton.jit
def _cvhin_consumer_early_sink(
    acc,
    sumsq,
    ph0,
    ph1,
    ph2,
    ph3,
    x_ptr,
    post_ptr,
    res_ptr,
    hin_ptr,
    alpha_ptr,
    bias_ptr,
    tile,
    subvec,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
    norm_eps: tl.constexpr,
    prefetch_hin: tl.constexpr,
    external_sink: tl.constexpr,
):
    rows_off = tl.arange(0, block_m)
    tl.arange(0, block_k)
    tl.arange(0, 48)
    store_n = tl.arange(0, 32)
    c_off = tl.arange(0, block_c)
    lane4 = tl.arange(0, 4)
    a0 = tl.load(alpha_ptr)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    b0 = tl.load(bias_ptr)
    b1 = tl.load(bias_ptr + 1)
    b2 = tl.load(bias_ptr + 2)
    b3 = tl.load(bias_ptr + 3)
    bpost = tl.load(bias_ptr + 4 + lane4)
    bres = tl.load(bias_ptr + 8 + tl.arange(0, 16))
    eps = 1e-06
    rows = tile * (2 * block_m) + subvec * block_m + rows_off
    inv = tl.rsqrt(sumsq / k_size + norm_eps)
    first_hi = al.extract_slice(acc, (0, 0), (block_m, 16), (1, 1))
    first_lo = al.extract_slice(acc, (0, 24), (block_m, 16), (1, 1))
    first = first_hi + first_lo
    tail_hi = al.extract_slice(acc, (0, 16), (block_m, 8), (1, 1))
    tail_lo = al.extract_slice(acc, (0, 40), (block_m, 8), (1, 1))
    tail = tail_hi + tail_lo
    mix32 = tl.zeros((block_m, 32), dtype=tl.float32)
    mix32 = al.insert_slice(mix32, first, (0, 0), (block_m, 16), (1, 1))
    mix32 = al.insert_slice(mix32, tail, (0, 16), (block_m, 8), (1, 1))
    mix = mix32 * inv[:, None]
    p0 = tl.sigmoid(tl.sum(mix * (store_n == 0), axis=1) * a0 + b0) + eps
    p1 = tl.sigmoid(tl.sum(mix * (store_n == 1), axis=1) * a0 + b1) + eps
    p2 = tl.sigmoid(tl.sum(mix * (store_n == 2), axis=1) * a0 + b2) + eps
    p3 = tl.sigmoid(tl.sum(mix * (store_n == 3), axis=1) * a0 + b3) + eps
    post = 2.0 * tl.sigmoid(al.extract_slice(mix, (0, 4), (block_m, 4), (1, 1)) * a1 + bpost[None, :])
    tl.store(post_ptr + rows[:, None] * 4 + lane4[None, :], post)
    logits = al.extract_slice(mix, (0, 8), (block_m, 16), (1, 1)) * a2 + bres[None, :]
    cols16 = tl.arange(0, 16)
    logits_columns = tl.trans(logits).reshape(block_m * 16).reshape(16, block_m)
    r0c0 = al.extract_slice(logits_columns, (0, 0), (1, block_m), (1, 1)).reshape(block_m)
    r0c1 = al.extract_slice(logits_columns, (1, 0), (1, block_m), (1, 1)).reshape(block_m)
    r0c2 = al.extract_slice(logits_columns, (2, 0), (1, block_m), (1, 1)).reshape(block_m)
    r0c3 = al.extract_slice(logits_columns, (3, 0), (1, block_m), (1, 1)).reshape(block_m)
    r1c0 = al.extract_slice(logits_columns, (4, 0), (1, block_m), (1, 1)).reshape(block_m)
    r1c1 = al.extract_slice(logits_columns, (5, 0), (1, block_m), (1, 1)).reshape(block_m)
    r1c2 = al.extract_slice(logits_columns, (6, 0), (1, block_m), (1, 1)).reshape(block_m)
    r1c3 = al.extract_slice(logits_columns, (7, 0), (1, block_m), (1, 1)).reshape(block_m)
    r2c0 = al.extract_slice(logits_columns, (8, 0), (1, block_m), (1, 1)).reshape(block_m)
    r2c1 = al.extract_slice(logits_columns, (9, 0), (1, block_m), (1, 1)).reshape(block_m)
    r2c2 = al.extract_slice(logits_columns, (10, 0), (1, block_m), (1, 1)).reshape(block_m)
    r2c3 = al.extract_slice(logits_columns, (11, 0), (1, block_m), (1, 1)).reshape(block_m)
    r3c0 = al.extract_slice(logits_columns, (12, 0), (1, block_m), (1, 1)).reshape(block_m)
    r3c1 = al.extract_slice(logits_columns, (13, 0), (1, block_m), (1, 1)).reshape(block_m)
    r3c2 = al.extract_slice(logits_columns, (14, 0), (1, block_m), (1, 1)).reshape(block_m)
    r3c3 = al.extract_slice(logits_columns, (15, 0), (1, block_m), (1, 1)).reshape(block_m)
    peak = tl.maximum(tl.maximum(r0c0, r0c1), tl.maximum(r0c2, r0c3))
    r0c0 = tl.exp(r0c0 - peak)
    r0c1 = tl.exp(r0c1 - peak)
    r0c2 = tl.exp(r0c2 - peak)
    r0c3 = tl.exp(r0c3 - peak)
    denom = r0c0 + r0c1 + r0c2 + r0c3
    r0c0 = r0c0 / denom + 1e-06
    r0c1 = r0c1 / denom + 1e-06
    r0c2 = r0c2 / denom + 1e-06
    r0c3 = r0c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r1c0, r1c1), tl.maximum(r1c2, r1c3))
    r1c0 = tl.exp(r1c0 - peak)
    r1c1 = tl.exp(r1c1 - peak)
    r1c2 = tl.exp(r1c2 - peak)
    r1c3 = tl.exp(r1c3 - peak)
    denom = r1c0 + r1c1 + r1c2 + r1c3
    r1c0 = r1c0 / denom + 1e-06
    r1c1 = r1c1 / denom + 1e-06
    r1c2 = r1c2 / denom + 1e-06
    r1c3 = r1c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r2c0, r2c1), tl.maximum(r2c2, r2c3))
    r2c0 = tl.exp(r2c0 - peak)
    r2c1 = tl.exp(r2c1 - peak)
    r2c2 = tl.exp(r2c2 - peak)
    r2c3 = tl.exp(r2c3 - peak)
    denom = r2c0 + r2c1 + r2c2 + r2c3
    r2c0 = r2c0 / denom + 1e-06
    r2c1 = r2c1 / denom + 1e-06
    r2c2 = r2c2 / denom + 1e-06
    r2c3 = r2c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r3c0, r3c1), tl.maximum(r3c2, r3c3))
    r3c0 = tl.exp(r3c0 - peak)
    r3c1 = tl.exp(r3c1 - peak)
    r3c2 = tl.exp(r3c2 - peak)
    r3c3 = tl.exp(r3c3 - peak)
    denom = r3c0 + r3c1 + r3c2 + r3c3
    r3c0 = r3c0 / denom + 1e-06
    r3c1 = r3c1 / denom + 1e-06
    r3c2 = r3c2 / denom + 1e-06
    r3c3 = r3c3 / denom + 1e-06
    denom = r0c0 + r1c0 + r2c0 + r3c0 + 1e-06
    r0c0 = r0c0 / denom
    r1c0 = r1c0 / denom
    r2c0 = r2c0 / denom
    r3c0 = r3c0 / denom
    denom = r0c1 + r1c1 + r2c1 + r3c1 + 1e-06
    r0c1 = r0c1 / denom
    r1c1 = r1c1 / denom
    r2c1 = r2c1 / denom
    r3c1 = r3c1 / denom
    denom = r0c2 + r1c2 + r2c2 + r3c2 + 1e-06
    r0c2 = r0c2 / denom
    r1c2 = r1c2 / denom
    r2c2 = r2c2 / denom
    r3c2 = r3c2 / denom
    denom = r0c3 + r1c3 + r2c3 + r3c3 + 1e-06
    r0c3 = r0c3 / denom
    r1c3 = r1c3 / denom
    r2c3 = r2c3 / denom
    r3c3 = r3c3 / denom
    for norm_step in tl.range(19):
        row_d0 = r0c0 + r0c1
        row_d1 = r1c0 + r1c1
        row_d2 = r2c0 + r2c1
        row_d3 = r3c0 + r3c1
        row_d0 = row_d0 + r0c2
        row_d1 = row_d1 + r1c2
        row_d2 = row_d2 + r2c2
        row_d3 = row_d3 + r3c2
        row_d0 = row_d0 + r0c3
        row_d1 = row_d1 + r1c3
        row_d2 = row_d2 + r2c3
        row_d3 = row_d3 + r3c3
        row_d0 = row_d0 + 1e-6
        row_d1 = row_d1 + 1e-6
        row_d2 = row_d2 + 1e-6
        row_d3 = row_d3 + 1e-6
        r0c0 = r0c0 / row_d0
        r1c0 = r1c0 / row_d1
        r2c0 = r2c0 / row_d2
        r3c0 = r3c0 / row_d3
        r0c1 = r0c1 / row_d0
        r1c1 = r1c1 / row_d1
        r2c1 = r2c1 / row_d2
        r3c1 = r3c1 / row_d3
        r0c2 = r0c2 / row_d0
        r1c2 = r1c2 / row_d1
        r2c2 = r2c2 / row_d2
        r3c2 = r3c2 / row_d3
        r0c3 = r0c3 / row_d0
        r1c3 = r1c3 / row_d1
        r2c3 = r2c3 / row_d2
        r3c3 = r3c3 / row_d3
        col_d0 = r0c0 + r1c0
        col_d1 = r0c1 + r1c1
        col_d2 = r0c2 + r1c2
        col_d3 = r0c3 + r1c3
        col_d0 = col_d0 + r2c0
        col_d1 = col_d1 + r2c1
        col_d2 = col_d2 + r2c2
        col_d3 = col_d3 + r2c3
        col_d0 = col_d0 + r3c0
        col_d1 = col_d1 + r3c1
        col_d2 = col_d2 + r3c2
        col_d3 = col_d3 + r3c3
        col_d0 = col_d0 + 1e-6
        col_d1 = col_d1 + 1e-6
        col_d2 = col_d2 + 1e-6
        col_d3 = col_d3 + 1e-6
        r0c0 = r0c0 / col_d0
        r0c1 = r0c1 / col_d1
        r0c2 = r0c2 / col_d2
        r0c3 = r0c3 / col_d3
        r1c0 = r1c0 / col_d0
        r1c1 = r1c1 / col_d1
        r1c2 = r1c2 / col_d2
        r1c3 = r1c3 / col_d3
        r2c0 = r2c0 / col_d0
        r2c1 = r2c1 / col_d1
        r2c2 = r2c2 / col_d2
        r2c3 = r2c3 / col_d3
        r3c0 = r3c0 / col_d0
        r3c1 = r3c1 / col_d1
        r3c2 = r3c2 / col_d2
        r3c3 = r3c3 / col_d3
    packed_columns = tl.zeros((16, block_m), tl.float32)
    packed_columns = al.insert_slice(packed_columns, r0c0[None, :], (0, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r0c1[None, :], (1, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r0c2[None, :], (2, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r0c3[None, :], (3, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r1c0[None, :], (4, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r1c1[None, :], (5, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r1c2[None, :], (6, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r1c3[None, :], (7, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r2c0[None, :], (8, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r2c1[None, :], (9, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r2c2[None, :], (10, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r2c3[None, :], (11, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r3c0[None, :], (12, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r3c1[None, :], (13, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r3c2[None, :], (14, 0), (1, block_m), (1, 1))
    packed_columns = al.insert_slice(packed_columns, r3c3[None, :], (15, 0), (1, block_m), (1, 1))
    tl.store(res_ptr + rows[:, None] * 16 + cols16[None, :], tl.trans(packed_columns))
    if prefetch_hin:
        hp = ph0.to(tl.float32) * p0[:, None]
        hp += ph1.to(tl.float32) * p1[:, None]
        hp += ph2.to(tl.float32) * p2[:, None]
        hp += ph3.to(tl.float32) * p3[:, None]
        tl.store(hin_ptr + rows[:, None] * hidden + c_off[None, :], hp.to(tl.bfloat16))
    for ct in tl.range(1 if prefetch_hin else 0, hidden // block_c):
        cols = ct * block_c + c_off
        base = x_ptr + rows[:, None] * k_size + cols[None, :]
        h = tl.load(base).to(tl.float32) * p0[:, None]
        h += tl.load(base + hidden).to(tl.float32) * p1[:, None]
        h += tl.load(base + 2 * hidden).to(tl.float32) * p2[:, None]
        h += tl.load(base + 3 * hidden).to(tl.float32) * p3[:, None]
        tl.store(hin_ptr + rows[:, None] * hidden + cols[None, :], h.to(tl.bfloat16))


@triton.jit
def fused_logits_n48_cvhin_res_trans(
    x_ptr,
    packed_ptr,
    post_ptr,
    res_ptr,
    hin_ptr,
    alpha_ptr,
    bias_ptr,
    workspace,
    tokens,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
    norm_eps: tl.constexpr,
    prefetch_hin: tl.constexpr,
    external_sink: tl.constexpr,
    early_sink: tl.constexpr = False,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    with al.scope(core_mode="cube"):
        cr = tl.arange(0, block_m)
        cn = tl.arange(0, 48)
        cg = tl.arange(0, block_m)
        for ctile in tl.range(pid, tokens // block_m, nprog):
            cacc, cgram = _cvhin_producer(x_ptr, packed_ptr, ctile, k_size, block_m, block_k)
            al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            cbase = workspace + pid * (block_m * 48 + block_m * block_m)
            tl.store(cbase + cr[:, None] * 48 + cn[None, :], cacc)
            tl.store(cbase + block_m * 48 + cr[:, None] * block_m + cg[None, :], cgram)
            al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
        al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
    with al.scope(core_mode="vector"):
        vsub = al.sub_vec_id()
        vr = vsub * (block_m // 2) + tl.arange(0, block_m // 2)
        vn = tl.arange(0, 48)
        vg = tl.arange(0, block_m)
        vc = tl.arange(0, block_c)
        al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
        for vtile in tl.range(pid, tokens // block_m, nprog):
            if prefetch_hin:
                vrows = vtile * block_m + vr
                vptr = x_ptr + vrows[:, None] * k_size + vc[None, :]
                ph0 = tl.load(vptr)
                ph1 = tl.load(vptr + hidden)
                ph2 = tl.load(vptr + 2 * hidden)
                ph3 = tl.load(vptr + 3 * hidden)
            else:
                ph0 = tl.full((block_m // 2, block_c), 0, tl.bfloat16)
                ph1 = ph0
                ph2 = ph0
                ph3 = ph0
            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
            vbase = workspace + pid * (block_m * 48 + block_m * block_m)
            vacc = tl.load(vbase + vr[:, None] * 48 + vn[None, :])
            vgram = tl.load(vbase + block_m * 48 + vr[:, None] * block_m + vg[None, :])
            al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            vss = tl.sum(tl.where(vr.to(tl.float32)[:, None] == vg.to(tl.float32)[None, :], vgram, 0.0), axis=1)
            if early_sink:
                _cvhin_consumer_early_sink(
                    vacc,
                    vss,
                    ph0,
                    ph1,
                    ph2,
                    ph3,
                    x_ptr,
                    post_ptr,
                    res_ptr,
                    hin_ptr,
                    alpha_ptr,
                    bias_ptr,
                    vtile,
                    vsub,
                    k_size,
                    hidden,
                    block_m // 2,
                    block_k,
                    block_c,
                    norm_eps,
                    prefetch_hin,
                    external_sink,
                )
            else:
                _cvhin_consumer_res_trans(
                    vacc,
                    vss,
                    ph0,
                    ph1,
                    ph2,
                    ph3,
                    x_ptr,
                    post_ptr,
                    res_ptr,
                    hin_ptr,
                    alpha_ptr,
                    bias_ptr,
                    vtile,
                    vsub,
                    k_size,
                    hidden,
                    block_m // 2,
                    block_k,
                    block_c,
                    norm_eps,
                    prefetch_hin,
                    external_sink,
                )


@triton.jit
def tiny_family_scores(
    x_ptr, phi_ptr, row, k_size: tl.constexpr, n: tl.constexpr, n_start: tl.constexpr, block_k: tl.constexpr
):
    ns = n_start + tl.arange(0, n)
    ks = tl.arange(0, block_k)
    acc = tl.zeros((n, block_k), tl.float32)
    ss = tl.zeros((block_k,), tl.float32)
    for ki in tl.range(k_size // block_k):
        k = ki * block_k + ks
        v = tl.load(x_ptr + row * k_size + k).to(tl.float32)
        w = tl.load(phi_ptr + ns[:, None] * k_size + k[None, :])
        acc += w * v[None, :]
        ss += v * v
    score = tl.sum(acc, axis=1)
    sumsq = tl.sum(ss, axis=0)
    return score * tl.rsqrt(sumsq / k_size + 1e-05)


@triton.jit
def tiny_partition_forward(
    x_ptr,
    phi_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    res_ptr,
    tokens,
    hidden: tl.constexpr,
    k_size: tl.constexpr,
    block_h: tl.constexpr,
    store_bf16: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if pid < tokens:
        row = pid
        score = tiny_family_scores(x_ptr, phi_ptr, row, k_size, 16, 8, 512)
        lane16 = tl.arange(0, 16)
        logits = (score * tl.load(alpha_ptr + 2) + tl.load(bias_ptr + 8 + lane16)).reshape(4, 4)
        expo = tl.exp(logits - tl.max(logits, axis=1)[:, None])
        normed = expo / tl.sum(expo, axis=1)[:, None] + 1e-06
        normed = normed / (tl.sum(normed, axis=0)[None, :] + 1e-06)
        for step in tl.static_range(19):
            normed = normed / (tl.sum(normed, axis=1)[:, None] + 1e-06)
            normed = normed / (tl.sum(normed, axis=0)[None, :] + 1e-06)
        tl.store(res_ptr + row * 16 + lane16, normed.reshape(16))
    else:
        for work in tl.range(pid - tokens, tokens * 2, nprog - tokens):
            if work < tokens:
                row = work
                score = tiny_family_scores(x_ptr, phi_ptr, row, k_size, 4, 4, 2048)
                lane4 = tl.arange(0, 4)
                post = 2.0 * tl.sigmoid(score * tl.load(alpha_ptr + 1) + tl.load(bias_ptr + 4 + lane4))
                tl.store(post_ptr + row * 4 + lane4, post)
            else:
                row = work - tokens
                score = tiny_family_scores(x_ptr, phi_ptr, row, k_size, 4, 0, 2048)
                lane4 = tl.arange(0, 4)
                pre = tl.sigmoid(score * tl.load(alpha_ptr) + tl.load(bias_ptr + lane4)) + 1e-06
                c = tl.arange(0, block_h)
                hacc = tl.zeros((block_h,), tl.float32)
                for stream in tl.static_range(4):
                    x = tl.load(x_ptr + row * k_size + stream * hidden + c, mask=c < hidden, other=0.0).to(tl.float32)
                    weight = tl.sum(tl.where(lane4 == stream, pre, 0.0), axis=0)
                    hacc += x * weight
                if store_bf16:
                    output = hacc.to(tl.bfloat16)
                else:
                    output = hacc.to(tl.float16)
                tl.store(hin_ptr + row * hidden + c, output, mask=c < hidden)


@triton.jit
def _kt128_cube_tile(
    x_ptr, result_ws, w0, w1, pid, tile, tokens: tl.constexpr, k_size: tl.constexpr, block_k: tl.constexpr
):
    cr = tl.arange(0, 32)
    ck = tl.arange(0, block_k)
    cn = tl.arange(0, 48)
    rows = tile * 32 + cr
    x0 = tl.load(x_ptr + rows[:, None] * k_size + pid * (2 * block_k) + ck[None, :])
    acc = tl.dot(x0, tl.trans(w0))
    gram = tl.dot(x0, tl.trans(x0))
    x1 = tl.load(x_ptr + rows[:, None] * k_size + pid * (2 * block_k) + block_k + ck[None, :])
    acc = tl.dot(x1, tl.trans(w1), acc)
    gram = tl.dot(x1, tl.trans(x1), gram)
    al.sync_block_wait("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
    out = result_ws + pid * (32 * 48 + 32 * 32)
    tl.store(out + cr[:, None] * 48 + cn[None, :], acc)
    tl.store(out + 32 * 48 + cr[:, None] * 32 + cr[None, :], gram)
    al.sync_block_set("cube", "vector", 4, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)


@triton.jit
def tiny_phi_cv_partial(
    x_ptr,
    phi_ptr,
    partial_ptr,
    ss_ptr,
    weight_ws,
    result_ws,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    with al.scope(core_mode="cube"):
        ck = tl.arange(0, block_k)
        cn = tl.arange(0, 48)
        al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        al.sync_block_set("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        al.sync_block_wait("vector", "cube", 0, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
        w0 = tl.load(weight_ws + pid * 2 * 48 * block_k + cn[:, None] * block_k + ck[None, :])
        al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
        w1 = tl.load(weight_ws + (pid * 2 + 1) * 48 * block_k + cn[:, None] * block_k + ck[None, :])
        al.sync_block_set("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        for tile in tl.range(tokens // 32):
            _kt128_cube_tile(x_ptr, result_ws, w0, w1, pid, tile, tokens, k_size, block_k)
        al.sync_block_wait("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
    with al.scope(core_mode="vector"):
        sub = al.sub_vec_id()
        vp = sub * 12 + tl.arange(0, 12)
        vk = tl.arange(0, block_k)
        al.sync_block_set("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
        for slot in tl.static_range(2):
            al.sync_block_wait("cube", "vector", 2 + slot, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            vcol = (pid * 2 + slot) * block_k + vk
            val = tl.load(phi_ptr + vp[:, None] * k_size + vcol[None, :])
            hi = val.to(tl.bfloat16)
            lo = (val - hi.to(tl.float32)).to(tl.bfloat16)
            wb = weight_ws + (pid * 2 + slot) * 48 * block_k
            tl.store(wb + vp[:, None] * block_k + vk[None, :], hi)
            tl.store(wb + (vp[:, None] + 24) * block_k + vk[None, :], lo)
            al.sync_block_set("vector", "cube", slot, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
        al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        al.sync_block_wait("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
        vr = sub * 16 + tl.arange(0, 16)
        vn = tl.arange(0, 32)
        out = result_ws + pid * (32 * 48 + 32 * 32)
        for tile in tl.range(tokens // 32):
            al.sync_block_wait("cube", "vector", 4, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
            ah = tl.load(out + vr[:, None] * 48 + vn[None, :], mask=vn[None, :] < 24, other=0.0)
            alow = tl.load(out + vr[:, None] * 48 + 24 + vn[None, :], mask=vn[None, :] < 24, other=0.0)
            gr = tl.load(out + 32 * 48 + vr[:, None] * 32 + vn[None, :])
            al.sync_block_set("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            ss = tl.sum(tl.where(vr.to(tl.float32)[:, None] == vn.to(tl.float32)[None, :], gr, 0.0), axis=1)
            rows = tile * 32 + vr
            tl.store(partial_ptr + (pid * tokens + rows[:, None]) * 32 + vn[None, :], ah + alow)
            tl.store(ss_ptr + pid * tokens + rows, ss)


@triton.jit
def tiny_partial_cols(
    partial_ptr,
    row,
    start: tl.constexpr,
    tokens: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
    n: tl.constexpr,
):
    ps = tl.arange(0, block_p)
    ns = start + tl.arange(0, n)
    val = tl.load(partial_ptr + (ps[:, None] * tokens + row) * 32 + ns[None, :], mask=ps[:, None] < parts, other=0.0)
    return tl.sum(val, axis=0)


@triton.jit
def tiny_combine_gate(
    partial_ptr,
    ss_ptr,
    x_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    res_ptr,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
    block_h: tl.constexpr,
    store_bf16: tl.constexpr,
):
    cols = tl.arange(0, 4)
    ps = tl.arange(0, block_p)
    for row in tl.range(tl.program_id(0), tokens, tl.num_programs(0)):
        ss = tl.sum(tl.load(ss_ptr + ps * tokens + row, mask=ps < parts, other=0.0), axis=0)
        inv = tl.rsqrt(ss / k_size + 1e-05)
        pre_mix = tiny_partial_cols(partial_ptr, row, 0, tokens, parts, block_p, 4) * inv
        post_mix = tiny_partial_cols(partial_ptr, row, 4, tokens, parts, block_p, 4) * inv
        res_mix = tiny_partial_cols(partial_ptr, row, 8, tokens, parts, block_p, 16) * inv
        pre = tl.sigmoid(pre_mix * tl.load(alpha_ptr) + tl.load(bias_ptr + cols)) + 1e-06
        post = 2.0 * tl.sigmoid(post_mix * tl.load(alpha_ptr + 1) + tl.load(bias_ptr + 4 + cols))
        ns = tl.arange(0, 16)
        logits = (res_mix * tl.load(alpha_ptr + 2) + tl.load(bias_ptr + 8 + ns)).reshape(4, 4)
        expo = tl.exp(logits - tl.max(logits, axis=1)[:, None])
        normed = expo / tl.sum(expo, axis=1)[:, None] + 1e-06
        normed = normed / (tl.sum(normed, axis=0)[None, :] + 1e-06)
        for step in tl.static_range(19):
            normed = normed / (tl.sum(normed, axis=1)[:, None] + 1e-06)
            normed = normed / (tl.sum(normed, axis=0)[None, :] + 1e-06)
        tl.store(res_ptr + row * 16 + ns, normed.reshape(16))
        tl.store(post_ptr + row * 4 + cols, post)
        c = tl.arange(0, block_h)
        hacc = tl.zeros((block_h,), tl.float32)
        for stream in tl.static_range(4):
            v = tl.load(x_ptr + row * k_size + stream * hidden + c, mask=c < hidden, other=0.0).to(tl.float32)
            weight = tl.sum(tl.where(cols == stream, pre, 0.0), axis=0)
            hacc += v * weight
        if store_bf16:
            output = hacc.to(tl.bfloat16)
        else:
            output = hacc.to(tl.float16)
        tl.store(hin_ptr + row * hidden + c, output, mask=c < hidden)


@triton.jit
def tiny_split_pack_typed(
    phi_ptr, packed_ptr, k_size: tl.constexpr, mix_width: tl.constexpr, block_k: tl.constexpr, store_bf16: tl.constexpr
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, 32)
    valid = lanes < mix_width
    for tile in tl.range(pid, k_size // block_k, nprog):
        for part in tl.static_range(block_k // 256):
            k_idx = tile * block_k + part * 256 + tl.arange(0, 256)
            value = tl.load(phi_ptr + lanes[:, None] * k_size + k_idx[None, :], mask=valid[:, None], other=0.0)
            if store_bf16:
                hi = value.to(tl.bfloat16)
                lo = (value - hi.to(tl.float32)).to(tl.bfloat16)
            else:
                hi = value.to(tl.float16)
                lo = (value - hi.to(tl.float32)).to(tl.float16)
            tl.store(packed_ptr + lanes[:, None] * k_size + k_idx[None, :], hi)
            tl.store(packed_ptr + (32 + lanes[:, None]) * k_size + k_idx[None, :], lo)


@triton.jit
def tiny_packed_partial(
    x_ptr,
    packed_ptr,
    partial_ptr,
    ss_ptr,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    part_k: tl.constexpr,
    block_k: tl.constexpr,
):
    local_m = tl.arange(0, 32)
    ns = tl.arange(0, 64)
    ks = tl.arange(0, block_k)
    mt = tokens // 32
    for work in tl.range(tl.program_id(0), 4 * mt, tl.num_programs(0)):
        part = work // mt
        tile = work - part * mt
        rows = tile * 32 + local_m
        acc = tl.zeros((32, 64), tl.float32)
        gram = tl.zeros((32, 32), tl.float32)
        for ki in tl.range(part_k // block_k):
            k = part * part_k + ki * block_k + ks
            loaded = tl.load(x_ptr + rows[:, None] * k_size + k[None, :])
            weight = tl.load(packed_ptr + ns[:, None] * k_size + k[None, :])
            wt = tl.trans(weight)
            xt = tl.trans(loaded)
            acc = tl.dot(loaded, wt, acc)
            gram = tl.dot(loaded, xt, gram)
        ss = tl.sum(tl.where(local_m[:, None] == local_m[None, :], gram, 0.0), axis=1)
        hi = al.extract_slice(acc, (0, 0), (32, 32), (1, 1))
        lo = al.extract_slice(acc, (0, 32), (32, 32), (1, 1))
        mixed = hi + lo
        cols = tl.arange(0, 32)
        tl.store(partial_ptr + (part * tokens + rows[:, None]) * 32 + cols[None, :], mixed)
        tl.store(ss_ptr + part * tokens + rows, ss)


@triton.jit
def phi_cv_single_workspace(
    x_ptr, phi_ptr, weight_ws, workspace, tokens: tl.constexpr, k_size: tl.constexpr, block_k: tl.constexpr
):
    parts: tl.constexpr = k_size // (2 * block_k)
    result_ws = workspace
    partial_ptr = result_ws + parts * (32 * 48 + 32 * 32)
    ss_ptr = partial_ptr + parts * tokens * 32
    tiny_phi_cv_partial(x_ptr, phi_ptr, partial_ptr, ss_ptr, weight_ws, result_ws, tokens, k_size, block_k)


@triton.jit
def _tiny_partial_batch(
    partial_ptr,
    rows,
    start: tl.constexpr,
    tokens: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
    n: tl.constexpr,
):
    ps = tl.arange(0, block_p)
    ns = start + tl.arange(0, n)
    vals = tl.load(
        partial_ptr + (ps[:, None, None] * tokens + rows[None, :, None]) * 32 + ns[None, None, :],
        mask=ps[:, None, None] < parts,
        other=0.0,
    )
    return tl.sum(vals, axis=0)


@triton.jit
def tiny_combine_gate_b2(
    partial_ptr,
    ss_ptr,
    x_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    res_ptr,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
    block_h: tl.constexpr,
    store_bf16: tl.constexpr,
    block_r: tl.constexpr,
):
    rr = tl.arange(0, block_r)
    cols = tl.arange(0, 4)
    ps = tl.arange(0, block_p)
    ns = tl.arange(0, 16)
    cc = tl.arange(0, block_h)
    for tile in tl.range(tl.program_id(0), tokens // block_r, tl.num_programs(0)):
        rows = tile * block_r + rr
        ss = tl.sum(tl.load(ss_ptr + ps[:, None] * tokens + rows[None, :], mask=ps[:, None] < parts, other=0.0), axis=0)
        inv = tl.rsqrt(ss / k_size + 1e-5)
        pre_mix = _tiny_partial_batch(partial_ptr, rows, 0, tokens, parts, block_p, 4) * inv[:, None]
        post_mix = _tiny_partial_batch(partial_ptr, rows, 4, tokens, parts, block_p, 4) * inv[:, None]
        res_mix = _tiny_partial_batch(partial_ptr, rows, 8, tokens, parts, block_p, 16) * inv[:, None]
        pre = tl.sigmoid(pre_mix * tl.load(alpha_ptr) + tl.load(bias_ptr + cols)[None, :]) + 1e-6
        post = 2.0 * tl.sigmoid(post_mix * tl.load(alpha_ptr + 1) + tl.load(bias_ptr + 4 + cols)[None, :])
        logits = (res_mix * tl.load(alpha_ptr + 2) + tl.load(bias_ptr + 8 + ns)[None, :]).reshape(block_r, 4, 4)
        expo = tl.exp(logits - tl.max(logits, axis=2)[:, :, None])
        normed = expo / tl.sum(expo, axis=2)[:, :, None] + 1e-6
        normed = normed / (tl.sum(normed, axis=1)[:, None, :] + 1e-6)
        for step in tl.static_range(19):
            normed = normed / (tl.sum(normed, axis=2)[:, :, None] + 1e-6)
            normed = normed / (tl.sum(normed, axis=1)[:, None, :] + 1e-6)
        tl.store(res_ptr + rows[:, None] * 16 + ns[None, :], normed.reshape(block_r, 16))
        tl.store(post_ptr + rows[:, None] * 4 + cols[None, :], post)
        hacc = tl.zeros((block_r, block_h), tl.float32)
        for stream in tl.static_range(4):
            v = tl.load(
                x_ptr + rows[:, None] * k_size + stream * hidden + cc[None, :], mask=cc[None, :] < hidden, other=0.0
            ).to(tl.float32)
            weight = tl.sum(tl.where(cols[None, :] == stream, pre, 0.0), axis=1)
            hacc += v * weight[:, None]
        if store_bf16:
            output = hacc.to(tl.bfloat16)
        else:
            output = hacc.to(tl.float16)
        tl.store(hin_ptr + rows[:, None] * hidden + cc[None, :], output, mask=cc[None, :] < hidden)


@triton.jit
def cv_workspace_combine(
    workspace,
    x_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    res_ptr,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
    block_h: tl.constexpr,
    store_bf16: tl.constexpr,
):
    result_ws = workspace
    partial_ptr = result_ws + parts * (32 * 48 + 32 * 32)
    ss_ptr = partial_ptr + parts * tokens * 32
    if hidden == 7168:
        tiny_combine_gate_b2(
            partial_ptr,
            ss_ptr,
            x_ptr,
            alpha_ptr,
            bias_ptr,
            hin_ptr,
            post_ptr,
            res_ptr,
            tokens,
            k_size,
            hidden,
            parts,
            block_p,
            block_h,
            store_bf16,
            2,
        )
    else:
        tiny_combine_gate(
            partial_ptr,
            ss_ptr,
            x_ptr,
            alpha_ptr,
            bias_ptr,
            hin_ptr,
            post_ptr,
            res_ptr,
            tokens,
            k_size,
            hidden,
            parts,
            block_p,
            block_h,
            store_bf16,
        )


@triton.jit
def _cv_partition_res32(
    partial_ptr,
    ss_ptr,
    alpha_ptr,
    bias_ptr,
    res_ptr,
    work_pid,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
):
    rows = work_pid * 32 + tl.arange(0, 32)
    ps = tl.arange(0, block_p)
    nn = tl.arange(0, 16)
    ss = tl.sum(tl.load(ss_ptr + ps[:, None] * tokens + rows[None, :], mask=ps[:, None] < parts, other=0.0), axis=0)
    inv = tl.rsqrt(ss / k_size + 1e-5)
    mixed = _tiny_partial_batch(partial_ptr, rows, 8, tokens, parts, block_p, 16) * inv[:, None]
    logits = mixed * tl.load(alpha_ptr + 2) + tl.load(bias_ptr + 8 + nn)[None, :]
    r0c0 = tl.sum(tl.where(nn[None, :] == 0, logits, 0.0), axis=1)
    r0c1 = tl.sum(tl.where(nn[None, :] == 1, logits, 0.0), axis=1)
    r0c2 = tl.sum(tl.where(nn[None, :] == 2, logits, 0.0), axis=1)
    r0c3 = tl.sum(tl.where(nn[None, :] == 3, logits, 0.0), axis=1)
    r1c0 = tl.sum(tl.where(nn[None, :] == 4, logits, 0.0), axis=1)
    r1c1 = tl.sum(tl.where(nn[None, :] == 5, logits, 0.0), axis=1)
    r1c2 = tl.sum(tl.where(nn[None, :] == 6, logits, 0.0), axis=1)
    r1c3 = tl.sum(tl.where(nn[None, :] == 7, logits, 0.0), axis=1)
    r2c0 = tl.sum(tl.where(nn[None, :] == 8, logits, 0.0), axis=1)
    r2c1 = tl.sum(tl.where(nn[None, :] == 9, logits, 0.0), axis=1)
    r2c2 = tl.sum(tl.where(nn[None, :] == 10, logits, 0.0), axis=1)
    r2c3 = tl.sum(tl.where(nn[None, :] == 11, logits, 0.0), axis=1)
    r3c0 = tl.sum(tl.where(nn[None, :] == 12, logits, 0.0), axis=1)
    r3c1 = tl.sum(tl.where(nn[None, :] == 13, logits, 0.0), axis=1)
    r3c2 = tl.sum(tl.where(nn[None, :] == 14, logits, 0.0), axis=1)
    r3c3 = tl.sum(tl.where(nn[None, :] == 15, logits, 0.0), axis=1)
    peak = tl.maximum(tl.maximum(r0c0, r0c1), tl.maximum(r0c2, r0c3))
    r0c0 = tl.exp(r0c0 - peak)
    r0c1 = tl.exp(r0c1 - peak)
    r0c2 = tl.exp(r0c2 - peak)
    r0c3 = tl.exp(r0c3 - peak)
    denom = r0c0 + r0c1 + r0c2 + r0c3
    r0c0 = r0c0 / denom + 1e-06
    r0c1 = r0c1 / denom + 1e-06
    r0c2 = r0c2 / denom + 1e-06
    r0c3 = r0c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r1c0, r1c1), tl.maximum(r1c2, r1c3))
    r1c0 = tl.exp(r1c0 - peak)
    r1c1 = tl.exp(r1c1 - peak)
    r1c2 = tl.exp(r1c2 - peak)
    r1c3 = tl.exp(r1c3 - peak)
    denom = r1c0 + r1c1 + r1c2 + r1c3
    r1c0 = r1c0 / denom + 1e-06
    r1c1 = r1c1 / denom + 1e-06
    r1c2 = r1c2 / denom + 1e-06
    r1c3 = r1c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r2c0, r2c1), tl.maximum(r2c2, r2c3))
    r2c0 = tl.exp(r2c0 - peak)
    r2c1 = tl.exp(r2c1 - peak)
    r2c2 = tl.exp(r2c2 - peak)
    r2c3 = tl.exp(r2c3 - peak)
    denom = r2c0 + r2c1 + r2c2 + r2c3
    r2c0 = r2c0 / denom + 1e-06
    r2c1 = r2c1 / denom + 1e-06
    r2c2 = r2c2 / denom + 1e-06
    r2c3 = r2c3 / denom + 1e-06
    peak = tl.maximum(tl.maximum(r3c0, r3c1), tl.maximum(r3c2, r3c3))
    r3c0 = tl.exp(r3c0 - peak)
    r3c1 = tl.exp(r3c1 - peak)
    r3c2 = tl.exp(r3c2 - peak)
    r3c3 = tl.exp(r3c3 - peak)
    denom = r3c0 + r3c1 + r3c2 + r3c3
    r3c0 = r3c0 / denom + 1e-06
    r3c1 = r3c1 / denom + 1e-06
    r3c2 = r3c2 / denom + 1e-06
    r3c3 = r3c3 / denom + 1e-06
    denom = r0c0 + r1c0 + r2c0 + r3c0 + 1e-06
    r0c0 = r0c0 / denom
    r1c0 = r1c0 / denom
    r2c0 = r2c0 / denom
    r3c0 = r3c0 / denom
    denom = r0c1 + r1c1 + r2c1 + r3c1 + 1e-06
    r0c1 = r0c1 / denom
    r1c1 = r1c1 / denom
    r2c1 = r2c1 / denom
    r3c1 = r3c1 / denom
    denom = r0c2 + r1c2 + r2c2 + r3c2 + 1e-06
    r0c2 = r0c2 / denom
    r1c2 = r1c2 / denom
    r2c2 = r2c2 / denom
    r3c2 = r3c2 / denom
    denom = r0c3 + r1c3 + r2c3 + r3c3 + 1e-06
    r0c3 = r0c3 / denom
    r1c3 = r1c3 / denom
    r2c3 = r2c3 / denom
    r3c3 = r3c3 / denom
    for norm_step in tl.range(19):
        row_d0 = r0c0 + r0c1
        row_d1 = r1c0 + r1c1
        row_d2 = r2c0 + r2c1
        row_d3 = r3c0 + r3c1
        row_d0 = row_d0 + r0c2
        row_d1 = row_d1 + r1c2
        row_d2 = row_d2 + r2c2
        row_d3 = row_d3 + r3c2
        row_d0 = row_d0 + r0c3
        row_d1 = row_d1 + r1c3
        row_d2 = row_d2 + r2c3
        row_d3 = row_d3 + r3c3
        row_d0 = row_d0 + 1e-6
        row_d1 = row_d1 + 1e-6
        row_d2 = row_d2 + 1e-6
        row_d3 = row_d3 + 1e-6
        r0c0 = r0c0 / row_d0
        r1c0 = r1c0 / row_d1
        r2c0 = r2c0 / row_d2
        r3c0 = r3c0 / row_d3
        r0c1 = r0c1 / row_d0
        r1c1 = r1c1 / row_d1
        r2c1 = r2c1 / row_d2
        r3c1 = r3c1 / row_d3
        r0c2 = r0c2 / row_d0
        r1c2 = r1c2 / row_d1
        r2c2 = r2c2 / row_d2
        r3c2 = r3c2 / row_d3
        r0c3 = r0c3 / row_d0
        r1c3 = r1c3 / row_d1
        r2c3 = r2c3 / row_d2
        r3c3 = r3c3 / row_d3
        col_d0 = r0c0 + r1c0
        col_d1 = r0c1 + r1c1
        col_d2 = r0c2 + r1c2
        col_d3 = r0c3 + r1c3
        col_d0 = col_d0 + r2c0
        col_d1 = col_d1 + r2c1
        col_d2 = col_d2 + r2c2
        col_d3 = col_d3 + r2c3
        col_d0 = col_d0 + r3c0
        col_d1 = col_d1 + r3c1
        col_d2 = col_d2 + r3c2
        col_d3 = col_d3 + r3c3
        col_d0 = col_d0 + 1e-6
        col_d1 = col_d1 + 1e-6
        col_d2 = col_d2 + 1e-6
        col_d3 = col_d3 + 1e-6
        r0c0 = r0c0 / col_d0
        r0c1 = r0c1 / col_d1
        r0c2 = r0c2 / col_d2
        r0c3 = r0c3 / col_d3
        r1c0 = r1c0 / col_d0
        r1c1 = r1c1 / col_d1
        r1c2 = r1c2 / col_d2
        r1c3 = r1c3 / col_d3
        r2c0 = r2c0 / col_d0
        r2c1 = r2c1 / col_d1
        r2c2 = r2c2 / col_d2
        r2c3 = r2c3 / col_d3
        r3c0 = r3c0 / col_d0
        r3c1 = r3c1 / col_d1
        r3c2 = r3c2 / col_d2
        r3c3 = r3c3 / col_d3
    tl.store(res_ptr + rows * 16 + 0, r0c0)
    tl.store(res_ptr + rows * 16 + 1, r0c1)
    tl.store(res_ptr + rows * 16 + 2, r0c2)
    tl.store(res_ptr + rows * 16 + 3, r0c3)
    tl.store(res_ptr + rows * 16 + 4, r1c0)
    tl.store(res_ptr + rows * 16 + 5, r1c1)
    tl.store(res_ptr + rows * 16 + 6, r1c2)
    tl.store(res_ptr + rows * 16 + 7, r1c3)
    tl.store(res_ptr + rows * 16 + 8, r2c0)
    tl.store(res_ptr + rows * 16 + 9, r2c1)
    tl.store(res_ptr + rows * 16 + 10, r2c2)
    tl.store(res_ptr + rows * 16 + 11, r2c3)
    tl.store(res_ptr + rows * 16 + 12, r3c0)
    tl.store(res_ptr + rows * 16 + 13, r3c1)
    tl.store(res_ptr + rows * 16 + 14, r3c2)
    tl.store(res_ptr + rows * 16 + 15, r3c3)


@triton.jit
def _cv_partition_hin4(
    partial_ptr,
    ss_ptr,
    x_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    work_pid,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
):
    rr = tl.arange(0, 4)
    ps = tl.arange(0, block_p)
    cc = tl.arange(0, 2048)
    cols4 = tl.arange(0, 4)
    for tile in tl.range(work_pid - 4, tokens // 4, 36):
        rows = tile * 4 + rr
        ss = tl.sum(tl.load(ss_ptr + ps[:, None] * tokens + rows[None, :], mask=ps[:, None] < parts, other=0.0), axis=0)
        inv = tl.rsqrt(ss / k_size + 1e-5)
        mixed = _tiny_partial_batch(partial_ptr, rows, 0, tokens, parts, block_p, 8) * inv[:, None]
        premix = al.extract_slice(mixed, (0, 0), (4, 4), (1, 1))
        postmix = al.extract_slice(mixed, (0, 4), (4, 4), (1, 1))
        pre = tl.sigmoid(premix * tl.load(alpha_ptr) + tl.load(bias_ptr + cols4)[None, :]) + 1e-6
        post = 2.0 * tl.sigmoid(postmix * tl.load(alpha_ptr + 1) + tl.load(bias_ptr + 4 + cols4)[None, :])
        tl.store(post_ptr + rows[:, None] * 4 + cols4[None, :], post)
        p0 = tl.sum(tl.where(cols4[None, :] == 0, pre, 0.0), axis=1)
        p1 = tl.sum(tl.where(cols4[None, :] == 1, pre, 0.0), axis=1)
        p2 = tl.sum(tl.where(cols4[None, :] == 2, pre, 0.0), axis=1)
        p3 = tl.sum(tl.where(cols4[None, :] == 3, pre, 0.0), axis=1)
        for c_tile in tl.range(tl.cdiv(hidden, 2048)):
            channels = c_tile * 2048 + cc
            base = x_ptr + rows[:, None] * k_size + channels[None, :]
            h = tl.load(base, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p0[:, None]
            h += tl.load(base + hidden, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p1[:, None]
            h += tl.load(base + 2 * hidden, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p2[:, None]
            h += tl.load(base + 3 * hidden, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p3[:, None]
            tl.store(
                hin_ptr + rows[:, None] * hidden + channels[None, :], h.to(tl.bfloat16), mask=channels[None, :] < hidden
            )


@triton.jit
def phi_cv_gate_fused128(
    x_ptr,
    phi_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    res_ptr,
    weight_ws,
    workspace,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    result_ws = workspace
    partial_ptr = result_ws + parts * (32 * 48 + 32 * 32)
    ss_ptr = partial_ptr + parts * tokens * 32
    with al.scope(core_mode="cube"):
        if pid < parts:
            ck = tl.arange(0, block_k)
            cn = tl.arange(0, 48)
            al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_set("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_wait("vector", "cube", 0, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            w0 = tl.load(weight_ws + pid * 2 * 48 * block_k + cn[:, None] * block_k + ck[None, :])
            al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            w1 = tl.load(weight_ws + (pid * 2 + 1) * 48 * block_k + cn[:, None] * block_k + ck[None, :])
            al.sync_block_set("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            for tile in tl.range(tokens // 32):
                _kt128_cube_tile(x_ptr, result_ws, w0, w1, pid, tile, tokens, k_size, block_k)
            al.sync_block_wait("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
    with al.scope(core_mode="vector"):
        sub = al.sub_vec_id()
        if pid < parts:
            vp = sub * 12 + tl.arange(0, 12)
            vk = tl.arange(0, block_k)
            al.sync_block_set("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            for slot in tl.static_range(2):
                al.sync_block_wait("cube", "vector", 2 + slot, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
                vcol = (pid * 2 + slot) * block_k + vk
                val = tl.load(phi_ptr + vp[:, None] * k_size + vcol[None, :])
                hi = val.to(tl.bfloat16)
                lo = (val - hi.to(tl.float32)).to(tl.bfloat16)
                wb = weight_ws + (pid * 2 + slot) * 48 * block_k
                tl.store(wb + vp[:, None] * block_k + vk[None, :], hi)
                tl.store(wb + (vp[:, None] + 24) * block_k + vk[None, :], lo)
                al.sync_block_set("vector", "cube", slot, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_wait("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            vr = sub * 16 + tl.arange(0, 16)
            vn = tl.arange(0, 32)
            out = result_ws + pid * (32 * 48 + 32 * 32)
            for tile in tl.range(tokens // 32):
                al.sync_block_wait("cube", "vector", 4, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
                ah = tl.load(out + vr[:, None] * 48 + vn[None, :], mask=vn[None, :] < 24, other=0.0)
                alow = tl.load(out + vr[:, None] * 48 + 24 + vn[None, :], mask=vn[None, :] < 24, other=0.0)
                gr = tl.load(out + 32 * 48 + vr[:, None] * 32 + vn[None, :])
                al.sync_block_set("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
                ss = tl.sum(tl.where(vr.to(tl.float32)[:, None] == vn.to(tl.float32)[None, :], gr, 0.0), axis=1)
                rows = tile * 32 + vr
                tl.store(partial_ptr + (pid * tokens + rows[:, None]) * 32 + vn[None, :], ah + alow)
                tl.store(ss_ptr + pid * tokens + rows, ss)
        al.sync_block_all("all_vector", 10)
        work_pid = (pid * 2 + sub).to(tl.int32)
        if work_pid < 4:
            _cv_partition_res32(
                partial_ptr, ss_ptr, alpha_ptr, bias_ptr, res_ptr, work_pid, tokens, k_size, parts, block_p
            )
        else:
            _cv_partition_hin4(
                partial_ptr,
                ss_ptr,
                x_ptr,
                alpha_ptr,
                bias_ptr,
                hin_ptr,
                post_ptr,
                work_pid,
                tokens,
                k_size,
                hidden,
                parts,
                block_p,
            )


@triton.jit
def _cv_partition_hin4_fp16(
    partial_ptr,
    ss_ptr,
    x_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    work_pid,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
):
    rr = tl.arange(0, 4)
    ps = tl.arange(0, block_p)
    cc = tl.arange(0, 2048)
    cols4 = tl.arange(0, 4)
    for tile in tl.range(work_pid - 4, tokens // 4, 36):
        rows = tile * 4 + rr
        ss = tl.sum(tl.load(ss_ptr + ps[:, None] * tokens + rows[None, :], mask=ps[:, None] < parts, other=0.0), axis=0)
        inv = tl.rsqrt(ss / k_size + 1e-5)
        mixed = _tiny_partial_batch(partial_ptr, rows, 0, tokens, parts, block_p, 8) * inv[:, None]
        premix = al.extract_slice(mixed, (0, 0), (4, 4), (1, 1))
        postmix = al.extract_slice(mixed, (0, 4), (4, 4), (1, 1))
        pre = tl.sigmoid(premix * tl.load(alpha_ptr) + tl.load(bias_ptr + cols4)[None, :]) + 1e-6
        post = 2.0 * tl.sigmoid(postmix * tl.load(alpha_ptr + 1) + tl.load(bias_ptr + 4 + cols4)[None, :])
        tl.store(post_ptr + rows[:, None] * 4 + cols4[None, :], post)
        p0 = tl.sum(tl.where(cols4[None, :] == 0, pre, 0.0), axis=1)
        p1 = tl.sum(tl.where(cols4[None, :] == 1, pre, 0.0), axis=1)
        p2 = tl.sum(tl.where(cols4[None, :] == 2, pre, 0.0), axis=1)
        p3 = tl.sum(tl.where(cols4[None, :] == 3, pre, 0.0), axis=1)
        for c_tile in tl.range(tl.cdiv(hidden, 2048)):
            channels = c_tile * 2048 + cc
            base = x_ptr + rows[:, None] * k_size + channels[None, :]
            h = tl.load(base, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p0[:, None]
            h += tl.load(base + hidden, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p1[:, None]
            h += tl.load(base + 2 * hidden, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p2[:, None]
            h += tl.load(base + 3 * hidden, mask=channels[None, :] < hidden, other=0.0).to(tl.float32) * p3[:, None]
            tl.store(
                hin_ptr + rows[:, None] * hidden + channels[None, :], h.to(tl.float16), mask=channels[None, :] < hidden
            )


@triton.jit
def phi_cv_gate_fused128_fp16(
    x_ptr,
    phi_ptr,
    alpha_ptr,
    bias_ptr,
    hin_ptr,
    post_ptr,
    res_ptr,
    weight_ws,
    workspace,
    tokens: tl.constexpr,
    k_size: tl.constexpr,
    hidden: tl.constexpr,
    parts: tl.constexpr,
    block_p: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    result_ws = workspace
    partial_ptr = result_ws + parts * (32 * 48 + 32 * 32)
    ss_ptr = partial_ptr + parts * tokens * 32
    with al.scope(core_mode="cube"):
        if pid < parts:
            ck = tl.arange(0, block_k)
            cn = tl.arange(0, 48)
            al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_set("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_wait("vector", "cube", 0, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            w0 = tl.load(weight_ws + pid * 2 * 48 * block_k + cn[:, None] * block_k + ck[None, :])
            al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            w1 = tl.load(weight_ws + (pid * 2 + 1) * 48 * block_k + cn[:, None] * block_k + ck[None, :])
            al.sync_block_set("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            for tile in tl.range(tokens // 32):
                _kt128_cube_tile(x_ptr, result_ws, w0, w1, pid, tile, tokens, k_size, block_k)
            al.sync_block_wait("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
    with al.scope(core_mode="vector"):
        sub = al.sub_vec_id()
        if pid < parts:
            vp = sub * 12 + tl.arange(0, 12)
            vk = tl.arange(0, block_k)
            al.sync_block_set("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
            for slot in tl.static_range(2):
                al.sync_block_wait("cube", "vector", 2 + slot, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
                vcol = (pid * 2 + slot) * block_k + vk
                val = tl.load(phi_ptr + vp[:, None] * k_size + vcol[None, :])
                hi = val.to(tl.float16)
                lo = (val - hi.to(tl.float32)).to(tl.float16)
                wb = weight_ws + (pid * 2 + slot) * 48 * block_k
                tl.store(wb + vp[:, None] * block_k + vk[None, :], hi)
                tl.store(wb + (vp[:, None] + 24) * block_k + vk[None, :], lo)
                al.sync_block_set("vector", "cube", slot, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            al.sync_block_wait("cube", "vector", 3, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
            vr = sub * 16 + tl.arange(0, 16)
            vn = tl.arange(0, 32)
            out = result_ws + pid * (32 * 48 + 32 * 32)
            for tile in tl.range(tokens // 32):
                al.sync_block_wait("cube", "vector", 4, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
                ah = tl.load(out + vr[:, None] * 48 + vn[None, :], mask=vn[None, :] < 24, other=0.0)
                alow = tl.load(out + vr[:, None] * 48 + 24 + vn[None, :], mask=vn[None, :] < 24, other=0.0)
                gr = tl.load(out + 32 * 48 + vr[:, None] * 32 + vn[None, :])
                al.sync_block_set("vector", "cube", 5, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX)
                ss = tl.sum(tl.where(vr.to(tl.float32)[:, None] == vn.to(tl.float32)[None, :], gr, 0.0), axis=1)
                rows = tile * 32 + vr
                tl.store(partial_ptr + (pid * tokens + rows[:, None]) * 32 + vn[None, :], ah + alow)
                tl.store(ss_ptr + pid * tokens + rows, ss)
        al.sync_block_all("all_vector", 10)
        work_pid = (pid * 2 + sub).to(tl.int32)
        if work_pid < 4:
            _cv_partition_res32(
                partial_ptr, ss_ptr, alpha_ptr, bias_ptr, res_ptr, work_pid, tokens, k_size, parts, block_p
            )
        else:
            _cv_partition_hin4_fp16(
                partial_ptr,
                ss_ptr,
                x_ptr,
                alpha_ptr,
                bias_ptr,
                hin_ptr,
                post_ptr,
                work_pid,
                tokens,
                k_size,
                hidden,
                parts,
                block_p,
            )


@lru_cache(maxsize=None)
def _core_counts(device_index):
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties["num_aicore"]), int(properties["num_vectorcore"])


def _launch(x, phi, alpha, bias):
    cube_cores, vector_cores = _core_counts(x.device.index)
    batches, seq, streams, hidden = x.shape
    tokens = batches * seq
    if tokens == 128 and cube_cores == 20 and (x.dtype == torch.float16):
        k_size = 4 * hidden
        parts = k_size // 2048
        workspace = torch.empty((parts * (32 * 48 + 32 * 32 + tokens * 33),), device=x.device, dtype=torch.float32)
        weights = torch.empty((parts * 2 * 48 * 1024,), device=x.device, dtype=torch.float16)
        hin = torch.empty((batches, seq, hidden), device=x.device, dtype=x.dtype)
        post = torch.empty((batches, seq, 4), device=x.device, dtype=torch.float32)
        res = torch.empty((batches, seq, 16), device=x.device, dtype=torch.float32)
        phi_cv_gate_fused128_fp16[20,](
            x,
            phi,
            alpha,
            bias,
            hin,
            post,
            res,
            weights,
            workspace,
            tokens,
            k_size=k_size,
            hidden=hidden,
            parts=parts,
            block_p=triton.next_power_of_2(parts),
            block_h=triton.next_power_of_2(hidden),
            block_k=1024,
            disable_auto_inject_block_sync=True,
        )
        return (hin, post, res)
    if tokens == 128 and cube_cores == 20 and (x.dtype == torch.bfloat16):
        k_size = 4 * hidden
        parts = k_size // 2048
        workspace = torch.empty((parts * (32 * 48 + 32 * 32 + tokens * 33),), device=x.device, dtype=torch.float32)
        weights = torch.empty((parts * 2 * 48 * 1024,), device=x.device, dtype=torch.bfloat16)
        hin = torch.empty((batches, seq, hidden), device=x.device, dtype=x.dtype)
        post = torch.empty((batches, seq, 4), device=x.device, dtype=torch.float32)
        res = torch.empty((batches, seq, 16), device=x.device, dtype=torch.float32)
        phi_cv_gate_fused128[20,](
            x,
            phi,
            alpha,
            bias,
            hin,
            post,
            res,
            weights,
            workspace,
            tokens,
            k_size=k_size,
            hidden=hidden,
            parts=parts,
            block_p=triton.next_power_of_2(parts),
            block_h=triton.next_power_of_2(hidden),
            block_k=1024,
            disable_auto_inject_block_sync=True,
        )
        return (hin, post, res)
    if tokens == 128 and x.dtype == torch.bfloat16 and (4 * hidden // 2048 <= cube_cores):
        k_size = 4 * hidden
        parts = k_size // 2048
        workspace = torch.empty((parts * (32 * 48 + 32 * 32 + tokens * 33),), device=x.device, dtype=torch.float32)
        weights = torch.empty((parts * 2 * 48 * 1024,), device=x.device, dtype=torch.bfloat16)
        hin = torch.empty((batches, seq, hidden), device=x.device, dtype=x.dtype)
        post = torch.empty((batches, seq, 4), device=x.device, dtype=torch.float32)
        res = torch.empty((batches, seq, 16), device=x.device, dtype=torch.float32)
        phi_cv_single_workspace[parts,](
            x, phi, weights, workspace, tokens, k_size=k_size, block_k=1024, disable_auto_inject_block_sync=True
        )
        cv_workspace_combine[min(tokens, vector_cores),](
            workspace,
            x,
            alpha,
            bias,
            hin,
            post,
            res,
            tokens,
            k_size=k_size,
            hidden=hidden,
            parts=parts,
            block_p=triton.next_power_of_2(parts),
            block_h=triton.next_power_of_2(hidden),
            store_bf16=True,
        )
        return (hin, post, res)
    if tokens == 128 and x.dtype == torch.bfloat16:
        k_size = 4 * hidden
        packed = torch.empty((k_size * 64,), device=x.device, dtype=x.dtype)
        partial = torch.empty((4, tokens, 32), device=x.device, dtype=torch.float32)
        ss = torch.empty((4, tokens), device=x.device, dtype=torch.float32)
        hin = torch.empty((batches, seq, hidden), device=x.device, dtype=x.dtype)
        post = torch.empty((batches, seq, 4), device=x.device, dtype=torch.float32)
        res = torch.empty((batches, seq, 16), device=x.device, dtype=torch.float32)
        tiny_split_pack_typed[min(triton.cdiv(k_size, 1024), cube_cores),](
            phi, packed, k_size=k_size, mix_width=24, block_k=1024, store_bf16=x.dtype == torch.bfloat16
        )
        tiny_packed_partial[min(4 * (tokens // 32), cube_cores),](
            x, packed, partial, ss, tokens, k_size=k_size, part_k=k_size // 4, block_k=1024
        )
        tiny_combine_gate[min(tokens, vector_cores),](
            partial,
            ss,
            x,
            alpha,
            bias,
            hin,
            post,
            res,
            tokens,
            k_size=k_size,
            hidden=hidden,
            parts=4,
            block_p=4,
            block_h=triton.next_power_of_2(hidden),
            store_bf16=x.dtype == torch.bfloat16,
        )
        return (hin, post, res)
    if tokens <= min(17, vector_cores // 2):
        hin = torch.empty((batches, seq, hidden), device=x.device, dtype=x.dtype)
        post = torch.empty((batches, seq, 4), device=x.device, dtype=torch.float32)
        res = torch.empty((batches, seq, 16), device=x.device, dtype=torch.float32)
        tiny_partition_forward[min(tokens * 3, vector_cores),](
            x,
            phi,
            alpha,
            bias,
            hin,
            post,
            res,
            tokens,
            hidden=hidden,
            k_size=4 * hidden,
            block_h=triton.next_power_of_2(hidden),
            store_bf16=x.dtype == torch.bfloat16,
        )
        return (hin, post, res)
    if (
        (256 <= tokens <= 2048 or tokens == 8192)
        and tokens % 64 == 0
        and (streams == 4)
        and (x.dtype == torch.bfloat16)
        and (4 * hidden % 1024 == 0)
        and x.is_contiguous()
        and phi.is_contiguous()
        and alpha.is_contiguous()
        and bias.is_contiguous()
    ):
        k_size = 4 * hidden
        packed = torch.empty((k_size * 48,), device=x.device, dtype=torch.bfloat16)
        post = torch.empty((tokens, 4), device=x.device, dtype=torch.float32)
        res = torch.empty((tokens, 16), device=x.device, dtype=torch.float32)
        hin = torch.empty((tokens, hidden), device=x.device, dtype=x.dtype)
        split_pack_phi_flat48[min(triton.cdiv(24 * k_size, 4096), vector_cores),](
            phi, packed, k_size=k_size, mix_width=24, block=4096
        )
        _cvhin_m = 32 if tokens == 512 else 64
        _cvhin_grid = min(triton.cdiv(tokens, _cvhin_m), cube_cores)
        _cvhin_workspace = torch.empty(
            (_cvhin_grid * (_cvhin_m * 48 + _cvhin_m * _cvhin_m),), device=x.device, dtype=torch.float32
        )
        if tokens == 8192:
            fused_m64_gram_balanced_cv[min(triton.cdiv(tokens, 32 if tokens == 512 else 64), cube_cores),](
                x.view(tokens, k_size),
                packed,
                post,
                res,
                hin,
                alpha,
                bias,
                _cvhin_workspace,
                tokens,
                k_size=k_size,
                hidden=hidden,
                block_m=32 if tokens == 512 else 64,
                block_k=1024,
                block_c=256,
                norm_eps=NORM_EPS,
                prefetch_hin=True,
                external_sink=False,
                regular_tiles=tokens // 64 // cube_cores * cube_cores,
                tail_tiles=(tokens - tokens // 64 // cube_cores * cube_cores * 64) // 32,
                disable_auto_inject_block_sync=True,
            )
        else:
            fused_logits_n48_cvhin_res_trans[min(triton.cdiv(tokens, 32 if tokens == 512 else 64), cube_cores),](
                x.view(tokens, k_size),
                packed,
                post,
                res,
                hin,
                alpha,
                bias,
                _cvhin_workspace,
                tokens,
                k_size=k_size,
                hidden=hidden,
                block_m=32 if tokens == 512 else 64,
                block_k=1024,
                block_c=256,
                norm_eps=NORM_EPS,
                prefetch_hin=True,
                external_sink=False,
                early_sink=tokens == 1024 and hidden == 7168,
                disable_auto_inject_block_sync=True,
            )
        return (hin.view(batches, seq, hidden), post.view(batches, seq, 4), res.view(batches, seq, 16))
    if (
        x.dtype == torch.bfloat16
        and x.shape[0] * x.shape[1] >= 256
        and (x.shape[0] * x.shape[1] % 64 == 0)
        and (4 * x.shape[3] % 1024 == 0)
    ):
        k_size = 4 * hidden
        packed = torch.empty((k_size * 64,), device=x.device, dtype=torch.bfloat16)
        split_pack_phi[cube_cores,](phi, packed, k_size=k_size, mix_width=24, block_k=1024)
        post = torch.empty((tokens, 4), device=x.device, dtype=torch.float32)
        res = torch.empty((tokens, 16), device=x.device, dtype=torch.float32)
        hin = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
        regular_tiles = tokens // 64 // cube_cores * cube_cores
        tail_tiles = (tokens - regular_tiles * 64) // 32
        if tokens >= 8192 and 0 < tail_tiles <= cube_cores:
            fused_logits_gram_balanced[cube_cores,](
                x.view(tokens, k_size),
                packed,
                post,
                res,
                hin,
                alpha,
                bias,
                tokens,
                k_size=k_size,
                hidden=hidden,
                block_k=1024,
                block_c=256,
                norm_eps=NORM_EPS,
                regular_tiles=regular_tiles,
                tail_tiles=tail_tiles,
            )
        else:
            fused_logits[cube_cores,](
                x.view(tokens, k_size),
                packed,
                post,
                res,
                hin,
                alpha,
                bias,
                tokens,
                k_size=k_size,
                hidden=hidden,
                block_m=64,
                block_k=1024,
                block_c=256,
                norm_eps=NORM_EPS,
            )
        sinkhorn20[triton.cdiv(tokens, 256),](res, res, tokens, block=256)
        return (hin.view(batches, seq, hidden), post.view(batches, seq, 4), res.view(batches, seq, 16))
    else:
        flat = x.view(tokens, streams * hidden)
        mixes = torch.empty((tokens, MIX_WIDTH), dtype=torch.float32, device=x.device)
        pre = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
        h_post = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
        h_res = torch.empty((tokens, streams * streams), dtype=torch.float32, device=x.device)
        hin = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
        large = tokens >= 256
        block_m = 32 if large else 16
        block_k = 512 if large else 256
        block_c = 1024 if large else 256
        gemm_tiles = triton.cdiv(tokens, block_m)
        gemm_grid = (min(gemm_tiles, cube_cores),)
        k_size = streams * hidden
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
            norm_eps=NORM_EPS,
        )
        if large:
            gate_tiles = triton.cdiv(tokens, BLOCK_T)
            gate_grid = (min(gate_tiles, vector_cores),)
            mhc_gate_kernel[gate_grid](
                mixes,
                alpha,
                bias,
                h_post,
                h_res,
                pre,
                tokens,
                mix_width=MIX_WIDTH,
                streams=HC_MULT,
                sinkhorn_iters=20,
                hc_eps=HC_EPS,
                block_t=BLOCK_T,
            )
        else:
            gate_grid = (min(tokens, vector_cores),)
            mhc_gate_hin_small_kernel[gate_grid](
                mixes,
                alpha,
                bias,
                h_post,
                h_res,
                flat,
                hin,
                tokens,
                hidden=hidden,
                block_h=triton.next_power_of_2(hidden),
                store_bf16=x.dtype == torch.bfloat16,
                mix_width=MIX_WIDTH,
                streams=HC_MULT,
                sinkhorn_iters=20,
                hc_eps=HC_EPS,
            )
        if large:
            c_tiles = hidden // block_c
            hin_tiles = tokens * c_tiles
            hin_grid = (min(hin_tiles, vector_cores),)
            mhc_hin_kernel[hin_grid](
                flat,
                pre,
                hin,
                tokens,
                hidden=hidden,
                streams=HC_MULT,
                block_c=block_c,
                store_bf16=x.dtype == torch.bfloat16,
            )
        return (
            hin.view(batches, seq, hidden),
            h_post.view(batches, seq, streams),
            h_res.view(batches, seq, streams * streams),
        )


@torch.library.custom_op("mojo_npu_triton_a2::mhc_coefficients", mutates_args=(), device_types="npu")
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
