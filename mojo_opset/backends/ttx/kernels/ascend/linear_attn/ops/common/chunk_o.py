# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, Hongmin Chen

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.ascend.utils import exp
from mojo_opset.backends.ttx.kernels.ascend.utils import get_num_cores


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[
#         triton.Config({"BK": 128, "BV": 128}),
#         triton.Config({"BK": 64, "BV": 64}),
#         triton.Config({"BK": 32, "BV": 32}),
#     ],
#     key=["H", "K", "V", "BT"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_dim: tl.int32,
    B: tl.int32,
    NV: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_bh = B * H
    total_tasks = NV * NT_dim * num_bh

    for task_id in range(pid, total_tasks, num_programs):
        i_bh = task_id // (NV * NT_dim)
        remaining = task_id % (NV * NT_dim)
        i_t = remaining // NV
        i_v = remaining % NV

        i_b, i_h = i_bh // H, i_bh % H

        i_hk = i_h // (H // HK)

        if IS_VARLEN:
            i_tg = i_t

            i_n, i_t_real = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_len = eos - bos

            NT = tl.cdiv(T_len, BT)
        else:
            T_len = T
            NT = tl.cdiv(T_len, BT)
            i_tg = i_b * NT + i_t
            bos, eos = i_b * T, i_b * T + T
            i_t_real = i_t

        if i_t < NT:
            q_base = q + (bos * H + i_h) * K
            k_base = k + (bos * HK + i_hk) * K
            v_base = v + (bos * HK + i_hk) * V
            o_base = o + (bos * H + i_h) * V

            h_base = h + (i_tg * HK + i_hk).to(tl.int64) * K * V

            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            b_A = tl.zeros([BT, BT], dtype=tl.float32)

            for i_k in range(tl.cdiv(K, BK)):
                p_q = tl.make_block_ptr(q_base, (T_len, K), (H * K, 1), (i_t_real * BT, i_k * BK), (BT, BK), (1, 0))

                p_k = tl.make_block_ptr(k_base, (K, T_len), (1, HK * K), (i_k * BK, i_t_real * BT), (BK, BT), (0, 1))

                p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))

                b_o += tl.dot(b_q, b_h)

                b_A += tl.dot(b_q, b_k)

            if USE_G:
                g_base = g + bos * HK + i_hk
                p_g = tl.make_block_ptr(g_base, (T_len,), (HK,), (i_t_real * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))

                b_o = b_o * tl.exp(b_g)[:, None]

                b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

            if USE_G_GAMMA:
                b_gamma = tl.load(g_gamma + i_hk)
                b_g_pos = b_gamma * (tl.arange(0, BT) + i_t_real * BT + 1)

                mask_pos = (i_t_real * BT + tl.arange(0, BT)) < T_len
                b_g_pos = tl.where(mask_pos, b_g_pos, 0)

                b_o = b_o * tl.exp(b_g_pos)[:, None]

                b_A = b_A * tl.exp(b_g_pos[:, None] - b_g_pos[None, :])

            o_t = i_t_real * BT + tl.arange(0, BT)
            m_t = o_t < T_len

            m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
            b_A = tl.where(m_A, b_A, 0)

            p_v = tl.make_block_ptr(v_base, (T_len, V), (HK * V, 1), (i_t_real * BT, i_v * BV), (BT, BV), (1, 0))
            p_o = tl.make_block_ptr(o_base, (T_len, V), (H * V, 1), (i_t_real * BT, i_v * BV), (BT, BV), (1, 0))

            b_v = tl.load(p_v, boundary_check=(0, 1))

            b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    do,
    dh,
    dq,
    dk,
    dg,
    w,
    dv,
    dw,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.int32,
    T,
    H_dim: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NK_dim: tl.int32,
    NT_dim: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_bh = B * H_dim
    total_tasks = NK_dim * NT_dim * num_bh

    for task_id in range(pid, total_tasks, num_programs):
        i_bh = task_id // (NK_dim * NT_dim)
        remaining = task_id % (NK_dim * NT_dim)
        i_t_idx = remaining // NK_dim
        i_k = remaining % NK_dim

        i_b, i_h = i_bh // H_dim, i_bh % H_dim
        i_hk = i_h // (H_dim // HK)

        if IS_VARLEN:
            i_tg = i_t_idx
            i_n, i_t_real = (
                tl.load(chunk_indices + i_t_idx * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_idx * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_total = T
            T_len = eos - bos
            NT = tl.cdiv(T_len, BT)

            all_idx_for_dg = T_total
        else:
            T_len = T
            NT = tl.cdiv(T_len, BT)
            i_tg = i_b * NT + i_t_idx
            bos, eos = i_b * T, i_b * T + T
            i_t_real = i_t_idx
            T_total = T
            all_idx_for_dg = T_total

        if i_t_real < NT:
            v_base = v + (bos * HK + i_hk) * V
            do_base = do + (bos * H_dim + i_h) * V
            h_base = h + (i_tg * HK + i_hk).to(tl.int64) * K * V
            dh_base = dh + (i_tg * H_dim + i_h).to(tl.int64) * K * V
            q_base = q + (bos * H_dim + i_h) * K
            k_base = k + (bos * HK + i_hk) * K
            dq_base = dq + (bos * H_dim + i_h) * K
            dk_base = dk + (bos * H_dim + i_h) * K

            if USE_DW:
                w_base = w + (bos * HK + i_hk) * K
                dw_base = dw + (bos * H_dim + i_h) * K
                dv_base = dv + (bos * H_dim + i_h) * V

            if USE_G:
                dg_base = dg + i_k * (B * T_total * H_dim)
                b_dg_last = tl.zeros(
                    [
                        1,
                    ],
                    dtype=tl.float32,
                )

            if USE_G_GAMMA:
                b_gamma = tl.load(g_gamma + i_hk)

                b_g = b_gamma * (tl.arange(0, BT) + i_t_real * BT + 1)

                b_g_last_val = b_gamma * tl.minimum((i_t_real + 1) * BT, T_len)

            b_dq = tl.zeros([BT, BK], dtype=tl.float32)
            b_dk = tl.zeros([BT, BK], dtype=tl.float32)
            b_ds = tl.zeros([BT, BT], dtype=tl.float32)
            b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

            for i_v in range(tl.cdiv(V, BV)):
                p_v = tl.make_block_ptr(v_base, (T_len, V), (HK * V, 1), (i_t_real * BT, i_v * BV), (BT, BV), (1, 0))
                p_do = tl.make_block_ptr(
                    do_base, (T_len, V), (H_dim * V, 1), (i_t_real * BT, i_v * BV), (BT, BV), (1, 0)
                )

                p_h = tl.make_block_ptr(h_base, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh_base, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))

                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dh = tl.load(p_dh, boundary_check=(0, 1))

                if USE_G:
                    b_dg_last += tl.sum(b_h * b_dh)

                b_ds += tl.dot(b_do, tl.trans(b_v))

                b_dq += tl.dot(b_do, b_h.to(b_do.dtype))

                tl.debug_barrier()
                b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

                if USE_DW:
                    p_dv = tl.make_block_ptr(
                        dv_base, (T_len, V), (H_dim * V, 1), (i_t_real * BT, i_v * BV), (BT, BV), (1, 0)
                    )
                    b_dv_val = tl.load(p_dv, boundary_check=(0, 1))

                    b_dw += tl.dot(b_dv_val.to(b_v.dtype), b_h.to(b_v.dtype))

            if USE_DW:
                p_dw = tl.make_block_ptr(
                    dw_base, (T_len, K), (H_dim * K, 1), (i_t_real * BT, i_k * BK), (BT, BK), (1, 0)
                )

                tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

            tl.debug_barrier()

            p_q = tl.make_block_ptr(q_base, (T_len, K), (H_dim * K, 1), (i_t_real * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k_base, (T_len, K), (HK * K, 1), (i_t_real * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            p_dq = tl.make_block_ptr(dq_base, (T_len, K), (H_dim * K, 1), (i_t_real * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk_base, (T_len, K), (H_dim * K, 1), (i_t_real * BT, i_k * BK), (BT, BK), (1, 0))

            o_t = i_t_real * BT + tl.arange(0, BT)
            m_t = o_t < T_len

            m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)

            if USE_G:
                b_dg = tl.zeros(
                    [
                        BT,
                    ],
                    dtype=tl.float32,
                )
                g_base = g + bos * HK + i_hk
                dg_base = dg_base + bos * H_dim + i_h

                p_g = tl.make_block_ptr(g_base, (T_len,), (HK,), (i_t_real * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))

                last_idx = tl.minimum(i_t_real * BT + BT, T_len) - 1
                b_g_last = tl.load(g_base + last_idx * HK)

                b_dg_last *= tl.exp(b_g_last)

                b_dq = b_dq * tl.exp(b_g)[:, None] * scale

                b_dg += tl.sum(b_dq * b_q, axis=1)

                b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]

                b_dg -= tl.sum(b_k * b_dk, axis=1)

                b_dg_last += tl.sum(b_dk * b_k)

                b_ds = tl.where(m_A, b_ds * tl.exp(b_g[:, None] - b_g[None, :]), 0) * scale

                b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
                b_dg += tl.sum(b_ds2, axis=1)
                b_dg -= tl.sum(b_ds2, axis=0)

                b_ds = b_ds.to(b_k.dtype)

                b_dq += tl.dot(b_ds, b_k)

                b_dk += tl.dot(tl.trans(b_ds), b_q)

                p_dg = tl.make_block_ptr(dg_base, (T_len,), (H_dim,), (i_t_real * BT,), (BT,), (0,))

                last_idx_for_dg = tl.minimum(i_t_real * BT + BT, T_len) - 1
                b_dg = tl.where(o_t < last_idx_for_dg, b_dg, b_dg + b_dg_last)

                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

            elif USE_G_GAMMA:
                b_dq = b_dq * tl.exp(b_g)[:, None] * scale

                b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last_val), 0)[:, None]

                b_ds = tl.where(m_A, b_ds * tl.exp(b_g[:, None] - b_g[None, :]), 0) * scale

                b_ds = b_ds.to(b_k.dtype)

                b_dq += tl.dot(b_ds, b_k)

                b_dk += tl.dot(tl.trans(b_ds), b_q)

                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            else:
                b_ds = tl.where(m_A, b_ds, 0)
                b_ds = b_ds.to(b_k.dtype)

                b_dq += tl.dot(b_ds, b_k)

                b_dk += tl.dot(tl.trans(b_ds), b_q) * scale

                b_dq *= scale

                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({})],
    key=["H", "K", "V", "BT", "BK", "BV", "USE_G", "USE_G_GAMMA"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dv(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    dh,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)

    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G or USE_G_GAMMA:
        b_A = tl.where(m_A, b_A * exp(b_g[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
        b_dv *= tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
    else:
        b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)
    p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[
#         triton.Config({})
#     ],
#     key=["H", "K", "V", "BT", "BK", "BV", "USE_G"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_dim: tl.int32,
    B: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_bh = B * H
    total_tasks = NT_dim * num_bh

    for task_id in range(pid, total_tasks, num_programs):
        i_t_idx = task_id % NT_dim
        i_bh = task_id // NT_dim

        i_b, i_h = i_bh // H, i_bh % H
        i_hk = i_h // (H // HK)

        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_idx * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_idx * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_len = eos - bos
            NT = tl.cdiv(T_len, BT)
            i_t_real = i_t
        else:
            T_len = T
            NT = tl.cdiv(T_len, BT)
            bos, eos = i_b * T, i_b * T + T
            i_t_real = i_t_idx

        if i_t_real < NT:
            q_base = q + (bos * H + i_h) * K
            k_base = k + (bos * HK + i_hk) * K
            do_base = do + (bos * H + i_h) * V
            dv_base = dv + (bos * H + i_h) * V

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(k_base, (T_len, K), (HK * K, 1), (i_t_real * BT, i_k * BK), (BT, BK), (1, 0))

                p_q = tl.make_block_ptr(q_base, (K, T_len), (1, H * K), (i_k * BK, i_t_real * BT), (BK, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))

                b_A += tl.dot(b_k, b_q)

            if USE_G:
                g_base = g + bos * HK + i_hk
                p_g = tl.make_block_ptr(g_base, (T_len,), (HK,), (i_t_real * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))

            if USE_G_GAMMA:
                b_gamma = tl.load(g_gamma + i_hk)

                b_g = b_gamma * (tl.arange(0, BT) + 1)

            o_t = i_t_real * BT + tl.arange(0, BT)
            m_t = o_t < T_len

            m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)

            if USE_G or USE_G_GAMMA:
                if USE_G:
                    b_A = tl.where(m_A, b_A * tl.exp(b_g[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
                else:
                    b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)
            else:
                b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)

            for i_v in range(tl.cdiv(V, BV)):
                p_do = tl.make_block_ptr(do_base, (T_len, V), (H * V, 1), (i_t_real * BT, i_v * BV), (BT, BV), (1, 0))
                p_dv = tl.make_block_ptr(dv_base, (T_len, V), (H * V, 1), (i_t_real * BT, i_v * BV), (BT, BV), (1, 0))

                b_do = tl.load(p_do, boundary_check=(0, 1))

                b_dv = tl.dot(b_A.to(b_do.dtype), b_do)

                tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V, HK = *q.shape, v.shape[-1], k.shape[2]

    #
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    NT_dim = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K**-0.5

    o = torch.empty([B, T, H, V], device=v.device, dtype=v.dtype)

    BV = 64
    NV = triton.cdiv(V, BV)
    num_bh = B * H

    total_tasks = NV * NT_dim * num_bh

    num_cores = get_num_cores()
    grid = (num_cores,)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HK=HK,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_dim=NT_dim,
        B=B,
        NV=NV,
    )
    return o


def chunk_bwd_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dv = torch.zeros_like(do)
    grid = (NV, NT, B * H)
    chunk_bwd_kernel_dv[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dv=dv,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dv


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    scale: float = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_size: int = 16,
) -> torch.Tensor:
    B, T, H, K, V, HK = *q.shape, do.shape[-1], k.shape[2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    NT_dim = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if scale is None:
        scale = K**-0.5

    dv = torch.zeros_like(do)

    num_bh = B * H
    total_tasks = NT_dim * num_bh

    num_cores = get_num_cores()
    grid = (num_cores,)

    chunk_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HK=HK,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_dim=NT_dim,
        B=B,
    )

    torch.save(
        {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "g": g.detach().cpu() if g is not None else None,
            "do": do.detach().cpu(),
            "dv": dv.detach().cpu(),
        },
        "mojo_chunk_bwd_inputs.pt",
    )

    return dv


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HK = *q.shape, v.shape[-1], k.shape[2]

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT_dim = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK_dim = triton.cdiv(K, BK)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(q)

    dg = torch.empty([NK_dim, B, T, H], dtype=torch.float32, device=q.device) if g is not None else None
    dw = torch.zeros_like(q) if w is not None else None

    num_bh = B * H
    total_tasks = NK_dim * NT_dim * num_bh

    num_cores = get_num_cores()
    grid = (num_cores,)

    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh=dh,
        dv=dv,
        w=w,
        dw=dw,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H_dim=H,
        HK=HK,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_DW=dw is not None,
        IS_VARLEN=cu_seqlens is not None,
        NK_dim=NK_dim,
        NT_dim=NT_dim,
    )

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg
