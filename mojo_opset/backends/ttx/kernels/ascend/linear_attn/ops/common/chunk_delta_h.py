# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, Hongmin Chen

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.ascend.utils import get_num_cores


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[
#         triton.Config({"BV": BV})
#         for BV in [32, 64]
#     ],
#     key=["H", "K", "V", "BT"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    #
    NT_dim: tl.int32,
    N: tl.int32,
    NV: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_nh = N * H
    total_tasks = NV * num_nh

    for task_id in range(pid, total_tasks, num_programs):
        i_nh = task_id % num_nh
        i_v = task_id // num_nh

        i_n, i_h = i_nh // H, i_nh % H

        if IS_VARLEN:
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_len = eos - bos
            NT = tl.cdiv(T_len, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        else:
            bos, eos = i_n * T, i_n * T + T
            T_len = T
            NT = tl.cdiv(T_len, BT)
            boh = i_n * NT

        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

        h_base = h + (boh * H + i_h) * K * V
        v_base = v + (bos * H + i_h) * V
        k_base = k + (bos * H + i_h) * K
        w_base = w + (bos * H + i_h) * K
        if SAVE_NEW_VALUE:
            v_new_base = v_new + (bos * H + i_h) * V
        stride_v = H * V
        stride_h = H * K * V
        stride_k = H * K

        if USE_INITIAL_STATE:
            h0_ptr = h0 + i_nh * K * V
        if STORE_FINAL_STATE:
            ht_ptr = ht + i_nh * K * V

        if USE_INITIAL_STATE:
            p_h0_1 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_h0_2 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_h0_3 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_h0_4 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

        for i_t in range(NT):
            p_h1 = tl.make_block_ptr(h_base + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(h_base + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(h_base + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(h_base + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

            b_v = tl.zeros([BT, BV], dtype=tl.float32)

            p_w = tl.make_block_ptr(w_base, (T_len, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w_base, (T_len, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w_base, (T_len, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w_base, (T_len, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))

            p_v = tl.make_block_ptr(v_base, (T_len, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

            if SAVE_NEW_VALUE:
                p_v_new = tl.make_block_ptr(
                    v_new_base, (T_len, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
                )
                tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

            last_idx = tl.minimum((i_t + 1) * BT, T_len) - 1

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T_len
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T_len,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))

                b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]

                b_g_last = tl.exp(b_g_last)

                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            if USE_GK:
                o_k1 = tl.arange(0, 64)

                b_gk_last1 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                    mask=(o_k1 < K),
                    other=0.0,
                )
                b_h1 *= tl.exp(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                        mask=(o_k2 < K),
                        other=0.0,
                    )
                    b_h2 *= tl.exp(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                        mask=(o_k3 < K),
                        other=0.0,
                    )
                    b_h3 *= tl.exp(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                        mask=(o_k4 < K),
                        other=0.0,
                    )
                    b_h4 *= tl.exp(b_gk_last4)[:, None]

            b_v = b_v.to(k.dtype.element_ty)

            p_k = tl.make_block_ptr(k_base, (K, T_len), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            if USE_GK:
                p_g = tl.make_block_ptr(
                    gk + (bos * H + i_h) * K,
                    (K, T_len),
                    (1, H * K),
                    (0, i_t * BT),
                    (64, BT),
                    (0, 1),
                )

                b_k = (b_k * tl.exp(b_gk_last1[:, None] - tl.load(p_g, boundary_check=(0, 1)))).to(b_k.dtype)

            b_h1 += tl.dot(b_k, b_v)

            if K > 64:
                p_k = tl.make_block_ptr(k_base, (K, T_len), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    p_g = tl.make_block_ptr(
                        gk + (bos * H + i_h) * K,
                        (K, T_len),
                        (1, H * K),
                        (64, i_t * BT),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = (b_k * tl.exp(b_gk_last2[:, None] - tl.load(p_g, boundary_check=(0, 1)))).to(b_k.dtype)
                b_h2 += tl.dot(b_k, b_v)

            if K > 128:
                p_k = tl.make_block_ptr(k_base, (K, T_len), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    p_g = tl.make_block_ptr(
                        gk + (bos * H + i_h) * K,
                        (K, T_len),
                        (1, H * K),
                        (128, i_t * BT),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = (b_k * tl.exp(b_gk_last3[:, None] - tl.load(p_g, boundary_check=(0, 1)))).to(b_k.dtype)
                b_h3 += tl.dot(b_k, b_v)

            if K > 192:
                p_k = tl.make_block_ptr(k_base, (K, T_len), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    p_g = tl.make_block_ptr(
                        gk + (bos * H + i_h) * K,
                        (K, T_len),
                        (1, H * K),
                        (192, i_t * BT),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = (b_k * tl.exp(b_gk_last4[:, None] - tl.load(p_g, boundary_check=(0, 1)))).to(b_k.dtype)
                b_h4 += tl.dot(b_k, b_v)

        if STORE_FINAL_STATE:
            p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[
#         triton.Config({"BV": BV})
#         for BV in [64, 32]
#     ],
#     key=["H", "K", "V", "BT", "BV", "USE_G"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H_dim: tl.constexpr,  #
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    N: tl.int32,
    NT_dim: tl.int32,
    NV: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_nh = N * H_dim
    total_tasks = NV * num_nh

    for task_id in range(pid, total_tasks, num_programs):
        i_nh = task_id % num_nh
        i_v = task_id // num_nh

        i_n, i_h = i_nh // H_dim, i_nh % H_dim
        i_hk = i_h // (H_dim // HK)

        if IS_VARLEN:
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_len = eos - bos
            NT = tl.cdiv(T_len, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        else:
            T_len = T
            bos, eos = i_n * T, i_n * T + T
            NT = tl.cdiv(T_len, BT)
            boh = i_n * NT

        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

        dh_base = dh + (boh * H_dim + i_h) * K * V
        dv_base = dv + (bos * H_dim + i_h) * V
        dv2_base = dv2 + (bos * H_dim + i_h) * V
        q_base = q + (bos * H_dim + i_h) * K
        k_base = k + (bos * HK + i_hk) * K
        w_base = w + (bos * HK + i_hk) * K
        do_base = do + (bos * H_dim + i_h) * V
        stride_v = H_dim * V
        stride_h = H_dim * K * V
        stride_k = HK * K
        stride_q = H_dim * K

        if USE_INITIAL_STATE:
            dh0_ptr = dh0 + i_nh * K * V
        if USE_FINAL_STATE_GRADIENT:
            dht_ptr = dht + i_nh * K * V

        if USE_FINAL_STATE_GRADIENT:
            p_dht1 = tl.make_block_ptr(dht_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
            b_dh1 = (b_dh1 + b_dh1) / 2

            if K > 64:
                p_dht2 = tl.make_block_ptr(dht_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
                b_dh2 = (b_dh2 + b_dh2) / 2

            if K > 128:
                p_dht3 = tl.make_block_ptr(dht_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
                b_dh3 = (b_dh3 + b_dh3) / 2

            if K > 192:
                p_dht4 = tl.make_block_ptr(dht_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))
                b_dh4 = (b_dh4 + b_dh4) / 2

        for i_t in range(NT - 1, -1, -1):
            p_dh1 = tl.make_block_ptr(dh_base + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_dh2 = tl.make_block_ptr(dh_base + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_dh3 = tl.make_block_ptr(dh_base + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_dh4 = tl.make_block_ptr(dh_base + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

            if USE_G:
                last_idx = tl.minimum((i_t + 1) * BT, T_len) - 1
                bg_last = tl.load(g + (bos + last_idx) * HK + i_hk)
                bg_last_exp = tl.exp(bg_last)
                p_g = tl.make_block_ptr(g + bos * HK + i_hk, (T_len,), (HK,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_exp = tl.exp(b_g)
            else:
                bg_last = None
                last_idx = None
                b_g = None
                b_g_exp = None

            p_dv = tl.make_block_ptr(dv_base, (T_len, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do_base, (T_len, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv2 = tl.make_block_ptr(dv2_base, (T_len, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.zeros([BT, BV], dtype=tl.float32)

            p_k = tl.make_block_ptr(k_base, (T_len, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = tl.make_block_ptr(k_base, (T_len, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = tl.make_block_ptr(k_base, (T_len, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = tl.make_block_ptr(k_base, (T_len, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T_len

                b_dv *= tl.where(m_t, tl.exp(bg_last - b_g), 0)[:, None]

            b_dv += tl.load(p_dv, boundary_check=(0, 1))

            tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

            p_w = tl.make_block_ptr(w_base, (K, T_len), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q_base, (K, T_len), (1, stride_q), (0, i_t * BT), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))

            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]

            b_q = (b_q * scale).to(b_q.dtype)

            b_dh1 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = tl.make_block_ptr(q_base, (K, T_len), (1, stride_q), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w_base, (K, T_len), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                b_q = (b_q * scale).to(b_q.dtype)
                b_dh2 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = tl.make_block_ptr(q_base, (K, T_len), (1, stride_q), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w_base, (K, T_len), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                b_q = (b_q * scale).to(b_q.dtype)
                b_dh3 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = tl.make_block_ptr(q_base, (K, T_len), (1, stride_q), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w_base, (K, T_len), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                b_q = (b_q * scale).to(b_q.dtype)
                b_dh4 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

        if USE_INITIAL_STATE:
            p_dh0 = tl.make_block_ptr(dh0_ptr, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))

            if K > 64:
                p_dh1 = tl.make_block_ptr(dh0_ptr, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))

            if K > 128:
                p_dh2 = tl.make_block_ptr(dh0_ptr, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))

            if K > 192:
                p_dh3 = tl.make_block_ptr(dh0_ptr, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size
    BV = 64

    if cu_seqlens is None:
        N, NT_dim = B, triton.cdiv(T, BT)
    else:
        N, NT_dim = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT_dim, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.zeros_like(u) if save_new_value else None

    NV = triton.cdiv(V, BV)
    num_nh = N * H

    num_cores = get_num_cores()
    grid = (num_cores,)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        NT_dim=NT_dim,
        N=N,
        NV=NV,
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HK = *q.shape, do.shape[-1], k.shape[2]

    BT = 16
    BV = 64

    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if cu_seqlens is None:
        N, NT_dim = B, triton.cdiv(T, BT)
    else:
        N, NT_dim = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
        )

    dh = q.new_empty(B, NT_dim, H, K, V)
    dh0 = torch.zeros_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)

    NV = triton.cdiv(V, BV)
    num_nh = N * H
    total_tasks = NV * num_nh

    num_cores = get_num_cores()
    grid = (num_cores,)

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H_dim=H,
        HK=HK,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_G=g is not None,
        USE_INITIAL_STATE=h0 is not None,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        IS_VARLEN=cu_seqlens is not None,
        N=N,
        NT_dim=NT_dim,
        NV=NV,
    )
    return dh, dh0, dv2
