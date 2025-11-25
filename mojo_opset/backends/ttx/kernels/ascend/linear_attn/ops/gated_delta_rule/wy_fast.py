# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, Hongmin Chen

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.ascend.utils import get_num_cores


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    dbeta,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    H_dim: tl.constexpr,  #
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    B: tl.int32,
    NT_dim: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_bh = B * H_dim
    total_tasks = NT_dim * num_bh

    for task_id in range(pid, total_tasks, num_programs):
        i_bh = task_id // NT_dim
        i_t_idx = task_id % NT_dim

        i_b, i_h = i_bh // H_dim, i_bh % H_dim
        i_hk = i_h // (H_dim // HK)

        if IS_VARLEN:
            i_n, i_t_real = (
                tl.load(chunk_indices + i_t_idx * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_idx * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_len = eos - bos
            NT = tl.cdiv(T_len, BT)
        else:
            T_len = T
            bos, eos = i_b * T, i_b * T + T
            i_t_real = i_t_idx
            NT = tl.cdiv(T_len, BT)

        if i_t_real < NT:
            beta_base = beta + (bos * HK + i_hk)
            g_base = g + (bos * HK + i_hk)
            A_base = A + (bos * HK + i_hk) * BT
            k_base = k + (bos * HK + i_hk) * K
            v_base = v + (bos * HK + i_hk) * V
            dw_base = dw + (bos * H_dim + i_h) * K
            du_base = du + (bos * H_dim + i_h) * V
            dk_base = dk + (bos * H_dim + i_h) * K
            dv_base = dv + (bos * H_dim + i_h) * V
            dbeta_base = dbeta + (bos * H_dim + i_h)
            dg_base = dg + (bos * H_dim + i_h)

            p_beta = tl.make_block_ptr(beta_base, (T_len,), (HK,), (i_t_real * BT,), (BT,), (0,))
            p_g = tl.make_block_ptr(g_base, (T_len,), (HK,), (i_t_real * BT,), (BT,), (0,))

            p_A = tl.make_block_ptr(
                A_base,
                (BT, T_len),
                (1, HK * BT),
                (0, i_t_real * BT),
                (BT, BT),
                (0, 1),
            )

            b_A = tl.load(p_A, boundary_check=(0, 1))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = tl.exp(b_g)

            b_dbeta = tl.zeros([BT], dtype=tl.float32)
            b_dA = tl.zeros([BT, BT], dtype=tl.float32)
            b_dg = tl.zeros([BT], dtype=tl.float32)

            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k_base,
                    (T_len, K),
                    (HK * K, 1),
                    (i_t_real * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                p_dk = tl.make_block_ptr(
                    dk_base,
                    (T_len, K),
                    (H_dim * K, 1),
                    (i_t_real * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                p_dw = tl.make_block_ptr(
                    dw_base,
                    (T_len, K),
                    (H_dim * K, 1),
                    (i_t_real * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))

                b_k_beta_g = (b_k * b_beta[:, None] * b_g_exp[:, None]).to(b_k.dtype)
                b_dw = tl.load(p_dw, boundary_check=(0, 1))

                b_dA += tl.dot(b_dw, tl.trans(b_k_beta_g))

                b_dk_beta_g = tl.dot(b_A, b_dw)

                b_dk = b_dk_beta_g * b_beta[:, None] * b_g_exp[:, None]

                b_dbeta += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None], 1)

                b_dg += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None] * b_beta[:, None], 1)

                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            for i_v in range(tl.cdiv(V, BV)):
                p_v = tl.make_block_ptr(
                    v_base,
                    (T_len, V),
                    (HK * V, 1),
                    (i_t_real * BT, i_v * BV),
                    (BT, BV),
                    (1, 0),
                )
                p_dv = tl.make_block_ptr(
                    dv_base,
                    (T_len, V),
                    (H_dim * V, 1),
                    (i_t_real * BT, i_v * BV),
                    (BT, BV),
                    (1, 0),
                )
                p_du = tl.make_block_ptr(
                    du_base,
                    (T_len, V),
                    (H_dim * V, 1),
                    (i_t_real * BT, i_v * BV),
                    (BT, BV),
                    (1, 0),
                )
                b_v = tl.load(p_v, boundary_check=(0, 1))

                b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
                b_du = tl.load(p_du, boundary_check=(0, 1))

                b_dA += tl.dot(b_du, tl.trans(b_v_beta))

                b_dv_beta = tl.dot(b_A, b_du)

                b_dv = b_dv_beta * b_beta[:, None]

                b_dbeta += tl.sum(b_dv_beta * b_v, 1)

                tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

            o_t = i_t_real * BT + tl.arange(0, BT)
            m_t = o_t < T_len

            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

            b_dA = tl.where(m_A, b_dA, 0)

            b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)

            b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

            b_dA = tl.where(m_A, -b_dA * tl.exp(b_g[:, None] - b_g[None, :]), 0)
            b_dA = b_dA.to(k.dtype.element_ty)

            b_A = tl.zeros([BT, BT], dtype=tl.float32)

            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k_base,
                    (T_len, K),
                    (HK * K, 1),
                    (i_t_real * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                p_dk = tl.make_block_ptr(
                    dk_base,
                    (T_len, K),
                    (H_dim * K, 1),
                    (i_t_real * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dk = tl.load(p_dk, boundary_check=(0, 1))

                b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

                b_A += tl.dot(b_k_beta, tl.trans(b_k))

                b_dk_beta = tl.dot(b_dA, b_k)

                b_dbeta += tl.sum(b_dk_beta * b_k, 1)

                b_dk += tl.dot(tl.trans(b_dA), b_k_beta)

                b_dk += b_dk_beta * b_beta[:, None]

                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            b_dA_A = b_dA * b_A

            b_dg += tl.sum(b_dA_A, axis=1) - tl.sum(b_dA_A, axis=0)

            p_dg = tl.make_block_ptr(dg_base, (T_len,), (H_dim,), (i_t_real * BT,), (BT,), (0,))
            p_dbeta = tl.make_block_ptr(dbeta_base, (T_len,), (H_dim,), (i_t_real * BT,), (BT,), (0,))

            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
            tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[
#         triton.Config({})
#     ],
#     key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
# )
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT: tl.int32,
    B: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_bh = B * H
    total_tasks = NT * num_bh

    for task_id in range(pid, total_tasks, num_programs):
        i_t = task_id % NT
        i_bh = task_id // NT

        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t_real = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_len = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            T_len = T
            i_t_real = i_t

        p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T_len,), (H,), (i_t_real * BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        p_A = tl.make_block_ptr(
            A + (bos * H + i_h) * BT, (T_len, BT), (H * BT, 1), (i_t_real * BT, 0), (BT, BT), (1, 0)
        )
        b_A = tl.load(p_A, boundary_check=(0, 1))

        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (T_len, V),
                (H * V, 1),
                (i_t_real * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_u = tl.make_block_ptr(
                u + (bos * H + i_h) * V,
                (T_len, V),
                (H * V, 1),
                (i_t_real * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))

            b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)

            b_u = tl.dot(b_A, b_vb, allow_tf32=False)
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            p_g = tl.make_block_ptr(g + (bos * H + i_h), (T_len,), (H,), (i_t_real * BT,), (BT,), (0,))
            b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h) * K,
                (T_len, K),
                (H * K, 1),
                (i_t_real * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_w = tl.make_block_ptr(
                w + (bos * H + i_h) * K,
                (T_len, K),
                (H * K, 1),
                (i_t_real * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))

            b_kb = b_k * b_beta[:, None]

            if USE_G:
                b_kb *= b_g[:, None]
            if USE_GK:
                p_gk = tl.make_block_ptr(
                    gk + (bos * H + i_h) * K,
                    (T_len, K),
                    (H * K, 1),
                    (i_t_real * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                b_kb *= tl.exp(tl.load(p_gk, boundary_check=(0, 1)))

            b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    #
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.zeros_like(k)
    u = torch.zeros_like(v)

    num_cores = get_num_cores()
    grid = (num_cores,)

    recompute_w_u_fwd_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_GK=gk is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT=NT,
        B=B,
    )
    return w, u


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_indices: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HK = *dw.shape, v.shape[-1], k.shape[2]

    #
    BT = 16
    NT_dim = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk = torch.empty([B, T, H, K], device=k.device, dtype=k.dtype)
    dv = torch.empty([B, T, H, V], device=v.device, dtype=v.dtype)
    dbeta = torch.empty([B, T, H], device=beta.device, dtype=beta.dtype)
    dg = torch.empty([B, T, H], device=g.device, dtype=g.dtype)

    num_bh = B * H
    total_tasks = NT_dim * num_bh

    num_cores = get_num_cores()
    grid = (num_cores,)

    prepare_wy_repr_bwd_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        dbeta=dbeta,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H_dim=H,
        HK=HK,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        B=B,
        NT_dim=NT_dim,
    )
    return dk, dv, dbeta, dg


bwd_prepare_wy_repr = prepare_wy_repr_bwd

fwd_recompute_w_u = recompute_w_u_fwd
