# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, Hongmin Chen

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.ascend.utils import exp
from mojo_opset.backends.ttx.kernels.ascend.utils import get_num_cores


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
# @triton.autotune(
#     configs=[
#         triton.Config({"BK": BK})
#         for BK in [32, 64, 128]
#     ],
#     key=["H", "K", "BT", "IS_VARLEN"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
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
            o_t = i_t_real * BT + tl.arange(0, BT)
            m_t = o_t < T_len

            p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T_len,), (H,), (i_t_real * BT,), (BT,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k + (bos * H + i_h) * K,
                    (T_len, K),
                    (H * K, 1),
                    (i_t_real * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))

                b_A += tl.dot(b_k, tl.trans(b_k))

            if USE_G:
                p_g = tl.make_block_ptr(g + bos * H + i_h, (T_len,), (H,), (i_t_real * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_diff = b_g[:, None] - b_g[None, :]
                b_A *= tl.exp(b_g_diff)

            b_A *= b_beta[:, None]

            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
            b_A = tl.where(m_A, b_A, 0)

            p_A = tl.make_block_ptr(
                A + (bos * H + i_h) * BT, (T_len, BT), (BT * H, 1), (i_t_real * BT, 0), (BT, BT), (1, 0)
            )
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({"BK": BK}) for BK in [32, 64]],
    key=["BC"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        b_kt = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
        # [BC, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        # [BK, BC]
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1)) * exp(b_gn[:, None] - b_gk)
        # [BC, BC]
        b_A += tl.dot(b_k, b_kt)
    b_A *= b_beta[:, None]

    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_beta = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h

    b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_beta, mask=m_A, other=0)[:, None]
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_k * b_kt[None, :] * exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A, 0.0)

        tl.store(A + o_A + j, b_A, mask=m_A)
        p_kt += H * K
        p_gk += H * K


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    chunk_size: int = 32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B, T, H, K = k.shape
    BT = chunk_size

    NT_dim = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    num_bh = B * H

    if gk is None:
        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

        num_cores = get_num_cores()
        grid = (num_cores,)

        chunk_scaled_dot_kkt_fwd_kernel[grid](
            k=k,
            g=g,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BK=16,
            IS_VARLEN=cu_seqlens is not None,
            USE_G=g is not None,
            NT_dim=NT_dim,
            B=B,
        )
        return A
    raise NotImplementedError("gk is not None dont't support persistent mode now, please report a issue.")
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, NC * NC, B * H)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )

    grid = (NT, NC, B * H)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        k=k,
        g=gk,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
    )
    return A
