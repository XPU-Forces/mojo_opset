# Copyright 2025 Xunhao Lai & Jianqiao Lu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

from typing import Any
from typing import Optional

import torch
import triton
import triton.language as tl

import torch_npu


@triton.jit
def forward_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x h x d
    v_ptr,  # V: n x h x d
    o_ptr,  # O: n x h x d
    lse_ptr,  # LSE: h x n
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    qk_head_dim,
    v_head_dim,
    # sm_scale
    sm_scale,
    # causal
    causal: tl.constexpr,
    # gqa
    gqa_interleave,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, qk_head_dim),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    # k_ptrs = tl.make_block_ptr(
    #     base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
    #     shape=(qk_head_dim, k_len),
    #     strides=(stride_kd, stride_kn),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_SIZE_KD, BLOCK_SIZE_K),
    #     order=(0, 1),
    # )
    # v_ptrs = tl.make_block_ptr(
    #     base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
    #     shape=(k_len, v_head_dim),
    #     strides=(stride_vn, stride_vd),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_VD),
    #     order=(1, 0),
    # )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_k = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    # full attention or causal attention
    lo = 0
    if causal:
        hi = min(k_len, (pid_q + 1) * BLOCK_SIZE_Q)
    else:
        hi = k_len
    for i in range(lo, hi, BLOCK_SIZE_K):
        k_ptrs = tl.make_block_ptr(
            base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
            shape=(qk_head_dim, k_len),
            strides=(stride_kd, stride_kn),
            offsets=(0, i),
            block_shape=(BLOCK_SIZE_KD, BLOCK_SIZE_K),
            order=(0, 1),
        )
        v_ptrs = tl.make_block_ptr(
            base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
            shape=(k_len, v_head_dim),
            strides=(stride_vn, stride_vd),
            offsets=(i, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_VD),
            order=(1, 0),
        )
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        if causal:
            qk += tl.where(off_q[:, None] >= (i + off_k)[None, :], 0, float("-inf"))
        else:
            qk += tl.where((off_k < k_len - i)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * qk_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        # k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        # v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    # final scale
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    # save output
    # o_ptrs = tl.make_block_ptr(
    #     base=o_ptr + q_start * stride_on + pid_h * stride_oh,
    #     shape=(q_len, v_head_dim),
    #     strides=(stride_on, stride_od),
    #     offsets=(pid_q * BLOCK_SIZE_Q, 0),
    #     block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
    #     order=(1, 0),
    # )
    # tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    Q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    D = tl.arange(0, BLOCK_SIZE_VD)[None, :]
    mask = (Q < q_len) & (D < v_head_dim)
    ptr = o_ptr + q_start * stride_on + pid_h * stride_oh + Q * stride_on + D * stride_od
    tl.store(ptr, acc_o.to(o_ptr.dtype.element_ty), mask=mask)
    # save lse
    l_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh + off_q * stride_ln
    tl.store(l_ptrs, lse_i, mask=off_q < q_len)


@triton.jit
def backward_sum_o_do(
    o_ptr,  # O: n x h x d
    do_ptr,  # dO: n x h x d
    delta_ptr,  # D: h x n
    o_len,
    HEAD_DIM,
    stride_on,
    stride_oh,
    stride_od,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dh,
    stride_dn,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_n = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o = tl.load(
        o_ptr + off_n[:, None] * stride_on + pid_h * stride_oh + off_d[None, :] * stride_od,
        mask=(off_n[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + off_n[:, None] * stride_don + pid_h * stride_doh + off_d[None, :] * stride_dod,
        mask=(off_n[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + pid_h * stride_dh + off_n * stride_dn, delta, mask=off_n < o_len)


@triton.jit
def backward_dkdv(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: sh x n x kh x d
    dv_ptr,  # DV: sh x n x kh x d
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    qk_head_dim,
    v_head_dim,
    # sm_scale
    sm_scale,
    # causal
    causal: tl.constexpr,
    # gqa
    gqa_interleave,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dks,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvs,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if gqa_interleave:
        pid_kh = pid_h % NUM_SHARE_Q_HEADS
        pid_sh = pid_h // NUM_SHARE_Q_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
        pid_sh = pid_h % NUM_SHARE_Q_HEADS
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_K * pid_k >= k_len:
        return
    # init pointers
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, qk_head_dim),
        strides=(stride_kn, stride_kd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    # dk_ptrs = tl.make_block_ptr(
    #     base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh + pid_sh * stride_dks,
    #     shape=(k_len, qk_head_dim),
    #     strides=(stride_dkn, stride_dkd),
    #     offsets=(pid_k * BLOCK_SIZE_K, 0),
    #     block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_KD),
    #     order=(1, 0),
    # )
    Q = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    D = tl.arange(0, BLOCK_SIZE_KD)[None, :]
    mask = (Q < k_len) & (D < qk_head_dim)
    ptr = dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh + pid_sh * stride_dks + Q * stride_dkn + D * stride_dkd
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, v_head_dim),
        strides=(stride_vn, stride_vd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_VD),
        order=(1, 0),
    )
    # dv_ptrs = tl.make_block_ptr(
    #     base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh + pid_sh * stride_dvs,
    #     shape=(k_len, v_head_dim),
    #     strides=(stride_dvn, stride_dvd),
    #     offsets=(pid_k * BLOCK_SIZE_K, 0),
    #     block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_VD),
    #     order=(1, 0),
    # )
    Q1 = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    D1 = tl.arange(0, BLOCK_SIZE_VD)[None, :]
    mask1 = (Q1 < k_len) & (D1 < v_head_dim)
    ptr1 = dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh + pid_sh * stride_dvs + Q1 * stride_dvn + D1 * stride_dvd
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
    # load k v and keep in SRAM
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dk dv
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_KD), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_VD), dtype=tl.float32)
    # causal
    if causal:
        q_lo = pid_k * BLOCK_SIZE_K
    else:
        q_lo = 0
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, qk_head_dim),
        strides=(stride_qn, stride_qd),
        offsets=(q_lo, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    do_ptrs = tl.make_block_ptr(
        base=do_ptr + q_start * stride_don + pid_h * stride_doh,
        shape=(q_len, v_head_dim),
        strides=(stride_don, stride_dod),
        offsets=(q_lo, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
        order=(1, 0),
    )
    d_ptrs = tl.make_block_ptr(
        base=d_ptr + q_start * stride_dn + pid_h * stride_dh,
        # shape=(q_len, 1),
        # strides=(stride_dn, 1),
        # offsets=(q_lo, 0),
        # block_shape=(BLOCK_SIZE_Q, 1),
        # order=(0, 1),
        shape=(q_len,),
        strides=(stride_dn,),
        offsets=(q_lo,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
        # shape=(q_len, 1),
        # strides=(stride_ln, 1),
        # offsets=(q_lo, 0),
        # block_shape=(BLOCK_SIZE_Q, 1),
        # order=(0, 1),
        shape=(q_len,),
        strides=(stride_ln,),
        offsets=(q_lo,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )
    # loop for q blocks
    for i in range(q_lo, q_len, BLOCK_SIZE_Q):
        # load
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
        # lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        # d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0,), padding_option="zero")[:, None]
        d = tl.load(d_ptrs, boundary_check=(0,), padding_option="zero")[:, None]
        # compute qk
        if causal:
            qk = tl.where((off_q + i)[:, None] >= off_k[None, :], float(0.0), float("-inf"))
        else:
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        k_t = tl.trans(k)
        qk += tl.dot(q, k_t) * qk_scale
        # compute p, ds
        p = tl.math.exp2(qk - lse)
        v_t = tl.trans(v)
        dp = tl.dot(do, v_t)
        ds = sm_scale * p * (dp - d)
        # cast dtype
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        # update dk and dv
        ds_t = tl.trans(ds)
        dk += tl.dot(ds_t, q)
        p_t = tl.trans(p)
        dv += tl.dot(p_t, do)
        # increment pointers
        q_ptrs = tl.advance(q_ptrs, (BLOCK_SIZE_Q, 0))
        do_ptrs = tl.advance(do_ptrs, (BLOCK_SIZE_Q, 0))
        # lse_ptrs = tl.advance(lse_ptrs, (BLOCK_SIZE_Q, 0))
        # d_ptrs = tl.advance(d_ptrs, (BLOCK_SIZE_Q, 0))
        lse_ptrs = tl.advance(lse_ptrs, (BLOCK_SIZE_Q,))
        d_ptrs = tl.advance(d_ptrs, (BLOCK_SIZE_Q,))
    # save dk dv
    # tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    # tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(ptr, dk.to(dk_ptr.dtype.element_ty), mask=mask)
    tl.store(ptr1, dv.to(dv_ptr.dtype.element_ty), mask=mask1)


@triton.jit
def backward_dq(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dq_ptr,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    qk_head_dim,
    v_head_dim,
    # sm_scale
    sm_scale,
    # causal
    causal,
    # gqa
    gqa_interleave,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dqn,
    stride_dqh,
    stride_dqd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    # init pointers
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, qk_head_dim),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    # dq_ptrs = tl.make_block_ptr(
    #     base=dq_ptr + q_start * stride_dqn + pid_h * stride_dqh,
    #     shape=(q_len, qk_head_dim),
    #     strides=(stride_dqn, stride_dqd),
    #     offsets=(pid_q * BLOCK_SIZE_Q, 0),
    #     block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
    #     order=(1, 0),
    # )
    Q1 = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    D1 = tl.arange(0, BLOCK_SIZE_KD)[None, :]
    mask1 = (Q1 < q_len) & (D1 < qk_head_dim)
    ptr1 = dq_ptr + q_start * stride_dqn + pid_h * stride_dqh + Q1 * stride_dqn + D1 * stride_dqd
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, qk_head_dim),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, qk_head_dim),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_VD),
        order=(1, 0),
    )
    do_ptrs = tl.make_block_ptr(
        base=do_ptr + q_start * stride_don + pid_h * stride_doh,
        shape=(q_len, qk_head_dim),
        strides=(stride_don, stride_dod),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
        order=(1, 0),
    )
    d_ptrs = tl.make_block_ptr(
        base=d_ptr + q_start * stride_dn + pid_h * stride_dh,
        shape=(q_len, 1),
        strides=(stride_dn, stride_dh),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
        shape=(q_len, 1),
        strides=(stride_ln, stride_lh),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_k = tl.arange(0, BLOCK_SIZE_K)
    # load q, do, lse, delta, and keep in SRAM
    q = tl.load(q_ptrs, boundary_check=(1, 0), padding_option="zero")
    do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
    d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dq
    dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_KD), dtype=tl.float32)
    # causal
    if causal:
        k_hi = (pid_q + 1) * BLOCK_SIZE_Q
    else:
        k_hi = k_len
    for j in range(0, k_hi, BLOCK_SIZE_K):
        # load
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        # compute qk
        if causal:
            qk = tl.where(off_q[:, None] >= (off_k + j)[None, :], float(0.0), float("-inf"))
        else:
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        k_t = tl.trans(k)
        qk += tl.dot(q, k_t) * qk_scale
        # compute p, ds
        p = tl.math.exp2(qk - lse)
        v_t = tl.trans(v)
        dp = tl.dot(do, v_t)
        ds = sm_scale * p * (dp - d)
        # cast dtype
        ds = ds.to(q.dtype)
        # update dq
        dq += tl.dot(ds, k)
        # increment pointers
        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_K, 0))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    # save dq
    # tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(ptr1, dq.to(dq_ptr.dtype.element_ty), mask=mask1)


def _flash_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    sm_scale: float,
    gqa_interleave: bool = False,
):
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # shape
    q_len, num_q_heads, qk_head_dim = q.shape
    k_len, num_k_heads, qk_head_dim = k.shape
    v_len, num_v_heads, v_head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, "head_dim must be less than 256"
    assert q_len == k_len and k_len == v_len
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype, device=q.device)
    lse = torch.empty(num_q_heads, q_len, dtype=torch.float32, device=q.device)
    # launch kernel
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
    )
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_KD = triton.next_power_of_2(qk_head_dim)
    BLOCK_SIZE_VD = triton.next_power_of_2(v_head_dim)
    forward_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        qk_head_dim,
        v_head_dim,
        sm_scale,
        causal,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_VD=BLOCK_SIZE_VD,
    )
    return o, lse


def _flash_attention_bwd(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    sm_scale: float,
    gqa_interleave: bool = False,
):
    q_len, num_q_heads, qk_head_dim = q.shape
    k_len, num_k_heads, qk_head_dim = k.shape
    v_len, num_v_heads, v_head_dim = v.shape
    o_len, num_o_heads, v_head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    # compute D
    delta = torch.empty([num_o_heads, o_len], device=o.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(o_len, META["BLOCK_SIZE_O"]), num_o_heads)
    BLOCK_SIZE_O = 128
    BLOCK_SIZE_VD = triton.next_power_of_2(v_head_dim)
    backward_sum_o_do[grid](
        o,
        do,
        delta,
        o_len,
        v_head_dim,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        BLOCK_SIZE_D=BLOCK_SIZE_VD,
    )
    # compute dk dv
    dk = torch.empty(
        num_share_q_heads,
        k_len,
        num_k_heads,
        qk_head_dim,
        device=k.device,
        dtype=k.dtype,
    )
    dv = torch.empty(
        num_share_q_heads,
        k_len,
        num_k_heads,
        v_head_dim,
        device=k.device,
        dtype=k.dtype,
    )
    batch_size = cu_seqlens_q.shape[0] - 1
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_k, META["BLOCK_SIZE_K"]),
    )
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_KD = triton.next_power_of_2(qk_head_dim)
    BLOCK_SIZE_VD = triton.next_power_of_2(v_head_dim)
    backward_dkdv[grid](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        qk_head_dim,
        v_head_dim,
        sm_scale,
        causal,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_VD=BLOCK_SIZE_VD,
    )
    dk = dk.sum(0)
    dv = dv.sum(0)
    # compute dq
    dq = torch.empty_like(q)
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
    )
    BLOCK_SIZE_Q = 64 if max(qk_head_dim, v_head_dim) > 128 else 128
    BLOCK_SIZE_K = 32
    backward_dq[grid](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dq,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        qk_head_dim,
        v_head_dim,
        sm_scale,
        causal,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_KD=BLOCK_SIZE_KD,
        BLOCK_SIZE_VD=BLOCK_SIZE_VD,
    )
    return dq, dk, dv


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal=True,
        sm_scale=None,
        gqa_interleave=False,
    ):
        # softmax scale
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])
        o, lse = _flash_attention_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            sm_scale,
            gqa_interleave,
        )
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.gqa_interleave = gqa_interleave
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) -> Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        gqa_interleave = ctx.gqa_interleave
        dq, dk, dv = _flash_attention_bwd(
            o,
            do,
            lse,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            sm_scale,
            gqa_interleave,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """Flash attention with variable length based on triton.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_q_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_q_heads, head_dim]
        cu_seqlens_q (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        cu_seqlens_k (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_k in flash_attn_func_varlen.
        max_seqlen_q (int): max q len of the batch.
        max_seqlen_k (int): max k len of the batch.
        causal (bool, optional): Causal mask. Defaults to False.
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).
        gqa_interleave (bool, optional): GQA pattern. Defaults to False, use Llama style GQA.

    Returns:
        torch.Tensor: attention output with shape [total_q_len, num_q_heads, head_dim]
    """
    return FlashAttention.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        sm_scale,
        gqa_interleave,
    )


if __name__ == "__main__":
    for causal in [False, True]:
        # triton flash attention
        torch.manual_seed(42)
        q = torch.randn(1000, 32, 128, dtype=torch.float16, device="npu", requires_grad=True)
        k = torch.randn(1000, 16, 128, dtype=torch.float16, device="npu", requires_grad=True)
        v = torch.randn(1000, 16, 128, dtype=torch.float16, device="npu", requires_grad=True)
        cu_seqlens_q = torch.Tensor([0, 100, 384, 1000]).npu().to(torch.int32)
        cu_seqlens_k = torch.Tensor([0, 100, 384, 1000]).npu().to(torch.int32)
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max()
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()
        head_num = q.shape[1]
        if causal:
            atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to("npu")
            o = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                pse=None,
                padding_mask=None,
                atten_mask=atten_mask_npu,
                scale=1.0 / math.sqrt(q.shape[-1]),
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
                sparse_mode=3,
            )[0]
        else:
            o = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num,
                pse=None,
                atten_mask=None,
                scale=1.0 / math.sqrt(q.shape[-1]),
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            )[0]

        randn = torch.randn_like(o)
        loss = (o * randn).sum()
        loss.backward()

        # flash attention
        torch.manual_seed(42)
        q1 = q.clone().detach().requires_grad_()
        k1 = k.clone().detach().requires_grad_()
        v1 = v.clone().detach().requires_grad_()
        cu_seqlens_q1 = cu_seqlens_q.clone().detach()
        cu_seqlens_k1 = cu_seqlens_k.clone().detach()
        max_seqlen_q1 = (cu_seqlens_q1[1:] - cu_seqlens_q1[:-1]).max()
        max_seqlen_k1 = (cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).max()
        o1 = flash_attention_varlen(
            q1,
            k1,
            v1,
            cu_seqlens_q1,
            cu_seqlens_k1,
            max_seqlen_q1,
            max_seqlen_k1,
            causal=causal,
        )
        randn2 = randn.clone().detach()
        loss2 = (o1 * randn2).sum()
        loss2.backward()

        # diff
        print(f"=== Flash Attention Backward Test ({'causal' if causal else 'full'}) ===")
        print("Same Output:", torch.allclose(o, o1, atol=0.01, rtol=0.01))
        print("Max Error:", (o - o1).abs().max().item())
        print()
        print(
            "Same Query Gradient:",
            torch.allclose(q.grad, q1.grad, atol=0.01, rtol=0.01),
        )
        print("Max Query Gradient Error:", (q.grad - q1.grad).abs().max().item())
        print()
        print("Same Key Gradient:", torch.allclose(k.grad, k1.grad, atol=0.01, rtol=0.01))
        print("Max Key Gradient Error:", (k.grad - k1.grad).abs().max().item())
        print()
        print(
            "Same Value Gradient:",
            torch.allclose(v.grad, v1.grad, atol=0.01, rtol=0.01),
        )
        print("Max Value Gradient Error:", (v.grad - v1.grad).abs().max().item())
        print()

    # benchmark
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[512 * 2**i for i in range(1, 4)],
            line_arg="provider",
            line_vals=["flash", "triton-flash"],
            line_names=[
                "Flash",
                "Triton-Flash",
            ],
            styles=[("green", "-"), ("green", "--")],
            ylabel="ms",
            plot_name="** forward **",
            args={"H": 64, "D": 128},
        )
    )
    def benchmark(N, H, D, provider):
        q = torch.randn((N, H, D), device="npu", dtype=torch.bfloat16)
        k = torch.randn((N, H // 16, D), device="npu", dtype=torch.bfloat16)
        v = torch.randn((N, H // 16, D), device="npu", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, N], device="npu", dtype=torch.int32)
        sm_scale = 1 / math.sqrt(D)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "flash":
            head_num = q.shape[1]
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    head_num,
                    pse=None,
                    atten_mask=None,
                    scale=1.0 / math.sqrt(q.shape[-1]),
                    keep_prob=1,
                    input_layout="TND",
                    actual_seq_qlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                    actual_seq_kvlen=tuple(cu_seqlens[1:].cpu().numpy().tolist()),
                )[0],
                quantiles=quantiles,
            )
        if provider == "triton-flash":
            max_seqlen_q1 = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            max_seqlen_k1 = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: flash_attention_varlen(
                    q1,
                    k1,
                    v1,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen_q1,
                    max_seqlen_k1,
                    causal=False,
                ),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)
