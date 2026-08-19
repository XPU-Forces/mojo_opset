from typing import Optional
from typing import Tuple

import torch
import time
import triton
import triton.language as tl

from ..utils import get_num_cores
from ..utils import is_910


TILE_BLOCK_SIZE = 128


def _get_num_aicore():
    try:
        return max(get_num_cores(op_type="cube"), 1)
    except Exception:
        return 1


def _persistent_launch_config(num_tasks):
    num_tasks = max(int(num_tasks), 1)
    return (min(_get_num_aicore(), num_tasks),), num_tasks


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_partial_p", "stride_partial_m",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_out_z", "stride_out_h",
    ]
)
def flex_attention_kernel(
    Q,
    K,
    V,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    OUT,
    LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_out_z, stride_out_h, stride_out_m, stride_out_k,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_kv_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        out_offset = off_z * stride_out_z + off_hq * stride_out_h
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        OUT_ptr = OUT + out_offset
        LSE_ptr = LSE + lse_offset

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0
        )

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        partial_mask_offset = tl.load(PARTIAL_MASK_OFFSETS + q_sparse_idx * stride_partial_offset_m)
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        for start_n in range(0, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

            offs_n_load = kv_start + tl.arange(0, BLOCK_N)
            if USE_PACKED_PARTIAL_MASK:
                partial_block_idx = partial_mask_offset + blk_idx_in_list
                offs_m_in_block = offs_m - q_sparse_base
                offs_n_in_block = (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = load_packed_partial_mask(
                    PARTIAL_MASK_PACKED,
                    stride_partial_p,
                    stride_partial_m,
                    stride_partial_n,
                    partial_block_idx,
                    offs_m_in_block,
                    offs_n_in_block,
                    SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                )
            else:
                mask = load_dense_mask(
                    DENSE_MASK,
                    stride_mask_m,
                    stride_mask_n,
                    offs_m,
                    offs_n_load,
                    Q_LEN=Q_LEN,
                    KV_LEN=KV_LEN,
                )

            k = tl.load(
                K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n_load[:, None] < KV_LEN),
                other=0.0
            )
            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN),
                other=0.0
            )
            k = tl.trans(k)

            qk = tl.dot(q, k, input_precision="ieee")
            qk *= SM_SCALE

            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)

            alpha = tl.math.exp(m_i - m_ij_masked)
            p = tl.math.exp(qk - m_ij_masked[:, None])

            pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        if HAS_FULL_BLOCKS:
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

            for start_n in range(0, block_n_end):
                blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
                kv_block = tl.load(kv_indices + blk_idx_in_list)
                kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

                offs_n_load = kv_start + tl.arange(0, BLOCK_N)
                k = tl.load(
                    K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n_load[:, None] < KV_LEN),
                    other=0.0
                )
                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN),
                    other=0.0
                )
                k = tl.trans(k)

                qk = tl.dot(q, k, input_precision="ieee")
                qk *= SM_SCALE

                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
                alpha = tl.math.exp(m_i - m_ij)
                p = tl.math.exp(qk - m_ij[:, None])

                pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + pv
                m_i = m_ij
        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]

        out_mask = (offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM)
        tl.store(
            OUT_ptr + offs_m[:, None] * stride_out_m + offs_v[None, :] * stride_out_k,
            acc,
            mask=out_mask
        )

        lse = m_i + tl.math.log(l_i)
        tl.store(LSE_ptr + offs_m * stride_lse_m, lse, mask=offs_m < Q_LEN)


@triton.jit
def load_dense_mask(
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    offs_m,
    offs_n,
    Q_LEN,
    KV_LEN,
):
    stride_mask_m = stride_mask_m.to(tl.int64)
    ptrs = DENSE_MASK + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    valid = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def load_packed_partial_mask(
    PARTIAL_MASK_PACKED,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    partial_block_idx,
    offs_m_in_block,
    offs_n_in_block,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
):
    ptrs = (
        PARTIAL_MASK_PACKED
        + partial_block_idx * stride_partial_p
        + offs_m_in_block[:, None] * stride_partial_m
        + offs_n_in_block[None, :] * stride_partial_n
    )
    valid = (
        (offs_m_in_block[:, None] < SPARSE_Q_BLOCK_SIZE)
        & (offs_n_in_block[None, :] < SPARSE_KV_BLOCK_SIZE)
    )
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def bwd_dq_block_mn(
    q, do, lse, delta,
    K_ptr, V_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    Q_LEN, KV_LEN,
    offs_m, offs_n, offs_k, offs_v,
    q_sparse_idx, kv_block, kv_sub, q_sparse_base,
    stride_kn, stride_kk, stride_vn, stride_vk,
    MATMUL_PRECISION,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
):
    k = tl.load(
        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
        mask=(offs_n[:, None] < KV_LEN),
        other=0.0,
    )
    v = tl.load(
        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
        mask=(offs_n[:, None] < KV_LEN),
        other=0.0,
    )

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    mask = True
    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_block * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            offs_m_in_block = offs_m - q_sparse_base
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask & (offs_n[None, :] < KV_LEN), qk, float("-inf"))
    else:
        qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

    p = tl.math.exp(qk - lse[:, None])
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = p * (dp - delta[:, None])

    dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
    return dq



@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dqz", "stride_dqh",
    ]
)
def flex_attention_backward_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_kv_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS
    MATMUL_PRECISION = Q.dtype.element_ty

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        do_offset = off_z * stride_doz + off_hq * stride_doh
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
        delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h
        dq_offset = off_z * stride_dqz + off_hq * stride_dqh

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DO_ptr = DO + do_offset
        LSE_ptr = LSE + lse_offset
        DELTA_ptr = DELTA + delta_offset
        DQ_ptr = DQ + dq_offset

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0,
        )
        do = tl.load(
            DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0,
        )

        lse = tl.load(LSE_ptr + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
        delta = tl.load(DELTA_ptr + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
        lse = tl.where(lse == float("-inf"), 0.0, lse)

        dq = tl.zeros([BLOCK_M, QK_HEAD_DIM], dtype=tl.float32)

        q_sparse_idx = q_start // sparse_q_multiple
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        for blk_idx_in_list in range(0, kv_num_blocks):
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

            for kv_sub in range(NUM_KV_SUB_BLOCKS):
                start_n = kv_start_full + kv_sub * BLOCK_N
                offs_n = start_n + tl.arange(0, BLOCK_N)

                k = tl.load(
                    K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n[:, None] < KV_LEN),
                    other=0.0,
                )
                v = tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n[:, None] < KV_LEN),
                    other=0.0,
                )

                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                qk *= SM_SCALE

                if USE_PACKED_PARTIAL_MASK:
                    partial_block_idx = tl.load(
                        PARTIAL_BLOCK_TABLE
                        + q_sparse_idx * stride_partial_table_m
                        + kv_block * stride_partial_table_n
                    )
                    safe_partial_block_idx = tl.maximum(partial_block_idx, 0, propagate_nan=True)
                    offs_m_in_block = offs_m - q_sparse_base
                    offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask = load_packed_partial_mask(
                        PARTIAL_MASK_PACKED,
                        stride_partial_p,
                        stride_partial_m,
                        stride_partial_n,
                        safe_partial_block_idx,
                        offs_m_in_block,
                        offs_n_in_block,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                    )
                    mask = mask & (partial_block_idx >= 0)
                else:
                    mask = load_dense_mask(
                        DENSE_MASK,
                        stride_mask_m,
                        stride_mask_n,
                        offs_m,
                        offs_n,
                        Q_LEN=Q_LEN,
                        KV_LEN=KV_LEN,
                    )
                qk = tl.where(mask, qk, float("-inf"))

                p = tl.math.exp(qk - lse[:, None])
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta[:, None])
                ds *= SM_SCALE
                dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        if HAS_FULL_BLOCKS:
            kv_indices_f = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks_f = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            for blk_idx_in_list in range(0, kv_num_blocks_f):
                kv_block = tl.load(kv_indices_f + blk_idx_in_list)
                kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

                for kv_sub in range(NUM_KV_SUB_BLOCKS):
                    start_n = kv_start_full + kv_sub * BLOCK_N
                    offs_n = start_n + tl.arange(0, BLOCK_N)

                    k = tl.load(
                        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                        mask=(offs_n[:, None] < KV_LEN),
                        other=0.0,
                    )
                    v = tl.load(
                        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=(offs_n[:, None] < KV_LEN),
                        other=0.0,
                    )

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])
                    ds *= SM_SCALE
                    dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        tl.store(
            DQ_ptr + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                q_offset = off_z * stride_qz + off_hq * stride_qh
                do_offset = off_z * stride_doz + off_hq * stride_doh
                dq_offset = off_z * stride_qz + off_hq * stride_qh
                lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_h = Q + q_offset
                DQ_h = DQ + dq_offset
                DO_h = DO + do_offset
                LSE_h = LSE + lse_offset
                DELTA_h = DELTA + delta_offset

                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )
                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_dkdv_block_mn(
                        Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        COMPUTE_DQ=False,
                    )

                if HAS_FULL_BLOCKS:
                    q_indices = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end = tl.minimum(
                        q_num_blocks * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                    )

                    for start_m in range(0, block_m_end):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)

                        bwd_dkdv_block_mn(
                            Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                            COMPUTE_DQ=False,
                        )


@triton.jit
def bwd_dkdv_block_mn(
    Q, DO, DQ, DK_ptr, DELTA, LSE, DV_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
    stride_qm, stride_qk, stride_dom, stride_dok, stride_dqm, stride_dqd,
    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
    MATMUL_PRECISION,
    SM_SCALE: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
    COMPUTE_DQ: tl.constexpr = True,
):
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        other=0.0,
    )
    do = tl.load(
        DO + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    lse = tl.load(LSE + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == float("-inf"), 0.0, lse)

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_sparse_idx * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
            offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask, qk, float("-inf"))
    p = tl.math.exp(qk - lse[:, None])

    dv = tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do, input_precision="ieee")
    tl.atomic_add(
        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
        dv,
        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
    )

    Di = tl.load(DELTA + offs_m, mask=offs_m < Q_LEN, other=0.0)
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = (p * (dp - Di[:, None]))
    ds *= SM_SCALE

    if COMPUTE_DQ:
        dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
        tl.atomic_add(
            DQ + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqd,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )

    dk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
    tl.atomic_add(
        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk,
        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )


@triton.jit
def _bwd_dkdv_qblock_range(
    Q_h, DO_h, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n, offs_k, offs_v,
    q_indices, q_range_start, q_range_end,
    kv_sparse_idx, kv_sub,
    stride_qm, stride_qk, stride_dom, stride_dok,
    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
    MATMUL_PRECISION,
    SM_SCALE: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
):
    """遍历 [q_range_start, q_range_end) 范围内的 Q-block，累加 dk/dv 梯度。

    将 partial/full Q-block 的循环逻辑统一抽取，通过 IS_FULL_BLOCKS
    编译期常量控制 mask 加载行为，消除主 kernel 中的代码重复。

    Args:
        q_indices: 当前 kv_sparse_idx 对应的 Q-block 索引列表基址
        q_range_start / q_range_end: Q-block 线性索引的迭代范围 [start, end)
        其余参数与 bwd_dkdv_block_mn 保持一致
    """
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    for start_m in range(q_range_start, q_range_end):
        blk_idx_in_list = start_m // sparse_q_multiple
        q_block = tl.load(q_indices + blk_idx_in_list)
        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
        offs_m = q_start + tl.arange(0, BLOCK_M)

        bwd_dkdv_block_mn(
            Q_h, DO_h, DK_OUT_ptr, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
            DENSE_MASK, stride_mask_m, stride_mask_n,
            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
            k, v, Q_LEN, KV_LEN,
            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block,
            kv_sparse_idx, kv_sub, offs_k, offs_v,
            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
            MATMUL_PRECISION,
            SM_SCALE,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            QK_HEAD_DIM=QK_HEAD_DIM,
            V_HEAD_DIM=V_HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_FULL_BLOCKS=IS_FULL_BLOCKS,
            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
            COMPUTE_DQ=False,
        )


# ===========================================================================
# Task-list dkdv kernel: 基于 host 侧装箱结果的自适应负载均衡 kernel。
#
# 设计动机:
#   原始 qsplit kernel 仅处理"尾部不满核"场景, 且对所有尾部 task 统一拆分,
#   无法应对"某 KV 分块 Q-block 数远超均值"的长尾场景。本 kernel 由 host 侧
#   装箱算法生成 task list (merge 轻任务 + split 重任务), kernel 仅按 list 执行。
#
# 数据结构 (host 侧构建, 详见 _build_task_list):
#   work_items[j]  = (hkv, kv_block, sub_id, K, is_split)   # 5 元组, int32
#   task_offsets[m] = meta-task m 的 work-item 起始索引      # CSR 偏移, len = NUM_META+1
#
#   - meta-task: 分配给一个核的工作包, 含 1 个或多个 work-item, 重量 ≈ target
#   - work-item direct (is_split=0): 处理 base task 全部 Q-blocks, atomic_add 写 DK/DV
#   - work-item split  (is_split=1): 处理 base task 的 1/K 切片, atomic_add 写 DK_PARTIAL[sub_id]
#
# 写入语义:
#   - direct: 不同 (hkv, kv_block) 写不同地址, 无竞争
#   - split : 同一 (hkv, kv_block) 的不同 sub_id 写 DK_PARTIAL 不同 sub_id 维, 无竞争
#   - 被 split 的 kv_block 不会被 direct 路径写入 (host 侧保证互斥)
# ===========================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_META",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
        "stride_dkp_k", "stride_dvp_k",
    ]
)
def flex_attention_backward_dkdv_kernel_tasklist(
    Q, K, V, DO, LSE, DELTA,
    Q_NUM_BLKS, Q_IDX, FULL_Q_NUM_BLKS, FULL_Q_IDX,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, PARTIAL_MASK_OFFSETS, PARTIAL_BLOCK_TABLE,
    stride_partial_p, stride_partial_m, stride_partial_n,
    stride_partial_offset_m, stride_partial_table_m, stride_partial_table_n,
    DK, DV,                                  # direct 路径写入目标 (merge bin 内 atomic_add)
    DK_PARTIAL, DV_PARTIAL,                  # split 路径写入目标 (核间隔离)
    stride_dkp_k, stride_dvp_k,              # partial 第 0 维 (sub_id) stride
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_q_idx_m,
    WORK_ITEMS,                               # [num_work, 5]: (hkv, kv_block, sub_id, K, is_split)
    TASK_OFFSETS,                             # [NUM_META+1]: CSR 偏移
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_META,                                 # meta-task 数 (= len(task_offsets)-1)
    KV_HEAD,                                  # = Hkv (Z=1)
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    # constexpr 除法, 提升至 kernel 级避免每个 task 重复计算
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

    # ================================================================
    # meta-task 主循环 (persistent): 每个 pid 顺序领取多个 meta-task
    # 通常 NUM_META = num_core, 每核恰好 1 个 meta-task, 无 round-robin 尾部
    # ================================================================
    for meta_id in range(pid, NUM_META, num_core):
        work_start = tl.load(TASK_OFFSETS + meta_id)
        work_end = tl.load(TASK_OFFSETS + meta_id + 1)

        # ============================================================
        # work-item 内层循环: 一个 meta-task 内顺序处理多个 work-item
        #   - direct (is_split=0): merge bin 内多 base task, 各自 atomic_add DK/DV
        #   - split  (is_split=1): 单 base task 的 1/K 切片, atomic_add DK_PARTIAL
        # 同一 meta-task 内可混合 direct/split (处理不同 kv_block, 无写冲突)
        # ============================================================
        for widx in range(work_start, work_end):
            # ---- 从 work-item 中读取任务参数 ----
            off_hkv  = tl.load(WORK_ITEMS + widx * 5 + 0).to(tl.int64)
            kv_block = tl.load(WORK_ITEMS + widx * 5 + 1)
            sub_id   = tl.load(WORK_ITEMS + widx * 5 + 2)
            K_split  = tl.load(WORK_ITEMS + widx * 5 + 3)
            is_split = tl.load(WORK_ITEMS + widx * 5 + 4)

            off_z = tl.zeros_like(off_hkv)              # Z=1

            # ---- 指针基址 (按 hkv 切片) ----
            k_offset  = off_z * stride_kz  + off_hkv * stride_kh
            v_offset  = off_z * stride_vz  + off_hkv * stride_vh
            dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
            dv_offset = off_z * stride_dvz + off_hkv * stride_dvh
            K_ptr = K + k_offset
            V_ptr = V + v_offset

            # ---- 输出指针分流: direct -> DK/DV; split -> DK_PARTIAL[sub_id] ----
            if is_split == 0:
                DK_OUT_ptr = DK + dk_offset
                DV_OUT_ptr = DV + dv_offset
            else:
                DK_OUT_ptr = DK_PARTIAL + sub_id * stride_dkp_k + dk_offset
                DV_OUT_ptr = DV_PARTIAL + sub_id * stride_dvp_k + dv_offset

            start_n_full = kv_block * KV_BLOCK_SIZE
            kv_sparse_idx = kv_block // sparse_kv_multiple
            sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

            # ---- 计算 Q-block 迭代范围 [q_start, q_end) ----
            #   direct: 全部 [0, block_m_end)
            #   split : 第 sub_id/K 切片 (与 qsplit kernel 切片公式一致)
            q_indices = Q_IDX + sparse_q_idx_offset
            q_num_blocks = tl.load(Q_NUM_BLKS + kv_sparse_idx)
            block_m_end_p = tl.minimum(
                q_num_blocks * sparse_q_multiple,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            if is_split == 0:
                q_start_p = 0
                q_end_p = block_m_end_p
            else:
                q_start_p = sub_id * block_m_end_p // K_split
                q_end_p = (sub_id + 1) * block_m_end_p // K_split

            # Full Q-blocks (仅当 HAS_FULL_BLOCKS 时加载, 否则范围为空)
            q_start_f = 0
            q_end_f = 0
            q_indices_f = q_indices  # dummy; 仅 HAS_FULL_BLOCKS 时使用
            if HAS_FULL_BLOCKS:
                q_indices_f = FULL_Q_IDX + sparse_q_idx_offset
                q_num_blocks_f = tl.load(FULL_Q_NUM_BLKS + kv_sparse_idx)
                block_m_end_f = tl.minimum(
                    q_num_blocks_f * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL,
                )
                if is_split == 0:
                    q_start_f = 0
                    q_end_f = block_m_end_f
                else:
                    q_start_f = sub_id * block_m_end_f // K_split
                    q_end_f = (sub_id + 1) * block_m_end_f // K_split

            # ========================================================
            # KV sub-block 循环: 加载 k/v tile, 遍历 GQA heads
            # 复用 _bwd_dkdv_qblock_range 处理 [q_start, q_end) 切片
            # ========================================================
            for kv_sub in range(NUM_KV_SUB_BLOCKS):
                start_n = start_n_full + kv_sub * BLOCK_N
                offs_n = start_n + tl.arange(0, BLOCK_N)
                n_mask = offs_n < KV_LEN

                k = tl.load(
                    K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0,
                )
                v = tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0,
                )

                for off_g in range(0, GQA_SHARED_HEADS):
                    off_hq = (off_hkv * GQA_SHARED_HEADS + off_g).to(tl.int64)

                    Q_h = Q + off_z * stride_qz + off_hq * stride_qh
                    DO_h = DO + off_z * stride_doz + off_hq * stride_doh
                    LSE_h = LSE + off_z * stride_lse_z + off_hq * stride_lse_h
                    DELTA_h = DELTA + off_z * stride_delta_z + off_hq * stride_delta_h

                    # ---- partial Q-blocks 切片 ----
                    _bwd_dkdv_qblock_range(
                        Q_h, DO_h, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_k, offs_v,
                        q_indices, q_start_p, q_end_p,
                        kv_sparse_idx, kv_sub,
                        stride_qm, stride_qk, stride_dom, stride_dok,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE=SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                    )

                    # ---- full Q-blocks 切片 ----
                    if HAS_FULL_BLOCKS:
                        _bwd_dkdv_qblock_range(
                            Q_h, DO_h, DK_OUT_ptr, DELTA_h, LSE_h, DV_OUT_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_k, offs_v,
                            q_indices_f, q_start_f, q_end_f,
                            kv_sparse_idx, kv_sub,
                            stride_qm, stride_qk, stride_dom, stride_dok,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE=SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        )


# ===========================================================================
# Reduce kernel (tasklist 版): 对 split base task 求和合并 partial 梯度
#
# 改造要点 (vs 原 reduce_dkdv_kernel):
#   1. 移除 SPLIT_START 线性假设, 改用显式 SPLIT_BASES 列表
#   2. 每个 program 处理一个 (split_base, n_tile) 二维切片
#   3. 沿 sub_id 维累加 K 份 partial, 写入 DK/DV 原地址
#
# Grid: (num_split_base * num_n_tiles_per_kv,)
#   - num_split_base = len(split_bases)
#   - num_n_tiles_per_kv = SPARSE_KV_BLOCK_SIZE // BLOCK_N (autotune)
#
# 写入语义: tl.store 覆盖写 DK/DV。被 split 的 kv_block 仅由 split 路径写 partial,
#           direct 路径不写这些 kv_block (host 侧保证互斥), 故覆盖安全。
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}),
        triton.Config({"BLOCK_N": 64}),
        triton.Config({"BLOCK_N": 128}),
    ],
    key=["SPARSE_KV_BLOCK_SIZE"],
    restore_value=["DK", "DV"],
)
@triton.jit(
    do_not_specialize=[
        "NUM_SPLIT_BASE", "KV_LEN",
        "stride_dkp_k", "stride_dvp_k",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def reduce_dkdv_kernel_tasklist(
    DK, DV,                                  # 输出: 原梯度地址 (direct 已写入, reduce 累加 split 部分)
    DK_PARTIAL, DV_PARTIAL,                  # 输入: split 路径的 partial buffer
    SPLIT_BASES,                             # [num_split_base, 3]: (hkv, kv_block, K)
    stride_dkp_k, stride_dvp_k,              # partial 第 0 维 (sub_id) stride
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    NUM_SPLIT_BASE,
    KV_LEN,
    KV_HEAD,                                 # = Hkv (Z=1)
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    # ================================================================
    # 任务分解: pid -> (split_base_id, n_tile_in_kv)
    # ================================================================
    num_n_tiles_per_kv: tl.constexpr = SPARSE_KV_BLOCK_SIZE // BLOCK_N
    split_base_id = pid // num_n_tiles_per_kv
    n_tile_in_kv  = pid %  num_n_tiles_per_kv

    # ---- 从 SPLIT_BASES 读取当前 split base task 信息 ----
    off_hkv  = tl.load(SPLIT_BASES + split_base_id * 3 + 0).to(tl.int64)
    kv_block = tl.load(SPLIT_BASES + split_base_id * 3 + 1)
    K_split  = tl.load(SPLIT_BASES + split_base_id * 3 + 2)

    off_z = tl.zeros_like(off_hkv)                              # Z=1

    # ---- 定位 N 轴切片 (kv_block 内的 n_tile) ----
    start_n = kv_block * SPARSE_KV_BLOCK_SIZE + n_tile_in_kv * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < KV_LEN

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
    dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

    # ================================================================
    # 累加 K 份 partial: DK = sum_{s=0}^{K-1} DK_PARTIAL[s, hkv, kv_block, ...]
    # 固定迭代顺序 s=0,1,...,K-1, 保证累加顺序确定
    # ================================================================
    dk_sum = tl.zeros([BLOCK_N, QK_HEAD_DIM], dtype=tl.float32)
    for s in range(K_split):
        dk_sum += tl.load(
            DK_PARTIAL + s * stride_dkp_k + dk_offset
            + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
            mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0,
        )
    tl.store(
        DK + dk_offset
        + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk_sum,
        mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
    )

    # ---- DV 累加 ----
    dv_sum = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)
    for s in range(K_split):
        dv_sum += tl.load(
            DV_PARTIAL + s * stride_dvp_k + dv_offset
            + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
            mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
            other=0.0,
        )
    tl.store(
        DV + dv_offset
        + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
        dv_sum,
        mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
    )


# ===========================================================================
# Legacy Reduce kernel (仅被 legacy qsplit 分支使用, 保留供回退)
# 仅对 split KV block 求和 — DK = sum(DK_PARTIAL[0..K-1])
# Direct KV block 已在 qsplit kernel 中直接写入 DK/DV。
#
# 固定迭代顺序 s=0,1,...,K-1，保证结果确定。
# AutoTune 选择最优 BLOCK_N，grid 数 = num_split_base × (SPARSE_KV_BLOCK_SIZE // BLOCK_N)，
# 由 grid lambda 根据 autotuned BLOCK_N 动态计算，使总 tile 数自动适配硬件核数。
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}),
        triton.Config({"BLOCK_N": 64}),
        triton.Config({"BLOCK_N": 128}),
    ],
    key=["K", "SPARSE_KV_BLOCK_SIZE"],
    restore_value=["DK", "DV"],
)
@triton.jit(
    do_not_specialize=[
        "NUM_SPLIT_BASE", "KV_LEN", "SPLIT_START",
        "NUM_KV_BLOCKS",
        "stride_dkp_k", "stride_dvp_k",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def reduce_dkdv_kernel(
    DK, DV,
    DK_PARTIAL, DV_PARTIAL,
    stride_dkp_k, stride_dvp_k,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    NUM_SPLIT_BASE, KV_LEN, SPLIT_START, NUM_KV_BLOCKS,
    KV_HEAD,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    # 每个 program 处理一个 N-tile; NUM_N_TILES_PER_KV 由 constexpr 推导
    num_n_tiles_per_kv = SPARSE_KV_BLOCK_SIZE // BLOCK_N
    split_base_id = pid // num_n_tiles_per_kv
    n_tile_in_kv = pid % num_n_tiles_per_kv

    base_task = SPLIT_START + split_base_id
    kv_block = base_task % NUM_KV_BLOCKS
    zhkv = base_task // NUM_KV_BLOCKS
    off_z = (zhkv // KV_HEAD).to(tl.int64)
    off_hkv = (zhkv % KV_HEAD).to(tl.int64)

    start_n = kv_block * SPARSE_KV_BLOCK_SIZE + n_tile_in_kv * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < KV_LEN

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
    dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

    # Sum K partials for DK (fixed order -> deterministic)
    dk_sum = tl.zeros([BLOCK_N, QK_HEAD_DIM], dtype=tl.float32)
    for s in range(K):
        dk_sum += tl.load(
            DK_PARTIAL + s * stride_dkp_k + dk_offset
            + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
            mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0,
        )
    tl.store(
        DK + dk_offset + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk_sum,
        mask=n_mask[:, None] & (offs_k[None, :] < QK_HEAD_DIM),
    )

    # Sum K partials for DV
    dv_sum = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)
    for s in range(K):
        dv_sum += tl.load(
            DV_PARTIAL + s * stride_dvp_k + dv_offset
            + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
            mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
            other=0.0,
        )
    tl.store(
        DV + dv_offset + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
        dv_sum,
        mask=n_mask[:, None] & (offs_v[None, :] < V_HEAD_DIM),
    )


def _prepare_block_mask_attrs(block_mask, q, num_q_blocks, sparse_q_block_size, sparse_kv_block_size):
    N = q.shape[0] if q.dim() == 4 else q.shape[2]
    kv_num_blks = block_mask.kv_num_blocks
    kv_idx = block_mask.kv_indices
    full_kv_num_blks = getattr(block_mask, "full_kv_num_blocks", torch.zeros_like(kv_num_blks))
    full_kv_idx = getattr(block_mask, "full_kv_indices", torch.zeros_like(kv_idx))

    q_num_blks = getattr(block_mask, "q_num_blocks", None)
    q_idx = getattr(block_mask, "q_indices", None)
    assert q_num_blks is not None, "q_num_blocks and q_indices must be provided"
    assert q_idx is not None, "q_indices must be provided"
    full_q_num_blks = getattr(block_mask, "full_q_num_blocks", torch.zeros_like(q_num_blks))
    full_q_idx = getattr(block_mask, "full_q_indices", torch.zeros_like(q_idx))

    kv_num_blks = kv_num_blks.to(torch.int32).contiguous()
    kv_idx = kv_idx.to(torch.int32).contiguous()
    full_kv_num_blks = full_kv_num_blks.to(torch.int32).contiguous()
    full_kv_idx = full_kv_idx.to(torch.int32).contiguous()
    q_num_blks = q_num_blks.to(torch.int32).contiguous()
    q_idx = q_idx.to(torch.int32).contiguous()
    full_q_num_blks = full_q_num_blks.to(torch.int32).contiguous()
    full_q_idx = full_q_idx.to(torch.int32).contiguous()

    dense_mask = getattr(block_mask, "dense_mask", None)
    packed_partial_mask = getattr(block_mask, "packed_partial_mask", None)
    partial_mask_offsets = getattr(block_mask, "partial_mask_offsets", None)
    partial_block_table = getattr(block_mask, "partial_block_table", None)
    use_packed_partial_mask = (
        packed_partial_mask is not None
        and partial_mask_offsets is not None
        and partial_block_table is not None
    )

    if dense_mask is None:
        dense_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=q.device)
    dense_mask = dense_mask.contiguous()

    if use_packed_partial_mask:
        packed_partial_mask = packed_partial_mask.contiguous()
        partial_mask_offsets = partial_mask_offsets.to(torch.int32).contiguous()
        partial_block_table = partial_block_table.to(torch.int32).contiguous()
    else:
        packed_partial_mask = torch.zeros(
            (1, sparse_q_block_size, sparse_kv_block_size),
            dtype=torch.bool,
            device=q.device,
        )
        partial_mask_offsets = torch.zeros(
            (1, 1, max(num_q_blocks, 1)),
            dtype=torch.int32,
            device=q.device,
        )
        partial_block_table = torch.full(
            (max(num_q_blocks, 1), max((N + sparse_kv_block_size - 1) // sparse_kv_block_size, 1)),
            -1,
            dtype=torch.int32,
            device=q.device,
        )

    return {
        "kv_num_blks": kv_num_blks,
        "kv_idx": kv_idx,
        "full_kv_num_blks": full_kv_num_blks,
        "full_kv_idx": full_kv_idx,
        "q_num_blks": q_num_blks,
        "q_idx": q_idx,
        "full_q_num_blks": full_q_num_blks,
        "full_q_idx": full_q_idx,
        "dense_mask": dense_mask,
        "packed_partial_mask": packed_partial_mask,
        "partial_mask_offsets": partial_mask_offsets,
        "partial_block_table": partial_block_table,
        "use_packed_partial_mask": use_packed_partial_mask,
    }


def flex_attention_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape

    GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = TILE_BLOCK_SIZE
    BLOCK_N = TILE_BLOCK_SIZE
    SPARSE_Q_BLOCK_SIZE = BLOCK_M
    SPARSE_KV_BLOCK_SIZE = BLOCK_N

    num_q_blocks = (M + SPARSE_Q_BLOCK_SIZE - 1) // SPARSE_Q_BLOCK_SIZE

    output = torch.empty_like(q)
    lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    bm = _prepare_block_mask_attrs(block_mask, q, num_q_blocks, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)

    num_tasks = num_q_blocks * Z * Hq
    grid, num_tasks = _persistent_launch_config(num_tasks)

    flex_attention_kernel[grid](
        q, k, v,
        bm["kv_num_blks"], bm["kv_idx"], bm["full_kv_num_blks"], bm["full_kv_idx"],
        bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
        bm["packed_partial_mask"], bm["partial_mask_offsets"],
        bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
        bm["partial_mask_offsets"].stride(2),
        output, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        bm["kv_idx"].stride(2),
        SM_SCALE=sm_scale,
        QK_HEAD_DIM=D,
        V_HEAD_DIM=Dv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_TASKS=num_tasks,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_HEAD=Hq,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        Q_LEN=M,
        KV_LEN=N,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=True,
        USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True,
        intra_cache_num=3,
        inter_cache_num=2,
        enable_cross_if_fusion=True,
        enable_buffer_insert_optimization=True,
        enable_ub_refine_opt = True,
    )

    return output, lse


# ============================================================================
# Host 侧负载均衡: 任务列表构建 (Step 1)
#
# 输入: 每个 kv_sparse_idx 的有效 Q-block 数 (partial + full 等权)
# 输出: work_items / task_offsets / split_bases / max_sub
#
# 算法 (两阶段, 统一装箱):
#   阶段 A: 枚举所有 base task (hkv, kv_block)
#           - w <= target: 作为 direct work-item (整块, K=1)
#           - w >  target: 拆成 K=ceil(w/target) 个 split work-item (各 1/K 重量)
#   阶段 B: 所有 work-item 降序 first-fit 装入 num_core 个 bin
#           - 每 bin = 1 meta-task, 重量 ≈ target
#           - 同一 bin 可混合 direct/split (处理不同 kv_block, 无写冲突)
#
# 互斥保证: 一个 kv_block 要么整体 direct (w<=target), 要么整体 split (w>target),
#           两者不混存, 故 split 的 kv_block 的 DK/DV 地址仅由 reduce 写入, 覆盖安全。
# ============================================================================
# 装箱策略: hkv 分块连续装箱 (唯一策略)
#
# 每个 core 连续处理完一个 hkv 的所有 work-item 再切下一个 hkv,
# 缓存命中最高 (同 hkv 内 K/V 驻留 cache)。组内按重量降序,
# 每个 work-item 放入当前最轻的 bin, 保证负载均衡。
# ============================================================================

def _first_fit_into_bins(ordered_items, num_core, target, TOL):
    """通用 first-fit 装箱: 按 ordered_items 顺序依次放入第一个容得下的 bin。

    装不下时放入当前最轻的 bin (保证 bin 数 = num_core, 不溢出)。
    不同策略只需改变 ordered_items 的顺序即可复用本函数。
    """
    bins = [[] for _ in range(num_core)]
    bin_w = [0.0] * num_core
    for wi in ordered_items:
        w = wi[5]
        placed = False
        for b in range(num_core):
            if bin_w[b] + w <= target * TOL:
                bins[b].append(wi)
                bin_w[b] += w
                placed = True
                break
        if not placed:
            lightest = bin_w.index(min(bin_w))
            bins[lightest].append(wi)
            bin_w[lightest] += w
    return bins


def _bin_pack_hkv_continuous(work_items_list, num_core, target, TOL):
    """hkv 分块连续装箱。

    每个 core 连续处理完一个 hkv 的所有 work-item 再切下一个 hkv,
    缓存命中最高 (同 hkv 内 K/V 驻留 cache)。组内按重量降序,
    每个 work-item 放入当前最轻的 bin, 保证负载均衡。

    Args:
        work_items_list: [(hkv, kv_block, sub_id, K, is_split, w), ...]
        num_core: bin 数 (= 核数)
        target: 单核目标重量
        TOL: 装箱容忍 (bin_w + wi <= target*TOL)

    Returns:
        bins: List[List[work_item]], 每个 core 的串行执行序列
    """
    bins = [[] for _ in range(num_core)]
    bin_w = [0.0] * num_core

    # 按 hkv 分组, 组内按重量降序
    groups = {}
    for wi in work_items_list:
        groups.setdefault(wi[0], []).append(wi)

    for hkv in sorted(groups.keys()):
        group = sorted(groups[hkv], key=lambda t: t[5], reverse=True)
        # 组内: 每个 work-item 放入当前最轻的 bin, 让同 hkv 任务尽量集中
        for wi in group:
            w = wi[5]
            lightest = bin_w.index(min(bin_w))
            bins[lightest].append(wi)
            bin_w[lightest] += w
    return bins


def _build_task_list(
    w_sparse: torch.Tensor,   # [num_kv_sparse], 每个 kv_sparse_idx 的有效 Q-block 数
    Hkv: int,
    num_kv_blocks: int,
    sparse_kv_multiple: int,  # kv_block -> kv_sparse_idx 的换算因子
    target: float,            # 单核目标重量
    num_core: int,            # 核数
    device,
):
    """构建 task list (merge + split 统一装箱)。

    装箱策略固定为 hkv 分块连续装箱 (_bin_pack_hkv_continuous)。

    Returns:
        work_items_t:   [num_work, 5] int32, (hkv, kv_block, sub_id, K, is_split)
        task_offsets_t: [num_meta+1] int32, CSR 偏移
        split_bases_t:  [num_split_base, 3] int32, (hkv, kv_block, K)
        max_sub:        int, partial buffer 第 0 维大小
    """
    # ================================================================
    # 阶段 A: 生成所有 work-item
    #
    # 性能优化要点:
    #   1. 原实现在 Hkv * num_kv_blocks 内层循环中逐次调用
    #      w_sparse[...].item(), 每次触发一次 NPU->CPU 同步 (约 11 us/次),
    #      是该函数 >95% 耗时的根源。改为一次性 .tolist() 将整张小张量
    #      同步到 Python 列表, 把 Hkv*num_kv_blocks 次同步降为 1 次。
    #
    #   2. 每个 kv_block 的重量 w 仅依赖 kv_block // sparse_kv_multiple,
    #      与 hkv 完全无关。先构建 hkv=0 的模板 (template), 再按 hkv
    #      批量复制, 避免在 Hkv 重循环中重复执行分支判断与 ceil 除法。
    #      复制后 work_items_list / split_bases_list 的元素顺序与原实现
    #      严格一致 (hkv 外层升序, kv_block 内层升序)。
    # ================================================================
    w_sparse_list = w_sparse.tolist()
    target_int = int(target)

    # 预计算每个 kv_block 的重量 (与 hkv 无关, 只算一次)
    w_per_kv_block = [
        w_sparse_list[kv_block // sparse_kv_multiple]
        for kv_block in range(num_kv_blocks)
    ]

    # 构建 hkv=0 的模板: (kv_block, sub_id, K, is_split, w)
    template_items = []
    template_split_bases = []
    max_sub = 1

    for kv_block in range(num_kv_blocks):
        w = w_per_kv_block[kv_block]
        if w == 0:
            continue
        if w <= target:
            # direct: 整块, K=1
            template_items.append((kv_block, 0, 1, 0, float(w)))
        else:
            # split: 拆 K 份
            K = (w + target_int - 1) // target_int   # ceil(w/target)
            template_split_bases.append((kv_block, K))
            if K > max_sub:
                max_sub = K
            per_sub_w = w / K
            for sub in range(K):
                template_items.append((kv_block, sub, K, 1, per_sub_w))

    # 按 hkv 复制模板, 生成完整 work_items_list / split_bases_list
    work_items_list = []   # [(hkv, kv_block, sub_id, K, is_split, w), ...]
    split_bases_list = []  # [(hkv, kv_block, K), ...]
    for hkv in range(Hkv):
        for kv_block, sub_id, K_val, is_split, w in template_items:
            work_items_list.append((hkv, kv_block, sub_id, K_val, is_split, w))
        for kv_block, K_val in template_split_bases:
            split_bases_list.append((hkv, kv_block, K_val))

    # ================================================================
    # 阶段 B: hkv 分块连续装箱, 装入 num_core 个 bin
    # ================================================================
    TOL = 1.0   # 装箱容忍: bin_w + wi <= target*TOL
    bins = _bin_pack_hkv_continuous(work_items_list, num_core, target, TOL)

    # ================================================================
    # 组装 CSR: work_items_final[j] = (hkv, kv_block, sub_id, K, is_split)
    #           task_offsets[m] = meta-task m 的 work-item 起始索引
    # ================================================================
    work_items_final = []
    task_offsets = [0]
    for b in range(num_core):
        for wi in bins[b]:
            work_items_final.append((wi[0], wi[1], wi[2], wi[3], wi[4]))
        task_offsets.append(len(work_items_final))

    # ---- 转 tensor ----
    if work_items_final:
        work_items_t = torch.tensor(work_items_final, dtype=torch.int32, device=device)
    else:
        work_items_t = torch.zeros((0, 5), dtype=torch.int32, device=device)

    task_offsets_t = torch.tensor(task_offsets, dtype=torch.int32, device=device)

    if split_bases_list:
        split_bases_t = torch.tensor(split_bases_list, dtype=torch.int32, device=device)
    else:
        split_bases_t = torch.zeros((0, 3), dtype=torch.int32, device=device)

    return work_items_t, task_offsets_t, split_bases_t, max_sub


def _compute_and_cache_task_list(
    block_mask,
    w_sparse: torch.Tensor,
    Hkv: int,
    num_kv_blocks: int,
    sparse_kv_multiple: int,
    target: float,
    num_core: int,
    device,
    sparse_kv_block_size: int,
):
    """构建 bwd task list 并缓存到 block_mask 上 (延迟缓存)。

    task list 仅依赖 block_mask 的 q_num_blocks/full_q_num_blocks 以及 Hkv、
    SPARSE_KV_BLOCK_SIZE, 与具体 q/k/v 数据无关。首次 bwd 调用时计算并缓存,
    同一 mask 的后续 bwd 调用直接复用, 避免重复装箱。

    调用前提: bwd 侧 use_tasklist 判定已保证 sparse_kv_block_size == TILE_BLOCK_SIZE,
    故本函数无条件缓存。

    Args:
        block_mask: BlockMask 对象, 计算结果缓存到其 task_list_* 属性上。
        w_sparse: [num_kv_sparse] 每个 kv_sparse_idx 的有效 Q-block 数。
        Hkv: KV head 数。
        num_kv_blocks: KV block 总数。
        sparse_kv_multiple: kv_block -> kv_sparse_idx 的换算因子。
        target: 单核目标重量。
        num_core: 核数。
        device: 张量分配设备。
        sparse_kv_block_size: 当前执行的 SPARSE_KV_BLOCK_SIZE (校验/缓存用)。

    Returns:
        work_items_t:   [num_work, 5] int32, (hkv, kv_block, sub_id, K, is_split)
        task_offsets_t: [num_meta+1] int32, CSR 偏移
        split_bases_t:  [num_split_base, 3] int32, (hkv, kv_block, K)
        max_sub:        int, partial buffer 第 0 维大小
    """
    start_time = time.perf_counter()
    work_items_t, task_offsets_t, split_bases_t, max_sub = _build_task_list(
        w_sparse.reshape(-1), Hkv, num_kv_blocks, sparse_kv_multiple,
        target, num_core, device,
    )
    end_time = time.perf_counter()

    # 缓存到 block_mask (进入 tasklist 分支时 SPARSE_KV_BLOCK_SIZE == TILE_BLOCK_SIZE 已保证)
    block_mask.task_list_work_items = work_items_t
    block_mask.task_list_offsets = task_offsets_t
    block_mask.task_list_split_bases = split_bases_t
    block_mask.task_list_max_sub = max_sub
    block_mask.task_list_sparse_kv_block_size = sparse_kv_block_size
    block_mask.task_list_num_kv_heads = Hkv

    return work_items_t, task_offsets_t, split_bases_t, max_sub


def flex_attention_bwd_impl(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    block_mask,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape
    GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    grad_output = grad_output.contiguous()
    delta = (output * grad_output).sum(dim=-1).to(torch.float32).contiguous()

    SPARSE_Q_BLOCK_SIZE = TILE_BLOCK_SIZE
    SPARSE_KV_BLOCK_SIZE = TILE_BLOCK_SIZE
    num_q_blocks = triton.cdiv(M, SPARSE_Q_BLOCK_SIZE)

    bm = _prepare_block_mask_attrs(block_mask, q, num_q_blocks, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)

    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)

    BLOCK_M_DQ = TILE_BLOCK_SIZE
    BLOCK_N_DQ = TILE_BLOCK_SIZE
    NUM_KV_SUB_BLOCKS_VAL = SPARSE_KV_BLOCK_SIZE // BLOCK_N_DQ
    grid_dq, num_tasks_dq = _persistent_launch_config(num_q_blocks * Z * Hq)
    flex_attention_backward_dq_kernel[grid_dq](
        q, k, v, grad_output, lse, delta,
        bm["kv_num_blks"], bm["kv_idx"], bm["full_kv_num_blks"], bm["full_kv_idx"],
        bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
        bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
        bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
        bm["partial_mask_offsets"].stride(2),
        bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
        dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        bm["kv_idx"].stride(2),
        SM_SCALE=sm_scale,
        QK_HEAD_DIM=D,
        V_HEAD_DIM=Dv,
        BLOCK_M=BLOCK_M_DQ,
        BLOCK_N=BLOCK_N_DQ,
        NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
        NUM_TASKS=num_tasks_dq,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_HEAD=Hq,
        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        Q_LEN=M,
        KV_LEN=N,
        GQA_SHARED_HEADS=GQA_SHARED_HEADS,
        HAS_FULL_BLOCKS=True,
        USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True,
        enable_select_analysis=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
        intra_cache_num=3,
        inter_cache_num=2,
    )

    BLOCK_M_DKDV = TILE_BLOCK_SIZE
    BLOCK_N_DKDV = TILE_BLOCK_SIZE
    NUM_KV_SUB_BLOCKS_VAL = SPARSE_KV_BLOCK_SIZE // BLOCK_N_DKDV
    num_kv_blocks = triton.cdiv(N, SPARSE_KV_BLOCK_SIZE)

    # ========================================================================
    # DKDV 反向: 负载均衡分支判定 (Step 0)
    #
    # 重量定义: 每个 kv_sparse_idx 的有效 Q-block 数 (partial + full 等权)
    #   w_sparse[kv_sparse_idx] = q_num_blks + full_q_num_blks
    #   base task = (hkv, kv_block), 重量 = w_sparse[kv_block // sparse_kv_multiple]
    #
    # 仅在以下两种场景才进入 task-list 路径 (否则走零开销的 SIMPLE kernel):
    #
    # 场景一 — 尾块浪费显著 (total_base 不能整除核数):
    #   满轮数 full_rounds = total_base // num_core <= MAX_FULL_ROUNDS, 且
    #   尾轮核占比 tail_ratio = (total_base % num_core) / num_core < TAIL_RATIO_THRESHOLD
    #   (轮数少 + 尾轮空闲核多 → 尾块浪费占总工作比例大, 值得装箱均衡)
    #
    # 场景二 — 重量长尾 (total_base 能整除核数, 无尾块问题):
    #   max_w / mean_w > IMB_THRESHOLD, 各 KV 分块 Q-block 数差异大
    #   (任务数虽均衡但实际计算量不均衡, 需 split 重任务)
    # ========================================================================
    total_base = num_kv_blocks * Z * Hkv
    num_core = _get_num_aicore()
    KV_BLOCK_SIZE_DKDV = BLOCK_N_DKDV * NUM_KV_SUB_BLOCKS_VAL
    sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE_DKDV

    # ---- 重量统计 ----
    w_sparse = (bm["q_num_blks"] + bm["full_q_num_blks"]).to(torch.int32)
    total_w = Hkv * w_sparse.sum().item()
    mean_w = total_w / max(total_base, 1)
    max_w = w_sparse.max().item()
    target = total_w / num_core                       # 单核目标重量

    # ---- 分支判定阈值 ----
    IMB_THRESHOLD = 1.5                               # 重量长尾判定阈值
    TAIL_RATIO_THRESHOLD = 0.5                        # 尾轮核占比阈值
    MAX_FULL_ROUNDS = 2                               # 尾块判定的最大满轮数

    full_rounds = total_base // num_core
    tail_cores = total_base % num_core
    tail_ratio = tail_cores / num_core

    # 场景一: 尾块浪费显著 (不能整除 + 满轮数少 + 尾轮核占比低)
    has_significant_tail = (
        tail_cores > 0
        and full_rounds <= MAX_FULL_ROUNDS
        and tail_ratio < TAIL_RATIO_THRESHOLD
    )
    # 场景二: 重量长尾 (能整除, 无尾块问题, 但 KV 分块间计算量差异大)
    has_weight_imbalance = (
        tail_cores == 0
        and mean_w > 0
        and max_w / mean_w > IMB_THRESHOLD
    )
    use_tasklist = has_significant_tail or has_weight_imbalance
    # tasklist 路径的装箱参数仅在 SPARSE_KV_BLOCK_SIZE == TILE_BLOCK_SIZE 时有效,
    # 不满足时直接走 SIMPLE kernel 分支 (零额外开销)
    use_tasklist = use_tasklist and (SPARSE_KV_BLOCK_SIZE == TILE_BLOCK_SIZE)

    if not use_tasklist:
        # ================================================================
        # 均衡路径: 原始 SIMPLE kernel (零额外开销, 性能最优)
        # 每个 (z, hkv, kv_block) 一个 task, round-robin 调度
        # ================================================================
        grid_dkv, num_tasks_dkv = _persistent_launch_config(total_base)
        flex_attention_backward_dkdv_kernel[grid_dkv](
            q, k, v, grad_output, lse, delta,
            bm["q_num_blks"], bm["q_idx"], bm["full_q_num_blks"], bm["full_q_idx"],
            bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
            bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
            bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
            bm["partial_mask_offsets"].stride(2),
            bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            bm["q_idx"].stride(2),
            SM_SCALE=sm_scale,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
            NUM_TASKS=num_tasks_dkv,
            NUM_KV_BLOCKS=num_kv_blocks,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=True,
            USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            #unit_flag=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=2,
            inter_cache_num=1,
        )

    else:
        # ================================================================
        # 不均衡路径: task-list kernel (merge 轻任务 + split 重任务 + reduce)
        #
        # Step 1: host 侧构建任务列表 (统一装箱, bin 数 = num_core)
        # Step 2: launch task-list kernel (CSR 双层循环, 每核 1 meta-task)
        # Step 3: launch reduce kernel (仅对 split base task 合并 partial)
        # ================================================================

        # ---- Step 1: 获取任务列表 (延迟缓存: 首次计算并缓存, 后续复用) ----
        #
        # task list 仅依赖 block_mask 的 q_num_blocks/full_q_num_blocks 以及 Hkv、
        # SPARSE_KV_BLOCK_SIZE, 与具体 q/k/v 数据无关。因此首次 bwd 调用时计算,
        # 缓存到 block_mask 上, 同一 mask 的后续 bwd 调用直接复用, 避免重复装箱。
        #
        # 合法性拦截 (全部满足才复用缓存, 否则重新计算并刷新缓存):
        #   1. block_mask 上存在 task_list 缓存 (首次调用后写入)
        #   2. 缓存时的 SPARSE_KV_BLOCK_SIZE == 当前执行的 SPARSE_KV_BLOCK_SIZE
        #   3. 缓存时的 Hkv == 当前 Hkv
        #   (SPARSE_KV_BLOCK_SIZE == TILE_BLOCK_SIZE 已在 use_tasklist 判定时保证)
        cached_sparse_kv_bs = getattr(block_mask, "task_list_sparse_kv_block_size", None)
        cached_hkv = getattr(block_mask, "task_list_num_kv_heads", None)
        has_valid_cache = (
            cached_sparse_kv_bs is not None
            and cached_sparse_kv_bs == SPARSE_KV_BLOCK_SIZE
            and cached_hkv == Hkv
        )

        if has_valid_cache:
            work_items_t = block_mask.task_list_work_items
            task_offsets_t = block_mask.task_list_offsets
            split_bases_t = block_mask.task_list_split_bases
            max_sub = block_mask.task_list_max_sub
        else:
            # 首次调用或参数不匹配: 构建任务列表并缓存到 block_mask
            work_items_t, task_offsets_t, split_bases_t, max_sub = _compute_and_cache_task_list(
                block_mask, w_sparse, Hkv, num_kv_blocks, sparse_kv_multiple,
                target, num_core, k.device, SPARSE_KV_BLOCK_SIZE,
            )

        num_meta = num_core                             # meta-task 数 = 核数
        num_split_base = split_bases_t.shape[0]

        # ---- 分配 partial buffer (仅 split 路径需要) ----
        # partial: [max_sub, Z, Hkv, N, D], 各 sub_id 写入互不重叠
        # Z=1 (用户确认), 但保留 Z 维以兼容 dk/dv 的 stride
        dk_partial = torch.zeros(
            (max_sub, Z, Hkv, N, D), dtype=torch.float32, device=k.device,
        )
        dv_partial = torch.zeros(
            (max_sub, Z, Hkv, N, Dv), dtype=torch.float32, device=v.device,
        )

        # ---- Step 2: launch task-list kernel ----
        grid_tasklist, _ = _persistent_launch_config(num_meta)
        flex_attention_backward_dkdv_kernel_tasklist[grid_tasklist](
            q, k, v, grad_output, lse, delta,
            bm["q_num_blks"], bm["q_idx"], bm["full_q_num_blks"], bm["full_q_idx"],
            bm["dense_mask"], bm["dense_mask"].stride(2), bm["dense_mask"].stride(3),
            bm["packed_partial_mask"], bm["partial_mask_offsets"], bm["partial_block_table"],
            bm["packed_partial_mask"].stride(0), bm["packed_partial_mask"].stride(1), bm["packed_partial_mask"].stride(2),
            bm["partial_mask_offsets"].stride(2),
            bm["partial_block_table"].stride(0), bm["partial_block_table"].stride(1),
            dk, dv,                                      # direct 路径写入目标
            dk_partial, dv_partial,                      # split 路径写入目标
            dk_partial.stride(0), dv_partial.stride(0),  # partial sub_id 维 stride
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            bm["q_idx"].stride(2),
            work_items_t, task_offsets_t,                # 任务列表
            SM_SCALE=sm_scale,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
            NUM_META=num_meta,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=True,
            USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=2,
            inter_cache_num=1,
        )

        # ---- Step 3: launch reduce kernel (仅当存在 split base) ----
        if num_split_base > 0:
            grid_reduce = lambda meta: (
                num_split_base * (SPARSE_KV_BLOCK_SIZE // meta["BLOCK_N"]),
            )
            reduce_dkdv_kernel_tasklist[grid_reduce](
                dk, dv,
                dk_partial, dv_partial,
                split_bases_t,
                dk_partial.stride(0), dv_partial.stride(0),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                num_split_base, N,
                KV_HEAD=Hkv,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                QK_HEAD_DIM=D,
                V_HEAD_DIM=Dv,
            )

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


# ============================================================================
# BlockMask construction: streaming stripe packed-block builders
# ============================================================================
# Two construction strategies for building a packed BlockMask:
#
#   1. Kernel-based (_build_packed_block_mask_streaming):
#      Uses custom Triton create_mask_kernel + block_classify_kernel for fused
#      mask generation and classification.  Caller supplies a mask_type_str
#      (e.g. "sparse", "full") and a *problem* dict with pre-built index tables.
#
#   2. mask_mod-based (create_block_mask_patched):
#      Evaluates an arbitrary mask_mod callable ``(b, h, q_idx, kv_idx) -> bool``
#      in streaming stripes, then classifies and packs partial blocks.
#      API mirrors torch.nn.attention.flex_attention.create_block_mask.
# ============================================================================

import torch.nn.functional as _F

from torch.nn.attention.flex_attention import _convert_mask_to_block_mask
from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask
from torch.nn.attention.flex_attention import create_mask as _torch_create_mask

from mojo_opset.utils.platform import get_torch_device as _get_torch_device


# -- Constants ---------------------------------------------------------------
_MB = 1024 ** 2

# Target memory per stripe for streaming mask build (~256 MB of bool dense mask).
_STRIPE_TARGET_BYTES = 256 * _MB

# During mask_mod evaluation, each intermediate tensor is [stripe_q, KV_LEN] in
# int64 (8 bytes).  A typical mask_mod creates ~4 simultaneous intermediates, so
# we budget ~8 bytes per (q, kv) element when sizing each stripe.
_BYTES_PER_MASK_ELEMENT = 8

# Tile size used by the kernel-based mask generation path.
MASK_BLOCK_SIZE = TILE_BLOCK_SIZE if not is_910() else 64


# -- Utility helpers ---------------------------------------------------------
def _round_up_to_multiple(x, multiple):
    """Round *x* up to the nearest multiple of *multiple*."""
    return (x + multiple - 1) // multiple * multiple


def _get_num_vector_core():
    """Return the number of vector cores on the current NPU device (fallback 1)."""
    try:
        dev = torch.npu.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(dev)
        return max(int(props.get("num_vectorcore", 1)), 1)
    except Exception:
        return 1


# ============================================================================
# Kernel: create_mask_kernel
# ============================================================================
@triton.jit
def create_mask_kernel(
    OUT, stride_ob, stride_oh, stride_oq, stride_ok: tl.constexpr,
    BLOCK_FLAGS, stride_bf_q, stride_bf_k,
    TABLE1, stride_t1, TABLE2, stride_t2, TABLE3, stride_t3,
    Q_LEN, KV_LEN, W, G,
    Q_OFFSET,
    MASK_TYPE: tl.constexpr, TILE: tl.constexpr,
    STORE_MASK: tl.constexpr, CLASSIFY: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int32)
    pid_k = tl.program_id(1).to(tl.int32)
    q_off = Q_OFFSET + pid_q * TILE + tl.arange(0, TILE)
    k_off = pid_k * TILE + tl.arange(0, TILE)
    q_idx = q_off[:, None]
    k_idx = k_off[None, :]

    if MASK_TYPE == 0:
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=0)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-1)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        window = causal & ((q_idx - k_idx) <= W)
        ds_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        glob = causal & (k_idx >= ds_q) & (k_idx < ds_q + G)
        sparse = same_doc & (window | glob)
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = sparse | same_img
    elif MASK_TYPE == 1:
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        frame_causal = fid_q >= fid_k
        result = same_doc & frame_causal
    elif MASK_TYPE == 2:
        vid_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        vid_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_video = vid_q == vid_k
        fid_q = tl.load(TABLE2 + q_idx * stride_t2, mask=q_idx < Q_LEN, other=0)
        fid_k = tl.load(TABLE2 + k_idx * stride_t2, mask=k_idx < KV_LEN, other=-1)
        same_frame = fid_q == fid_k
        prev_frame = fid_q > fid_k
        result = same_video & (same_frame | prev_frame)
    elif MASK_TYPE == 3:
        causal = q_idx >= k_idx
        mod_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        is_video = mod_q > 0
        same_video = is_video & (mod_q == mod_k)
        result = causal | same_video
    elif MASK_TYPE == 4:
        seg_q = tl.load(TABLE1 + q_idx * stride_t1, mask=q_idx < Q_LEN, other=-1)
        seg_k = tl.load(TABLE1 + k_idx * stride_t1, mask=k_idx < KV_LEN, other=-2)
        same_doc = seg_q == seg_k
        causal = q_idx >= k_idx
        samedoc_causal = same_doc & causal
        mod_q = tl.load(TABLE3 + q_idx * stride_t3, mask=q_idx < Q_LEN, other=-1)
        mod_k = tl.load(TABLE3 + k_idx * stride_t3, mask=k_idx < KV_LEN, other=-2)
        is_img = mod_q > 0
        same_img = is_img & (mod_q == mod_k)
        result = samedoc_causal | same_img
    else:
        result = tl.full([TILE, TILE], False, tl.int1)

    valid = (q_idx < Q_LEN) & (k_idx < KV_LEN)

    if STORE_MASK:
        q_store = (pid_q * TILE + tl.arange(0, TILE))[:, None]
        ptrs = OUT + q_store * stride_oq + k_idx * stride_ok
        tl.store(ptrs, result, mask=valid)

    if CLASSIFY:
        result_i = tl.where(valid, result.to(tl.int32), 0)
        has_one = tl.max(tl.max(result_i, axis=1), axis=0) != 0
        all_one = tl.min(tl.min(result_i, axis=1), axis=0) != 0
        flag = tl.where(all_one, 2, tl.where(has_one, 1, 0))
        tl.store(BLOCK_FLAGS + pid_q * stride_bf_q + pid_k * stride_bf_k, flag.to(tl.int8))


_MASK_TYPE_MAP = {
    "sparse": 0, "stair": 1, "video_stair": 2,
    "cross_sample_causal_video_bidir": 3, "full": 4,
}


def _get_mask_kernel_tables(problem, mt):
    """Extract table tensors, strides, and W/G values for create_mask_kernel based on mask type."""
    device = problem["q"].device
    t1 = t2 = t3 = torch.empty(0, device=device)
    s1 = s2 = s3 = 0
    W_val = G_val = 0

    if mt == 0:
        t1, t2, t3 = problem["segment_ids"], problem["doc_start"], problem["modality"]
        s1, s2, s3 = t1.stride(0), t2.stride(0), t3.stride(0)
        W_val = problem["sliding_window"]
        G_val = problem["global_window"]
    elif mt in (1, 2):
        t1, t2 = problem["video_ids"], problem["frame_ids"]
        s1, s2 = t1.stride(0), t2.stride(0)
    elif mt == 3:
        t1 = problem["modality"]
        s1 = t1.stride(0)
    elif mt == 4:
        t1, t3 = problem["segment_ids"], problem["modality"]
        s1, s3 = t1.stride(0), t3.stride(0)

    return t1, s1, t2, s2, t3, s3, W_val, G_val


def triton_create_mask(problem, mask_type, tile_size=MASK_BLOCK_SIZE):
    """Generate a dense ``[1, 1, SEQ_LEN, SEQ_LEN]`` bool mask via create_mask_kernel."""
    SEQ_LEN = problem["total_s"]
    device = problem["q"].device
    out = torch.empty(1, 1, SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=device)
    mt = _MASK_TYPE_MAP[mask_type]
    t1, s1, t2, s2, t3, s3, W_val, G_val = _get_mask_kernel_tables(problem, mt)

    n_tiles = (SEQ_LEN + tile_size - 1) // tile_size
    dummy_flags = torch.empty(0, dtype=torch.int8, device=device)
    create_mask_kernel[(n_tiles, n_tiles)](
        out, out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dummy_flags, 0, 0,
        t1, s1, t2, s2, t3, s3,
        SEQ_LEN, SEQ_LEN, W_val, G_val,
        Q_OFFSET=0,
        MASK_TYPE=mt, TILE=tile_size,
        STORE_MASK=True, CLASSIFY=False,
    )
    return out


def _run_create_mask_stripe(problem, mask_type, q_start, q_height, kv_len_padded,
                             out_buffer, flags_buffer=None, tile_size=MASK_BLOCK_SIZE):
    """Run create_mask_kernel for a stripe.

    If flags_buffer is provided, also classify blocks (CLASSIFY=True).
    Otherwise only generate the dense mask (CLASSIFY=False).
    """
    SEQ_LEN = problem["total_s"]
    device = problem["q"].device
    mt = _MASK_TYPE_MAP[mask_type]
    t1, s1, t2, s2, t3, s3, W_val, G_val = _get_mask_kernel_tables(problem, mt)

    n_tiles_q = q_height // tile_size
    n_tiles_k = kv_len_padded // tile_size

    if flags_buffer is not None:
        create_mask_kernel[(n_tiles_q, n_tiles_k)](
            out_buffer, out_buffer.stride(0), out_buffer.stride(1), out_buffer.stride(2), out_buffer.stride(3),
            flags_buffer, flags_buffer.stride(0), flags_buffer.stride(1),
            t1, s1, t2, s2, t3, s3,
            SEQ_LEN, SEQ_LEN, W_val, G_val,
            Q_OFFSET=q_start,
            MASK_TYPE=mt, TILE=tile_size,
            STORE_MASK=True, CLASSIFY=True,
        )
    else:
        dummy_flags = torch.empty(0, dtype=torch.int8, device=device)
        create_mask_kernel[(n_tiles_q, n_tiles_k)](
            out_buffer, out_buffer.stride(0), out_buffer.stride(1), out_buffer.stride(2), out_buffer.stride(3),
            dummy_flags, 0, 0,
            t1, s1, t2, s2, t3, s3,
            SEQ_LEN, SEQ_LEN, W_val, G_val,
            Q_OFFSET=q_start,
            MASK_TYPE=mt, TILE=tile_size,
            STORE_MASK=True, CLASSIFY=False,
        )


# ============================================================================
# Kernel: block_classify_kernel
# ============================================================================
@triton.jit(
    do_not_specialize=["stride_mq", "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "NUM_TASKS"]
)
def block_classify_kernel(
    DENSE_MASK, stride_mb, stride_mh, stride_mq, stride_mk: tl.constexpr,
    BLOCK_FLAGS, stride_fb, stride_fh, stride_fqb, stride_fkb,
    Q_LEN, KV_LEN, NUM_TASKS,
    H: tl.constexpr, Q_NUM_BLOCKS, KV_NUM_BLOCKS,
    Q_BLOCK_SIZE: tl.constexpr, KV_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    num_blocks_per_bh = Q_NUM_BLOCKS * KV_NUM_BLOCKS
    TILE_M: tl.constexpr = 64
    TILE_N: tl.constexpr = 64

    for task_id in range(pid, NUM_TASKS, num_core):
        off_bh = task_id // num_blocks_per_bh
        off_inner = task_id % num_blocks_per_bh
        off_b = (off_bh // H).to(tl.int64)
        off_h = (off_bh % H).to(tl.int64)
        off_qb = (off_inner // KV_NUM_BLOCKS).to(tl.int64)
        off_kb = (off_inner % KV_NUM_BLOCKS).to(tl.int64)

        has_one = tl.full((), 0, dtype=tl.int32)
        all_one = tl.full((), 1, dtype=tl.int32)
        mask_base = DENSE_MASK + off_b * stride_mb + off_h * stride_mh

        for m0 in range(0, Q_BLOCK_SIZE, TILE_M):
            offs_m = off_qb * Q_BLOCK_SIZE + m0 + tl.arange(0, TILE_M)
            valid_m = offs_m < Q_LEN
            for n0 in range(0, KV_BLOCK_SIZE, TILE_N):
                offs_n = off_kb * KV_BLOCK_SIZE + n0 + tl.arange(0, TILE_N)
                valid_n = offs_n < KV_LEN
                valid = valid_m[:, None] & valid_n[None, :]
                ptrs = mask_base + offs_m[:, None] * stride_mq + offs_n[None, :] * stride_mk
                vals = tl.load(ptrs, mask=valid, other=0).to(tl.int32)
                tile_any = tl.max(tl.max(tl.where(valid, vals, 0), axis=1), axis=0)
                tile_all = tl.min(tl.min(tl.where(valid, vals, 0), axis=1), axis=0)
                has_one = tl.where(tile_any != 0, 1, has_one)
                all_one = tl.where(tile_all == 0, 0, all_one)

        partial = (has_one == 1) & (all_one == 0)
        full = all_one == 1
        flag = tl.where(full, 2, tl.where(partial, 1, 0))
        out_ptr = BLOCK_FLAGS + off_b * stride_fb + off_h * stride_fh + off_qb * stride_fqb + off_kb * stride_fkb
        tl.store(out_ptr, flag.to(tl.int8))


def _classify_stripe_from_hbm(stripe_buffer, q_height, kv_len, flags_buf, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Decoupled classify: read dense mask from HBM and classify block flags.

    Uses block_classify_kernel (separate from mask generation).
    """
    Q_NUM_BLOCKS = _round_up_to_multiple(q_height, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_NUM_BLOCKS = _round_up_to_multiple(kv_len, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    num_tasks = Q_NUM_BLOCKS * KV_NUM_BLOCKS
    grid = (min(_get_num_vector_core(), max(num_tasks, 1)),)
    block_classify_kernel[grid](
        stripe_buffer, stripe_buffer.stride(0), stripe_buffer.stride(1), stripe_buffer.stride(2), stripe_buffer.stride(3),
        flags_buf, 0, 0, flags_buf.stride(0), flags_buf.stride(1),
        q_height, kv_len, NUM_TASKS=num_tasks, H=1,
        Q_NUM_BLOCKS=Q_NUM_BLOCKS, KV_NUM_BLOCKS=KV_NUM_BLOCKS,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )


# ============================================================================
# Kernel: pack_partial_blocks_kernel
# ============================================================================
@triton.jit(
    do_not_specialize=[
        "stride_mq", "stride_mk", "stride_offset_q",
        "stride_local_q", "stride_local_k",
        "stride_flag_q", "stride_flag_k",
        "stride_table_q", "stride_table_k",
        "Q_NUM_BLOCKS", "KV_NUM_BLOCKS", "TOTAL_PARTIAL",
    ]
)
def pack_partial_blocks_kernel(
    DENSE_MASK, stride_mb, stride_mh, stride_mq, stride_mk: tl.constexpr,
    BLOCK_FLAGS, stride_flag_q, stride_flag_k: tl.constexpr,
    PARTIAL_OFFSETS, stride_offset_q,
    LOCAL_IDX, stride_local_q, stride_local_k,
    PACKED_MASK, stride_packed_p, stride_packed_m, stride_packed_n: tl.constexpr,
    BLOCK_TABLE, stride_table_q, stride_table_k: tl.constexpr,
    Q_LEN, KV_LEN, Q_NUM_BLOCKS, KV_NUM_BLOCKS, TOTAL_PARTIAL,
    Q_BLOCK_SIZE: tl.constexpr, KV_BLOCK_SIZE: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)
    if pid_q >= Q_NUM_BLOCKS:
        return

    row_offset = tl.load(PARTIAL_OFFSETS + pid_q * stride_offset_q).to(tl.int32)
    offs_m_local = tl.arange(0, Q_BLOCK_SIZE)[:, None].to(tl.int64)
    offs_n_local = tl.arange(0, KV_BLOCK_SIZE)[None, :].to(tl.int64)

    for kv_idx in range(KV_NUM_BLOCKS):
        flag = tl.load(BLOCK_FLAGS + pid_q * stride_flag_q + kv_idx * stride_flag_k).to(tl.int32)
        is_partial = flag == 1
        if is_partial:
            local_idx = tl.load(LOCAL_IDX + pid_q * stride_local_q + kv_idx * stride_local_k).to(tl.int32)
            packed_idx = (row_offset + local_idx - 1).to(tl.int64)
            offs_m = (pid_q * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE))[:, None].to(tl.int64)
            offs_n = (kv_idx * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE))[None, :].to(tl.int64)
            valid_src = (offs_m < Q_LEN) & (offs_n < KV_LEN)
            src_ptrs = DENSE_MASK + offs_m * stride_mq + offs_n * stride_mk
            block = tl.load(src_ptrs, mask=valid_src, other=0)
            dst_ptrs = PACKED_MASK + packed_idx * stride_packed_p + offs_m_local * stride_packed_m + offs_n_local * stride_packed_n
            tl.store(dst_ptrs, block)
            tl.store(BLOCK_TABLE + pid_q * stride_table_q + kv_idx * stride_table_k, packed_idx.to(tl.int32))


# ============================================================================
# Strategy 1: Kernel-based streaming packed BlockMask builder
# ============================================================================
def _build_packed_block_mask_streaming(mask_type_str, problem, SEQ_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
                                                stripe_q_blocks=None, classify_strategy="fused"):
    """Build a packed BlockMask using custom Triton kernels with streaming stripes.

    Args:
        mask_type_str: Mask type string key into ``_MASK_TYPE_MAP``
            (e.g. "sparse", "full", "stair", "video_stair",
            "cross_sample_causal_video_bidir").
        problem: Problem dict containing q/k/v tensors and index tables
            (segment_ids, modality, doc_start, video_ids, frame_ids, etc.).
        SEQ_LEN: Total sequence length (Q and KV are assumed equal).
        Q_BLOCK_SIZE: Sparse block size along the query dimension.
        KV_BLOCK_SIZE: Sparse block size along the key/value dimension.
        stripe_q_blocks: Number of Q blocks per streaming stripe.  If None,
            auto-computed to target ~256 MB per stripe.
        classify_strategy: "fused" (classify inside create_mask_kernel) or
            "decoupled" (separate block_classify_kernel pass).

    Returns:
        BlockMask with ``packed_partial_mask``, ``partial_mask_offsets``,
        and ``partial_block_table`` attributes set.
    """
    device = problem["q"].device
    Q_NUM_BLOCKS = _round_up_to_multiple(SEQ_LEN, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_NUM_BLOCKS = _round_up_to_multiple(SEQ_LEN, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    KV_LEN_PADDED = KV_NUM_BLOCKS * KV_BLOCK_SIZE

    if stripe_q_blocks is None:
        max_rows = max(1, _STRIPE_TARGET_BYTES // KV_LEN_PADDED)
        stripe_q_blocks = max(1, max_rows // Q_BLOCK_SIZE)
    stripe_q_blocks = min(stripe_q_blocks, Q_NUM_BLOCKS)

    max_stripe_height = stripe_q_blocks * Q_BLOCK_SIZE
    stripe_buffer = torch.zeros(1, 1, max_stripe_height, KV_LEN_PADDED, dtype=torch.bool, device=device)

    block_flags = torch.zeros((1, 1, Q_NUM_BLOCKS, KV_NUM_BLOCKS), device=device, dtype=torch.int8)
    flags_stripe_buf = torch.empty((stripe_q_blocks, KV_NUM_BLOCKS), device=device, dtype=torch.int8)
    partial_block_table = torch.full((Q_NUM_BLOCKS, KV_NUM_BLOCKS), -1, dtype=torch.int32, device=device)
    global_B = torch.zeros(Q_NUM_BLOCKS, dtype=torch.int32, device=device)

    stripe_caches = []
    stripe_meta = []
    running_total = 0

    for qs_block in range(0, Q_NUM_BLOCKS, stripe_q_blocks):
        qe_block = min(qs_block + stripe_q_blocks, Q_NUM_BLOCKS)
        q_start = qs_block * Q_BLOCK_SIZE
        q_height = (qe_block - qs_block) * Q_BLOCK_SIZE
        stripe_q_nb = qe_block - qs_block

        if qe_block >= Q_NUM_BLOCKS:
            stripe_buffer.zero_()

        if classify_strategy == "fused":
            _run_create_mask_stripe(
                problem, mask_type_str, q_start, q_height, KV_LEN_PADDED,
                stripe_buffer, flags_stripe_buf, tile_size=MASK_BLOCK_SIZE
            )
        elif classify_strategy == "decoupled":
            _run_create_mask_stripe(
                problem, mask_type_str, q_start, q_height, KV_LEN_PADDED, stripe_buffer, tile_size=MASK_BLOCK_SIZE
            )
            _classify_stripe_from_hbm(
                stripe_buffer, q_height, SEQ_LEN, flags_stripe_buf, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            )
        else:
            raise ValueError(f"Unknown classify_strategy: {classify_strategy}")

        block_flags[:, :, qs_block:qe_block, :] = flags_stripe_buf[:stripe_q_nb, :].unsqueeze(0).unsqueeze(0)

        flags_stripe = flags_stripe_buf[:stripe_q_nb, :].contiguous()
        A_stripe = (flags_stripe == 1).to(torch.int32).cumsum(dim=-1)
        B_stripe = A_stripe.max(dim=-1).values
        stripe_partial_count = int(B_stripe.sum().item())
        global_B[qs_block:qe_block] = B_stripe.to(torch.int32)

        if stripe_partial_count > 0:
            stripe_cache = torch.zeros(
                (stripe_partial_count, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device,
            )
            row_offset_local = (B_stripe.cumsum(dim=-1) - B_stripe).to(torch.int32).contiguous()
            local_idx_stripe = A_stripe.contiguous()
            table_stripe = partial_block_table[qs_block:qe_block, :]

            pack_partial_blocks_kernel[(stripe_q_nb,)](
                stripe_buffer, stripe_buffer.stride(0), stripe_buffer.stride(1), stripe_buffer.stride(2), stripe_buffer.stride(3),
                flags_stripe, flags_stripe.stride(0), flags_stripe.stride(1),
                row_offset_local, row_offset_local.stride(0),
                local_idx_stripe, local_idx_stripe.stride(0), local_idx_stripe.stride(1),
                stripe_cache, stripe_cache.stride(0), stripe_cache.stride(1), stripe_cache.stride(2),
                table_stripe, table_stripe.stride(0), table_stripe.stride(1),
                q_height, SEQ_LEN, Q_NUM_BLOCKS=stripe_q_nb, KV_NUM_BLOCKS=KV_NUM_BLOCKS, TOTAL_PARTIAL=stripe_partial_count,
                Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE,
            )
            stripe_caches.append(stripe_cache)
        else:
            stripe_caches.append(
                torch.zeros((0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device)
            )

        stripe_meta.append((qs_block, qe_block, running_total))
        running_total += stripe_partial_count

    del stripe_buffer

    total_partial = running_total

    if total_partial > 0:
        packed_partial_mask = torch.cat(stripe_caches, dim=0)
        for qs_block_i, qe_block_i, cache_offset_i in stripe_meta:
            if cache_offset_i > 0:
                table_slice = partial_block_table[qs_block_i:qe_block_i, :]
                valid = table_slice >= 0
                if valid.any():
                    table_slice[valid] += cache_offset_i
    else:
        packed_partial_mask = torch.zeros(
            (0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device,
        )

    del stripe_caches

    partial_mask_offsets_3d = (global_B.cumsum(dim=-1) - global_B).view(1, 1, Q_NUM_BLOCKS).contiguous()

    partial_bm = (block_flags == 1).to(dtype=torch.int8)
    full_bm = (block_flags == 2).to(dtype=torch.int8)
    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_bm, full_bm), 2, (SEQ_LEN, SEQ_LEN), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets_3d
    packed_block_mask.partial_block_table = partial_block_table

    del block_flags, global_B, partial_bm, full_bm
    return packed_block_mask


# ============================================================================
# Strategy 2: mask_mod-based streaming packed BlockMask builder
# ============================================================================
# Pipeline (per Q-block stripe):
#   1. _generate_stripe_mask       - evaluate mask_mod for a stripe of Q rows
#   2. _classify_stripe_blocks     - classify each (Q, KV) block as full/partial/empty
#   3. _pack_stripe_partial_blocks - extract partial blocks into packed cache + table
#   4. _assemble_packed_block_mask - merge all stripes into final BlockMask


def _generate_stripe_mask(mask_mod, q_start, actual_q, KV_LEN, B, H, device):
    """Evaluate ``mask_mod`` for a horizontal stripe of Q rows.

    Returns a bool tensor of shape ``[B, H, actual_q, KV_LEN]``.

    For the common B=1/H=1 case we call ``mask_mod`` directly with 2-D index
    tensors, which avoids the Python overhead of ``create_mask``'s vmap stack.
    For multi-batch/multi-head we fall back to ``create_mask`` with a shifted
    mask_mod closure.
    """
    if B == 1 and H == 1:
        q_idx = torch.arange(q_start, q_start + actual_q, device=device, dtype=torch.int64)[:, None]
        kv_idx = torch.arange(0, KV_LEN, device=device, dtype=torch.int64)[None, :]
        mask_2d = mask_mod(0, 0, q_idx, kv_idx)
        return mask_2d.view(1, 1, actual_q, KV_LEN)

    def _shifted_mm(b, h, q_idx, kv_idx, _mm=mask_mod, _offset=q_start):
        return _mm(b, h, q_idx + _offset, kv_idx)

    return _torch_create_mask(_shifted_mm, B, H, actual_q, KV_LEN, device=device)


def _classify_stripe_blocks(stripe_mask, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Classify each (Q-block, KV-block) tile as full / partial / empty.

    Args:
        stripe_mask: bool tensor ``[B, H, stripe_q, KV_LEN_PADDED]`` whose Q and
            KV dimensions are already padded to multiples of block sizes.

    Returns:
        flags: int8 tensor ``[stripe_q_nb, KV_num_blocks]`` where
            0 = empty, 1 = partial, 2 = full.
    """
    stripe_q_nb = stripe_mask.shape[2] // Q_BLOCK_SIZE
    kv_num_blocks = stripe_mask.shape[3] // KV_BLOCK_SIZE

    partial_dense, full_dense = _convert_mask_to_block_mask(
        stripe_mask,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=True,
    )

    flags = torch.zeros((stripe_q_nb, kv_num_blocks), dtype=torch.int8, device=stripe_mask.device)
    flags[partial_dense[0, 0] == 1] = 1
    flags[full_dense[0, 0] == 1] = 2
    return flags


def _pack_stripe_partial_blocks(stripe_mask, flags, qs_block, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
                                 running_total, partial_block_table):
    """Extract partial blocks from a stripe into the packed cache.

    For every (Q-block, KV-block) classified as partial, copy the
    ``[Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` tile from ``stripe_mask`` into a flat list
    and record its packed index in ``partial_block_table``.

    Returns:
        packed_tiles: ``[num_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` bool tensor.
        num_partial:  count of partial blocks in this stripe.
    """
    partial_bool = (flags == 1)
    num_partial = int(partial_bool.sum().item())
    if num_partial == 0:
        empty = torch.zeros((0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=stripe_mask.device)
        return empty, 0

    stripe_q_nb = flags.shape[0]
    kv_num_blocks = flags.shape[1]

    # Locate partial (q_blk, kv_blk) positions within this stripe.
    sq_idx, kv_blk_idx = partial_bool.nonzero(as_tuple=True)

    # Gather the actual [Q_BLOCK_SIZE, KV_BLOCK_SIZE] tiles.
    blocks = stripe_mask.view(stripe_q_nb, Q_BLOCK_SIZE, kv_num_blocks, KV_BLOCK_SIZE)
    packed_tiles = blocks[sq_idx, :, kv_blk_idx, :]

    # Compute the global packed index for each partial block.
    # Layout: partial blocks are packed row-major; each Q-block row's partial
    # count is cumulated so that row_offset_local[q] gives the starting index
    # of that row's partials within the stripe.
    cumsum_per_row = partial_bool.to(torch.int32).cumsum(dim=-1)
    per_row_count = cumsum_per_row.max(dim=-1).values
    row_offset_local = per_row_count.cumsum(dim=-1) - per_row_count
    local_idx = cumsum_per_row[sq_idx, kv_blk_idx] - 1
    packed_idx = (row_offset_local[sq_idx] + local_idx + running_total).to(torch.int32)

    # Record packed indices into the global table (offset by stripe's Q-block start).
    partial_block_table[qs_block + sq_idx, kv_blk_idx] = packed_idx

    return packed_tiles, num_partial


def _assemble_packed_block_mask(block_flags, packed_partial_mask, partial_block_table,
                                 global_per_row_count, Q_LEN, KV_LEN,
                                 Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Assemble the final BlockMask from streaming stripe outputs.

    Args:
        block_flags:          ``[B, H, Q_nb, KV_nb]`` int8 (0/1/2).
        packed_partial_mask:  ``[total_partial, Q_BLOCK_SIZE, KV_BLOCK_SIZE]`` bool.
        partial_block_table:  ``[Q_nb, KV_nb]`` int32 (packed index or -1).
        global_per_row_count: ``[Q_nb]`` int32, partial count per Q-block row.
    """
    Q_num_blocks = block_flags.shape[2]

    # partial_mask_offsets[q] = cumulative partial count before row q.
    partial_mask_offsets = (
        (global_per_row_count.cumsum(dim=-1) - global_per_row_count)
        .view(1, 1, Q_num_blocks).contiguous()
    )

    partial_bm = (block_flags == 1).to(dtype=torch.int8)
    full_bm = (block_flags == 2).to(dtype=torch.int8)

    packed_block_mask = _create_sparse_block_from_block_mask(
        (partial_bm, full_bm), 2, (Q_LEN, KV_LEN), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )
    packed_block_mask.packed_partial_mask = packed_partial_mask
    packed_block_mask.partial_mask_offsets = partial_mask_offsets
    packed_block_mask.partial_block_table = partial_block_table
    return packed_block_mask


def create_block_mask_patched(
    mask_mod,
    B=1,
    H=1,
    Q_LEN=None,
    KV_LEN=None,
    device=None,
    BLOCK_SIZE=128,
    stripe_q_blocks=None,
):
    """Build a packed BlockMask with streaming stripe processing.

    Parameters are aligned with ``torch.nn.attention.flex_attention.create_block_mask``.

    The mask is built incrementally: Q rows are processed in horizontal stripes,
    each stripe small enough to keep peak HBM bounded. For every stripe we
    evaluate ``mask_mod``, classify blocks (full/partial/empty), and immediately
    pack partial blocks into a flat cache. This avoids materialising the full
    ``[Q_LEN, KV_LEN]`` dense mask at any point.

    Args:
        mask_mod: A mask_mod callable ``(b, h, q_idx, kv_idx) -> bool``.
            Supports any flexible mask pattern, e.g. ``_full_mask_mod``,
            ``_cross_sample_causal_video_bidir_mask_mod``, ``_sparse_mask_mod``, etc.
        B: Batch size (default 1).
        H: Number of heads (default 1).
        Q_LEN: Query sequence length. If None, inferred from KV_LEN.
        KV_LEN: Key/value sequence length. If None, inferred from Q_LEN.
        device: Device for tensor allocation. If None, uses NPU/CUDA.
        BLOCK_SIZE: Block size as int (square) or ``(Q_BLOCK_SIZE, KV_BLOCK_SIZE)`` tuple.
        stripe_q_blocks: Number of Q blocks per streaming stripe. If None, auto-computed
            to target ~256MB per stripe. Controls HBM peak consumption.

    Returns:
        BlockMask with ``packed_partial_mask``, ``partial_mask_offsets``,
        and ``partial_block_table`` attributes set.
    """
    # ---- Resolve parameters --------------------------------------------------
    if device is None:
        device = _get_torch_device()
    if Q_LEN is None and KV_LEN is not None:
        Q_LEN = KV_LEN
    if KV_LEN is None and Q_LEN is not None:
        KV_LEN = Q_LEN
    assert Q_LEN is not None and KV_LEN is not None, "Q_LEN and KV_LEN must be provided"

    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE, BLOCK_SIZE
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

    Q_num_blocks = _round_up_to_multiple(Q_LEN, Q_BLOCK_SIZE) // Q_BLOCK_SIZE
    KV_num_blocks = _round_up_to_multiple(KV_LEN, KV_BLOCK_SIZE) // KV_BLOCK_SIZE
    KV_LEN_padded = KV_num_blocks * KV_BLOCK_SIZE

    # ---- Determine stripe size (controls HBM peak) ---------------------------
    if stripe_q_blocks is None:
        max_rows = max(1, _STRIPE_TARGET_BYTES // (KV_LEN_padded * _BYTES_PER_MASK_ELEMENT))
        stripe_q_blocks = max(1, max_rows // Q_BLOCK_SIZE)
    stripe_q_blocks = min(stripe_q_blocks, Q_num_blocks)

    # ---- Allocate accumulators ----------------------------------------------
    block_flags = torch.zeros((B, H, Q_num_blocks, KV_num_blocks), device=device, dtype=torch.int8)
    partial_block_table = torch.full((Q_num_blocks, KV_num_blocks), -1, dtype=torch.int32, device=device)
    global_per_row_count = torch.zeros(Q_num_blocks, dtype=torch.int32, device=device)
    packed_tiles_list = []
    running_total = 0

    # ---- Process stripes -----------------------------------------------------
    for qs_block in range(0, Q_num_blocks, stripe_q_blocks):
        qe_block = min(qs_block + stripe_q_blocks, Q_num_blocks)
        q_start = qs_block * Q_BLOCK_SIZE
        stripe_q = (qe_block - qs_block) * Q_BLOCK_SIZE
        actual_q = min(stripe_q, Q_LEN - q_start)

        # Step 1: generate dense mask for this stripe's Q rows.
        stripe_mask = _generate_stripe_mask(mask_mod, q_start, actual_q, KV_LEN, B, H, device)

        # Pad Q/KV to block boundaries so classification is exact.
        pad_q = stripe_q - actual_q
        pad_kv = KV_LEN_padded - KV_LEN
        if pad_q > 0 or pad_kv > 0:
            stripe_mask = _F.pad(stripe_mask, (0, pad_kv, 0, pad_q))

        # Step 2: classify each (Q-block, KV-block) tile.
        flags = _classify_stripe_blocks(stripe_mask, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        block_flags[:, :, qs_block:qe_block, :] = flags

        # Record per-row partial counts for offset computation later.
        partial_bool = (flags == 1)
        per_row_count = partial_bool.to(torch.int32).cumsum(dim=-1).max(dim=-1).values
        global_per_row_count[qs_block:qe_block] = per_row_count.to(torch.int32)

        # Step 3: pack partial blocks into flat cache + update table.
        packed_tiles, num_partial = _pack_stripe_partial_blocks(
            stripe_mask, flags, qs_block, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            running_total, partial_block_table,
        )
        packed_tiles_list.append(packed_tiles)
        running_total += num_partial

        del stripe_mask, flags, partial_bool, per_row_count

    # ---- Merge stripe caches -------------------------------------------------
    if running_total > 0:
        packed_partial_mask = torch.cat(packed_tiles_list, dim=0)
    else:
        packed_partial_mask = torch.zeros(
            (0, Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtype=torch.bool, device=device,
        )
    del packed_tiles_list

    # Step 4: assemble final BlockMask with packed attributes.
    packed_block_mask = _assemble_packed_block_mask(
        block_flags, packed_partial_mask, partial_block_table,
        global_per_row_count, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
    )

    del block_flags, global_per_row_count
    return packed_block_mask