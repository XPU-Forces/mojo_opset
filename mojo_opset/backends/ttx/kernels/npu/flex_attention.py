from typing import Optional
from typing import Tuple

import torch

import triton
import triton.language as tl

from .utils import get_num_cores
from .utils import is_910


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
        tl.extra.cann.extension.compile_hint(q, "cv_pipeline_lazy_load", True)

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
            tl.extra.cann.extension.compile_hint(k, "cv_pipeline_lazy_load", True)
            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN),
                other=0.0
            )
            tl.extra.cann.extension.compile_hint(v, "cv_pipeline_lazy_load", True)
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
                tl.extra.cann.extension.compile_hint(k, "cv_pipeline_lazy_load", True)
                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN),
                    other=0.0
                )
                tl.extra.cann.extension.compile_hint(v, "cv_pipeline_lazy_load", True)
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
        tl.extra.cann.extension.compile_hint(q, "cv_pipeline_lazy_load", True)
        do = tl.load(
            DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
            mask=(offs_m[:, None] < Q_LEN),
            other=0.0,
        )
        tl.extra.cann.extension.compile_hint(do, "cv_pipeline_lazy_load", True)

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
                tl.extra.cann.extension.compile_hint(k, "cv_pipeline_lazy_load", True)
                v = tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n[:, None] < KV_LEN),
                    other=0.0,
                )
                tl.extra.cann.extension.compile_hint(v, "cv_pipeline_lazy_load", True)

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
                    tl.extra.cann.extension.compile_hint(k, "cv_pipeline_lazy_load", True)
                    v = tl.load(
                        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=(offs_n[:, None] < KV_LEN),
                        other=0.0,
                    )
                    tl.extra.cann.extension.compile_hint(v, "cv_pipeline_lazy_load", True)

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
            tl.extra.cann.extension.compile_hint(k, "cv_pipeline_lazy_load", True)
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )
            tl.extra.cann.extension.compile_hint(v, "cv_pipeline_lazy_load", True)

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
    tl.extra.cann.extension.compile_hint(q, "cv_pipeline_lazy_load", True)
    do = tl.load(
        DO + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    tl.extra.cann.extension.compile_hint(do, "cv_pipeline_lazy_load", True)
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
        enable_preload=True,
        enable_dynamic_cv_pipeline=False,
    )

    return output, lse


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
        enable_preload=True,
        enable_dynamic_cv_pipeline=False,
    )

    BLOCK_M_DKDV = TILE_BLOCK_SIZE
    BLOCK_N_DKDV = TILE_BLOCK_SIZE
    NUM_KV_SUB_BLOCKS_VAL = SPARSE_KV_BLOCK_SIZE // BLOCK_N_DKDV
    num_kv_blocks = triton.cdiv(N, SPARSE_KV_BLOCK_SIZE)
    grid_dkv, num_tasks_dkv = _persistent_launch_config(num_kv_blocks * Z * Hkv)
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
        enable_preload=True,
        enable_dynamic_cv_pipeline=False,
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
