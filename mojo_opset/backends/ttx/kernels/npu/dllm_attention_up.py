from functools import cache
from typing import Any
from typing import Dict
from typing import Tuple

import torch
import triton
import triton.language as tl


@cache
def get_device_properties() -> Tuple[int, int]:
    device = torch.npu.current_device()
    device_properties: Dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(device)

    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


@triton.jit
def micro_kernel_fwd(
    block_q,
    k,
    v,
    block_o,
    block_m,
    block_l,
    scale,
    offset_c,
    offset_c_ed,
    block_mask,
    idx_n,
    idx_h,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_k = (
        k
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + idx_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + idx_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < offset_c_ed
    block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
    block_k = tl.trans(block_k)
    block_s = tl.dot(block_q, block_k) * scale
    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    if block_mask is not None:
        block_s += block_mask
    block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
    block_s = tl.exp(block_s - block_m_1[:, None])
    block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)
    block_o = tl.exp(block_m - block_m_1)[:, None].to(LOW_TYPE) * block_o 
    block_o = block_o + tl.dot(block_s.to(LOW_TYPE), block_v).to(
        LOW_TYPE
    )

    return block_o, block_m_1, block_l_1


@triton.jit
def micro_kernel_bwd_q(
    block_q,
    k,
    v,
    block_do,
    block_d,
    block_dq,
    block_lse,
    scale,
    offset_c,
    offset_c_ed,
    block_mask,
    idx_n,
    idx_h,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_k = (
        k
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + idx_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + idx_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < offset_c_ed
    block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
    block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
    if block_mask is not None:
        block_s += block_mask
    block_p = tl.exp(block_s - block_lse[:, None])
    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
    block_ds = block_p * (block_dp - block_d[:, None])
    block_dq += tl.dot(block_ds.to(LOW_TYPE), block_k).to(HIGH_TYPE) * scale
    return block_dq


@triton.jit
def micro_kernel_bwd_kv(
    q,
    block_k,
    block_v,
    do,
    d,
    block_dk,
    block_dv,
    lse,
    scale,
    offset_r,
    offset_r_ed,
    block_mask,
    idx_n,
    idx_h,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_q = (
        q + idx_n * STRIDE_Q_N + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + idx_h[None, :] * STRIDE_Q_H
    )
    ptr_do = (
        do + idx_n * STRIDE_Q_N + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + idx_h[None, :] * STRIDE_Q_H
    )
    ptr_d = d + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
    ptr_lse = lse + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

    mask_q = (offset_r + tl.arange(0, BLOCK_R))[:, None] < offset_r_ed
    mask_d = (offset_r + tl.arange(0, BLOCK_R))[:] < offset_r_ed

    block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
    block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
    block_s = tl.dot(block_q, block_k).to(HIGH_TYPE) * scale
    if block_mask is not None:
        block_s += block_mask
    block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
    block_p = tl.exp(block_s - block_lse[:, None])
    block_dv += tl.dot(block_p.to(LOW_TYPE).T, block_do).to(HIGH_TYPE)
    block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
    block_dp = tl.dot(block_do, block_v).to(HIGH_TYPE)
    block_ds = block_p * (block_dp - block_d[:, None])
    block_dk += tl.dot(block_ds.to(LOW_TYPE).T, block_q).to(HIGH_TYPE) * scale

    return block_dk, block_dv


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 128},
            # multibuffer=True,
            # unit_flag=True,
            # set_workspace_multibuffer=2,
            # enable_hivm_auto_cv_balance=True,
            # tile_mix_vector_loop=2,
            # tile_mix_cube_loop=2,
        )
    ],
    key=["N", "H"],
)
@triton.jit
def kernel_da_fwd_u(
    q,
    k,
    v,
    o,
    fp32o,
    lse,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ul,
    mask_ur,
    GROUP_SIZE: tl.constexpr,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_MASK,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
    # MAX_NUM_SEQ: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    offset_r_local = tl.arange(0, BLOCK_R)[:, None]
    offset_c_local = tl.arange(0, BLOCK_R)[None, :]
    block_mask_full_ul = tl.load(mask_ul + offset_r_local * STRIDE_MASK + offset_c_local)
    block_mask_full_ur = tl.load(mask_ur + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N,
            pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            idx_h = tl.arange(0, H)

            ptr_q = (
                q
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + idx_h[None, :] * STRIDE_Q_H
            )
            ptr_o = (
                o
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + idx_h[None, :] * STRIDE_Q_H
            )
            ptr_fp32o = (
                fp32o
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + idx_h[None, :] * STRIDE_Q_H
            )
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_lse = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_o = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
            block_l = tl.full([BLOCK_R], 0.0, dtype=HIGH_TYPE)
            block_m = tl.full([BLOCK_R], -1e6, dtype=HIGH_TYPE)

            block_mask_shape = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed) &
                (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )
            block_mask_bool = (
                block_mask_full_ul & block_mask_shape
            )
            block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q,
                k,
                v,
                block_o,
                block_m,
                block_l,
                scale,
                seq_st + idx_r * BLOCK_R,
                seq_ed,
                block_mask,
                idx_n,
                idx_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                LOW_TYPE,
                HIGH_TYPE,
            )

            block_mask_bool = (
                block_mask_full_ur & block_mask_shape
            )
            block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q,
                k,
                v,
                block_o,
                block_m,
                block_l,
                scale,
                S + seq_st + idx_r * BLOCK_R,
                S + seq_ed,
                block_mask,
                idx_n,
                idx_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                LOW_TYPE,
                HIGH_TYPE,
            )

            for idx_tile_r in range(idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R, idx_r):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q,
                    k,
                    v,
                    block_o,
                    block_m,
                    block_l,
                    scale,
                    S + seq_st + idx_tile_r * BLOCK_R,
                    S + seq_ed,
                    None,
                    idx_n,
                    idx_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_R,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q,
                    k,
                    v,
                    block_o,
                    block_m,
                    block_l,
                    scale,
                    S + seq_st + idx_c * BLOCK_C,
                    S + seq_ed,
                    None,
                    idx_n,
                    idx_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_C,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            block_o = block_o / block_l[:, None]
            block_lse = tl.log(block_l) + block_m
            tl.store(ptr_o, block_o.to(LOW_TYPE), mask=mask_q)
            tl.store(ptr_fp32o, block_o, mask=mask_q)
            tl.store(ptr_lse, block_lse, mask=mask_lse)

        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64},
        ),
    ],
    key=["N", "H"],
)
@triton.jit
def kernel_da_bwd_d(
    fp32o,
    do,
    d,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    for task_id in range(pid, num_r * N, tl.num_programs(axis=0)):
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)
        ptr_fp32o = (
            fp32o
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_do = (
            do
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_d = d + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        mask_o = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S

        block_o = tl.load(ptr_fp32o, mask=mask_o, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_o, other=0.0)
        block_d = tl.sum(block_do.to(HIGH_TYPE) * block_o, axis=1)
        tl.store(ptr_d, block_d, mask=mask_d)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 128},
            # multibuffer=True,
            # unit_flag=True,
            # set_workspace_multibuffer=2,
            # enable_hivm_auto_cv_balance=True,
            # tile_mix_vector_loop=2,
            # tile_mix_cube_loop=2,
        )
    ],
    key=["N", "H"],
)
@triton.jit
def kernel_da_bwd_q_u(
    q,
    k,
    v,
    do,
    d,
    lse,
    dq,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ur,
    mask_ul,
    GROUP_SIZE: tl.constexpr,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    offset_r_local = tl.arange(0, BLOCK_R)[:, None]
    offset_c_local = tl.arange(0, BLOCK_R)[None, :]
    block_mask_full_ul = tl.load(mask_ul + offset_r_local * STRIDE_MASK + offset_c_local)
    block_mask_full_ur = tl.load(mask_ur + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N,
            pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            idx_h = tl.arange(0, H)

            ptr_q = (
                q
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + idx_h[None, :] * STRIDE_Q_H
            )
            ptr_do = (
                do
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + idx_h[None, :] * STRIDE_Q_H
            )
            ptr_dq = (
                dq
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + idx_h[None, :] * STRIDE_Q_H
            )
            ptr_d = d + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            ptr_lse = lse + idx_n * STRIDE_D_N + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < seq_ed
            mask_d = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
            block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
            block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
            block_dq = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
            block_mask_shape = (
                (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed) &
                (seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed)
            )
            block_mask_bool = (
                block_mask_full_ul & block_mask_shape
            )
            block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6

            block_dq = micro_kernel_bwd_q(
                block_q,
                k,
                v,
                block_do,
                block_d,
                block_dq,
                block_lse,
                scale,
                seq_st + idx_r * BLOCK_R,
                seq_ed,
                block_mask,
                idx_n,
                idx_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                LOW_TYPE,
                HIGH_TYPE,
            )

            block_mask_bool = (
                block_mask_full_ur & block_mask_shape
            )
            block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6
            block_dq = micro_kernel_bwd_q(
                block_q,
                k,
                v,
                block_do,
                block_d,
                block_dq,
                block_lse,
                scale,
                S + seq_st + idx_r * BLOCK_R,
                S + seq_ed,
                block_mask,
                idx_n,
                idx_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                LOW_TYPE,
                HIGH_TYPE,
            )

            for idx_tile_r in range(idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R, idx_r):
                block_dq = micro_kernel_bwd_q(
                    block_q,
                    k,
                    v,
                    block_do,
                    block_d,
                    block_dq,
                    block_lse,
                    scale,
                    S + seq_st + idx_tile_r * BLOCK_R,
                    S + seq_ed,
                    None,
                    idx_n,
                    idx_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_R,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
                block_dq = micro_kernel_bwd_q(
                    block_q,
                    k,
                    v,
                    block_do,
                    block_d,
                    block_dq,
                    block_lse,
                    scale,
                    S + seq_st + idx_c * BLOCK_C,
                    S + seq_ed,
                    None,
                    idx_n,
                    idx_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_C,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            tl.store(ptr_dq, block_dq.to(LOW_TYPE), mask=mask_q)
        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 256, "BLOCK_C": 64},
            # multibuffer=True,
            # unit_flag=True,
            # set_workspace_multibuffer=2,
            # enable_hivm_auto_cv_balance=True,
            # tile_mix_vector_loop=2,
            # tile_mix_cube_loop=2,
        )
    ],
    key=["N", "H"],
)
@triton.jit
def kernel_da_bwd_kv_ul(
    q,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ul,
    GROUP_SIZE: tl.constexpr,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_c_st = 0
    NUM_GROUP = N // GROUP_SIZE

    offs_r_local = tl.arange(0, BLOCK_C)[:, None]
    offs_c_local = tl.arange(0, BLOCK_C)[None, :]
    block_mask_full_ul = tl.load(mask_ul + offs_r_local * STRIDE_MASK + offs_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_c_ed = offset_block_c_st + tl.cdiv(seq_ed - seq_st, BLOCK_C)
        for task_id in range(
            offset_block_c_st * NUM_GROUP + ((pid % pnum - offset_block_c_st * NUM_GROUP % pnum + pnum) % pnum),
            offset_block_c_ed * NUM_GROUP,
            pnum,
        ):
            idx_c = task_id // NUM_GROUP - offset_block_c_st
            idx_group = task_id % NUM_GROUP
            idx_h = tl.arange(0, H)

            ptr_k = (
                k
                + idx_group * STRIDE_K_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_group * STRIDE_V_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )
            ptr_dk = (
                dk
                + idx_group * STRIDE_K_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            ptr_dv = (
                dv
                + idx_group * STRIDE_V_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )
            mask_kv = (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < seq_ed

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
            block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

            block_k = tl.trans(block_k)
            block_v = tl.trans(block_v)

            for idx_ingroup in range(GROUP_SIZE):
                idx_n = idx_group * GROUP_SIZE + idx_ingroup

                block_mask_bool = (
                    block_mask_full_ul
                    & (seq_st + idx_c * BLOCK_C + offs_r_local < seq_ed)
                    & (seq_st + idx_c * BLOCK_C + offs_c_local < seq_ed)
                )
                block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6

                block_dk, block_dv = micro_kernel_bwd_kv(
                    q,
                    block_k,
                    block_v,
                    do,
                    d,
                    block_dk,
                    block_dv,
                    lse,
                    scale,
                    seq_st + idx_c * BLOCK_C,
                    seq_ed,
                    block_mask,
                    idx_n,
                    idx_h,
                    STRIDE_Q_S,
                    STRIDE_Q_N,
                    STRIDE_Q_H,
                    STRIDE_D_S,
                    STRIDE_D_N,
                    BLOCK_C,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
            tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)
        seq_st = seq_ed
        offset_block_c_st = offset_block_c_ed


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 128, "BLOCK_C": 64},
            # multibuffer=True,
            # unit_flag=True,
            # set_workspace_multibuffer=2,
            # enable_hivm_auto_cv_balance=True,
            # tile_mix_vector_loop=2,
            # tile_mix_cube_loop=2,
        )
    ],
    key=["N", "H"],
)
@triton.jit
def kernel_da_bwd_kv_ur(
    q,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ur,
    GROUP_SIZE: tl.constexpr,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_c_st = 0
    NUM_GROUP = N // GROUP_SIZE

    offs_r_local = tl.arange(0, BLOCK_C)[:, None]
    offs_c_local = tl.arange(0, BLOCK_C)[None, :]
    block_mask_full_ur = tl.load(mask_ur + offs_r_local * STRIDE_MASK + offs_c_local)
    
    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_c_ed = offset_block_c_st + tl.cdiv(seq_ed - seq_st, BLOCK_C)
        for task_id in range(
            offset_block_c_st * NUM_GROUP + ((pid % pnum - offset_block_c_st * NUM_GROUP % pnum + pnum) % pnum),
            offset_block_c_ed * NUM_GROUP,
            pnum,
        ):
            idx_c = task_id // NUM_GROUP - offset_block_c_st
            idx_group = task_id % NUM_GROUP
            idx_h = tl.arange(0, H)

            ptr_k = (
                k
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )
            ptr_dk = (
                dk
                + idx_group * STRIDE_K_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            ptr_dv = (
                dv
                + idx_group * STRIDE_V_N
                + (S + seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )
            mask_kv = (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < seq_ed

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
            block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

            block_k = tl.trans(block_k)
            block_v = tl.trans(block_v)

            for idx_ingroup in range(GROUP_SIZE):
                idx_n = idx_group * GROUP_SIZE + idx_ingroup

                block_mask_bool = (
                    block_mask_full_ur
                    & (seq_st + idx_c * BLOCK_C + offs_r_local < seq_ed)
                    & (seq_st + idx_c * BLOCK_C + offs_c_local < seq_ed)
                )
                block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6

                block_dk, block_dv = micro_kernel_bwd_kv(
                    q,
                    block_k,
                    block_v,
                    do,
                    d,
                    block_dk,
                    block_dv,
                    lse,
                    scale,
                    seq_st + idx_c * BLOCK_C,
                    seq_ed,
                    block_mask,
                    idx_n,
                    idx_h,
                    STRIDE_Q_S,
                    STRIDE_Q_N,
                    STRIDE_Q_H,
                    STRIDE_D_S,
                    STRIDE_D_N,
                    BLOCK_C,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

                for idx_tile_r in range(idx_c + 1, (idx_c * BLOCK_C // BLOCK_R + 1) * BLOCK_R // BLOCK_C):
                    block_dk, block_dv = micro_kernel_bwd_kv(
                        q,
                        block_k,
                        block_v,
                        do,
                        d,
                        block_dk,
                        block_dv,
                        lse,
                        scale,
                        seq_st + idx_tile_r * BLOCK_C,
                        seq_ed,
                        None,
                        idx_n,
                        idx_h,
                        STRIDE_Q_S,
                        STRIDE_Q_N,
                        STRIDE_Q_H,
                        STRIDE_D_S,
                        STRIDE_D_N,
                        BLOCK_C,
                        LOW_TYPE,
                        HIGH_TYPE,
                    )

                for idx_r in range(idx_c * BLOCK_C // BLOCK_R + 1, (seq_ed - seq_st + BLOCK_R - 1) // BLOCK_R):
                    block_dk, block_dv = micro_kernel_bwd_kv(
                        q,
                        block_k,
                        block_v,
                        do,
                        d,
                        block_dk,
                        block_dv,
                        lse,
                        scale,
                        seq_st + idx_r * BLOCK_R,
                        seq_ed,
                        None,
                        idx_n,
                        idx_h,
                        STRIDE_Q_S,
                        STRIDE_Q_N,
                        STRIDE_Q_H,
                        STRIDE_D_S,
                        STRIDE_D_N,
                        BLOCK_R,
                        LOW_TYPE,
                        HIGH_TYPE,
                    )

            tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
            tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)
        seq_st = seq_ed
        offset_block_c_st = offset_block_c_ed


def dllm_attention_up_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float = 1.0,
    BLOCK_SIZE: int = 8,
):
    """
    Forward computation interface:
    Args:
        q: Query tensor (Q), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM]
        k: Key tensor (K), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM]
        v: Value tensor (V), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM]
        cu_seqlen: Cumulative sequence lengths, shape [BSZ], with no leading zero.
        scale: Scaling factor for QK product
    Returns:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        o_f32: Attention output tensor in fp32, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        lse: LogSumExp tensor, shape [BSZ, Q_HEAD_NUM, SEQ]
    """

    # shape constraints
    assert len(q.shape) == 3 and len(k.shape) == 3 and len(v.shape) == 3
    assert (
        q.shape[0] == k.shape[0]
        and k.shape[0] == v.shape[0]
        and q.shape[0] >= cu_seqlen[cu_seqlen.shape[0] - 1].cpu().item()  # Only check in test.
    )
    assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] in {64, 128}

    o = torch.zeros_like(q)
    o_f32 = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.zeros((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    num_cores, _ = get_device_properties()

    if (not hasattr(dllm_attention_up_fwd_impl, "inited")):
        dllm_attention_up_fwd_impl.inited = True
        BLOCK_MASK = 64
        offset_r_local = torch.arange(0, BLOCK_MASK)[:, None]
        offset_c_local = torch.arange(0, BLOCK_MASK)[None, :]
        chunk_idx_r = offset_r_local // BLOCK_SIZE
        chunk_idx_c = offset_c_local // BLOCK_SIZE
        dllm_attention_up_fwd_impl.mask_ur = (chunk_idx_r > chunk_idx_c).to(q.device)
        dllm_attention_up_fwd_impl.mask_ul = (chunk_idx_r == chunk_idx_c).to(q.device)

    kernel_da_fwd_u[(num_cores,)](
        q,
        k,
        v,
        o,
        o_f32,
        lse,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        dllm_attention_up_fwd_impl.mask_ul,
        dllm_attention_up_fwd_impl.mask_ur,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
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
        dllm_attention_up_fwd_impl.mask_ul.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return o, o_f32, lse


def dllm_attention_up_bwd_impl(
    fp32o: torch.Tensor,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float = 1.0,
    BLOCK_SIZE: int = 8,
):
    """
    Backward computation interface:
    Args:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        do: Gradient tensor for o, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        q: Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        k: Key tensor (K), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        v: Value tensor (V), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        lse: Logsumexp tensor, shape [BSZ, Q_HEAD_NUM, SEQ]
        mask: Attention mask, shape [SEQ, SEQ]
        scale: Scaling factor for QK product
    Returns:
        dq: Gradient tensor for Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        dk: Gradient tensor for Key tensor (K), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        dv: Gradient tensor for Value tensor (V), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
    """

    # shape constraints
    assert len(q.shape) == 3 and len(k.shape) == 3 and len(v.shape) == 3
    assert (
        q.shape[0] == k.shape[0]
        and k.shape[0] == v.shape[0]
        and q.shape[0] >= cu_seqlen[cu_seqlen.shape[0] - 1].cpu().item()  # Only check in test.
    )
    assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] in {64, 128}
    assert q.shape[0] == lse.shape[0] and q.shape[1] == lse.shape[1]

    num_cores, num_vectorcore = get_device_properties()
    d = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if (not hasattr(dllm_attention_up_bwd_impl, "inited")):
        dllm_attention_up_bwd_impl.inited = True
        BLOCK_MASK = 64
        offset_r_local = torch.arange(0, BLOCK_MASK)[:, None]
        offset_c_local = torch.arange(0, BLOCK_MASK)[None, :]
        chunk_idx_r = offset_r_local // BLOCK_SIZE
        chunk_idx_c = offset_c_local // BLOCK_SIZE
        dllm_attention_up_bwd_impl.mask_ul = (chunk_idx_r == chunk_idx_c).to(q.device)
        dllm_attention_up_bwd_impl.mask_ur = (chunk_idx_r > chunk_idx_c).to(q.device)

    kernel_da_bwd_d[(num_vectorcore,)](
        fp32o,
        do,
        d,
        fp32o.shape[0],
        fp32o.shape[1],
        fp32o.shape[2],
        fp32o.stride(0),
        fp32o.stride(1),
        fp32o.stride(2),
        d.stride(0),
        d.stride(1),
    )
    kernel_da_bwd_q_u[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dq,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        dllm_attention_up_bwd_impl.mask_ur,
        dllm_attention_up_bwd_impl.mask_ul,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        dllm_attention_up_bwd_impl.mask_ul.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    kernel_da_bwd_kv_ul[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dk,
        dv,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        dllm_attention_up_bwd_impl.mask_ul,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        dllm_attention_up_bwd_impl.mask_ul.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    kernel_da_bwd_kv_ur[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dk,
        dv,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        dllm_attention_up_bwd_impl.mask_ur,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        dllm_attention_up_bwd_impl.mask_ul.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return dq, dk, dv
