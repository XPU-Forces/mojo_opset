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
    block_mask,
    idx_b,
    idx_n,
    idx_h,
    S: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_k = (
        k
        + idx_b * STRIDE_K_B
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + idx_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + idx_b * STRIDE_V_B
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + idx_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < S
    block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
    block_s = tl.dot(block_q, block_k.T) * scale
    if block_mask is not None:
        block_s += block_mask
    block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
    block_s = tl.exp(block_s - block_m_1[:, None])
    block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)

    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    block_o = tl.exp(block_m - block_m_1)[:, None].to(LOW_TYPE) * block_o + tl.dot(block_s.to(LOW_TYPE), block_v).to(
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
    block_mask,
    idx_b,
    idx_n,
    idx_h,
    S: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_k = (
        k
        + idx_b * STRIDE_K_B
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + idx_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + idx_b * STRIDE_V_B
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + idx_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < S
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
    block_mask,
    idx_b,
    idx_n,
    idx_h,
    S: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE,
    HIGH_TYPE,
):
    ptr_q = (
        q
        + idx_b * STRIDE_Q_B
        + idx_n * STRIDE_Q_N
        + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
        + idx_h[None, :] * STRIDE_Q_H
    )
    ptr_do = (
        do
        + idx_b * STRIDE_Q_B
        + idx_n * STRIDE_Q_N
        + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
        + idx_h[None, :] * STRIDE_Q_H
    )
    ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
    ptr_lse = lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

    mask_q = (offset_r + tl.arange(0, BLOCK_R))[:, None] < S
    mask_d = (offset_r + tl.arange(0, BLOCK_R))[:] < S

    block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
    block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
    if block_mask is not None:
        block_s += block_mask
    block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
    block_p = tl.exp(block_s - block_lse[:, None])
    block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
    block_dv += tl.dot(block_p.to(LOW_TYPE).T, block_do).to(HIGH_TYPE)
    block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
    block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
    block_ds = block_p * (block_dp - block_d[:, None])
    block_dk += tl.dot(block_ds.to(LOW_TYPE).T, block_q).to(HIGH_TYPE) * scale

    return block_dk, block_dv


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            multibuffer=True,
            unit_flag=True,
            set_workspace_multibuffer=2,
            enable_hivm_auto_cv_balance=True,
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_fwd_up(
    q,
    k,
    v,
    o,
    fp32o,
    lse,
    scale: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    num_r = tl.cdiv(SEQ_LEN, BLOCK_R)
    GROUP_SIZE = N // NUM_GROUP
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)

        ptr_q = (
            q
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_o = (
            o
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_fp32o = (
            fp32o
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_lse = (
            lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_lse = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S

        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_o = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
        block_l = tl.full([BLOCK_R], 0.0, dtype=HIGH_TYPE)
        block_m = tl.full([BLOCK_R], -1e6, dtype=HIGH_TYPE)

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r == chunk_idx_c
        block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6
        block_o, block_m, block_l = micro_kernel_fwd(
            block_q,
            k,
            v,
            block_o,
            block_m,
            block_l,
            scale,
            idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
            STRIDE_V_H,
            GROUP_SIZE,
            BLOCK_R,
            LOW_TYPE,
            HIGH_TYPE,
        )

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r > chunk_idx_c
        block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6
        block_o, block_m, block_l = micro_kernel_fwd(
            block_q,
            k,
            v,
            block_o,
            block_m,
            block_l,
            scale,
            SEQ_LEN + idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
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
                SEQ_LEN + idx_tile_r * BLOCK_R,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
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
                SEQ_LEN + idx_c * BLOCK_C,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
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


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 256},
            multibuffer=True,
            unit_flag=True,
            set_workspace_multibuffer=2,
            enable_hivm_auto_cv_balance=True,
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_fwd_down(
    q,
    k,
    v,
    o,
    fp32o,
    lse,
    scale: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    num_r = tl.cdiv(SEQ_LEN, BLOCK_R)
    GROUP_SIZE = N // NUM_GROUP
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r + SEQ_LEN // BLOCK_R
        idx_h = tl.arange(0, H)

        ptr_q = (
            q
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_o = (
            o
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_fp32o = (
            fp32o
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_lse = (
            lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_lse = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S

        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_o = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
        block_l = tl.full([BLOCK_R], 0.0, dtype=HIGH_TYPE)
        block_m = tl.full([BLOCK_R], -1e6, dtype=HIGH_TYPE)

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r >= chunk_idx_c
        block_mask = (block_mask_bool.to(LOW_TYPE) - 1.0) * 1e6
        block_o, block_m, block_l = micro_kernel_fwd(
            block_q,
            k,
            v,
            block_o,
            block_m,
            block_l,
            scale,
            idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
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
                idx_tile_r * BLOCK_R,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                LOW_TYPE,
                HIGH_TYPE,
            )

        for idx_c in range(SEQ_LEN // BLOCK_C, idx_r * BLOCK_R // BLOCK_C):
            block_o, block_m, block_l = micro_kernel_fwd(
                block_q,
                k,
                v,
                block_o,
                block_m,
                block_l,
                scale,
                idx_c * BLOCK_C,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 64}),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_bwd_d(
    fp32o,
    do,
    d,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_O_B: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    for task_id in range(pid, num_r * B * N, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)
        ptr_fp32o = (
            fp32o
            + idx_b * STRIDE_O_B
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_do = (
            do
            + idx_b * STRIDE_O_B
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
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
            multibuffer=True,
            unit_flag=True,
            set_workspace_multibuffer=2,
            enable_hivm_auto_cv_balance=True,
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_bwd_q_up(
    q,
    k,
    v,
    do,
    d,
    dq,
    lse,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    GROUP_SIZE = N // num_group
    num_r = tl.cdiv(SEQ_LEN, BLOCK_R)
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)

        ptr_q = (
            q
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_do = (
            do
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_dq = (
            dq
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        ptr_lse = (
            lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S
        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
        block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
        block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
        block_dq = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r == chunk_idx_c
        block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6
        block_dq = micro_kernel_bwd_q(
            block_q,
            k,
            v,
            block_do,
            block_d,
            block_dq,
            block_lse,
            scale,
            idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
            STRIDE_V_H,
            GROUP_SIZE,
            BLOCK_R,
            LOW_TYPE,
            HIGH_TYPE,
        )

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r > chunk_idx_c
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
            SEQ_LEN + idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
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
                SEQ_LEN + idx_tile_r * BLOCK_R,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
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
                SEQ_LEN + idx_c * BLOCK_C,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_C,
                LOW_TYPE,
                HIGH_TYPE,
            )

        tl.store(ptr_dq, block_dq.to(LOW_TYPE), mask=mask_q)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64, "BLOCK_C": 128},
            multibuffer=True,
            unit_flag=True,
            set_workspace_multibuffer=2,
            enable_hivm_auto_cv_balance=True,
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_bwd_q_down(
    q,
    k,
    v,
    do,
    d,
    dq,
    lse,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    GROUP_SIZE = N // num_group
    num_r = tl.cdiv(SEQ_LEN, BLOCK_R)
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r + SEQ_LEN // BLOCK_R
        idx_h = tl.arange(0, H)

        ptr_q = (
            q
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_do = (
            do
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_dq = (
            dq
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        ptr_lse = (
            lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S
        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
        block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
        block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
        block_dq = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)

        offs_r_local = tl.arange(0, BLOCK_R)[:, None]
        offs_c_local = tl.arange(0, BLOCK_R)[None, :]
        chunk_idx_r = offs_r_local // BLOCK_SIZE
        chunk_idx_c = offs_c_local // BLOCK_SIZE
        block_mask_bool = chunk_idx_r >= chunk_idx_c
        block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6
        block_dq = micro_kernel_bwd_q(
            block_q,
            k,
            v,
            block_do,
            block_d,
            block_dq,
            block_lse,
            scale,
            idx_r * BLOCK_R,
            block_mask,
            idx_b,
            idx_n,
            idx_h,
            S,
            STRIDE_K_B,
            STRIDE_K_N,
            STRIDE_K_S,
            STRIDE_K_H,
            STRIDE_V_B,
            STRIDE_V_N,
            STRIDE_V_S,
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
                idx_tile_r * BLOCK_R,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                LOW_TYPE,
                HIGH_TYPE,
            )

        for idx_c in range(SEQ_LEN // BLOCK_C, idx_r * BLOCK_R // BLOCK_C):
            block_dq = micro_kernel_bwd_q(
                block_q,
                k,
                v,
                block_do,
                block_d,
                block_dq,
                block_lse,
                scale,
                idx_c * BLOCK_C,
                None,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_K_B,
                STRIDE_K_N,
                STRIDE_K_S,
                STRIDE_K_H,
                STRIDE_V_B,
                STRIDE_V_N,
                STRIDE_V_S,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_C,
                LOW_TYPE,
                HIGH_TYPE,
            )

        tl.store(ptr_dq, block_dq.to(LOW_TYPE), mask=mask_q)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 256, "BLOCK_C": 64},
            multibuffer=True,
            unit_flag=True,
            set_workspace_multibuffer=2,
            enable_hivm_auto_cv_balance=True,
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_bwd_kv_left(
    q,
    k,
    v,
    do,
    d,
    dk,
    dv,
    lse,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    num_c = tl.cdiv(SEQ_LEN, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
        idx_b = task_id // (num_group * num_c)
        idx_group = task_id // num_c % num_group
        idx_c = task_id % num_c
        idx_h = tl.arange(0, H)

        ptr_k = (
            k
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_v = (
            v
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )
        ptr_dk = (
            dk
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_dv = (
            dv
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )

        mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
        block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
        block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
        block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
        block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

        for idx_ingroup in range(group_size):
            idx_n = idx_group * group_size + idx_ingroup

            offs_r_local = tl.arange(0, BLOCK_C)[:, None]
            offs_c_local = tl.arange(0, BLOCK_C)[None, :]
            chunk_idx_r = offs_r_local // BLOCK_SIZE
            chunk_idx_c = offs_c_local // BLOCK_SIZE
            block_mask_bool = chunk_idx_r == chunk_idx_c
            block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6

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
                idx_c * BLOCK_C,
                block_mask,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_Q_B,
                STRIDE_Q_N,
                STRIDE_Q_S,
                STRIDE_Q_H,
                STRIDE_D_B,
                STRIDE_D_N,
                STRIDE_D_S,
                BLOCK_C,
                LOW_TYPE,
                HIGH_TYPE,
            )

        tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
        tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 128, "BLOCK_C": 64},
            multibuffer=True,
            unit_flag=True,
            set_workspace_multibuffer=2,
            enable_hivm_auto_cv_balance=True,
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_bwd_kv_right(
    q,
    k,
    v,
    do,
    d,
    dk,
    dv,
    lse,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    num_c = tl.cdiv(SEQ_LEN, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
        idx_b = task_id // (num_group * num_c)
        idx_group = task_id // num_c % num_group
        idx_c = task_id % num_c + SEQ_LEN // BLOCK_C
        idx_h = tl.arange(0, H)

        ptr_k = (
            k
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_v = (
            v
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )
        ptr_dk = (
            dk
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_dv = (
            dv
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )

        mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
        block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
        block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
        block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
        block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

        for idx_ingroup in range(group_size):
            idx_n = idx_group * group_size + idx_ingroup

            offs_r_local = tl.arange(0, BLOCK_C)[:, None]
            offs_c_local = tl.arange(0, BLOCK_C)[None, :]
            chunk_idx_r = offs_r_local // BLOCK_SIZE
            chunk_idx_c = offs_c_local // BLOCK_SIZE
            block_mask_bool = chunk_idx_r > chunk_idx_c
            block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6

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
                idx_c * BLOCK_C - SEQ_LEN,
                block_mask,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_Q_B,
                STRIDE_Q_N,
                STRIDE_Q_S,
                STRIDE_Q_H,
                STRIDE_D_B,
                STRIDE_D_N,
                STRIDE_D_S,
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
                    idx_tile_r * BLOCK_C - SEQ_LEN,
                    None,
                    idx_b,
                    idx_n,
                    idx_h,
                    S,
                    STRIDE_Q_B,
                    STRIDE_Q_N,
                    STRIDE_Q_S,
                    STRIDE_Q_H,
                    STRIDE_D_B,
                    STRIDE_D_N,
                    STRIDE_D_S,
                    BLOCK_C,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            for idx_r in range(idx_c * BLOCK_C // BLOCK_R + 1, S // BLOCK_R):
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
                    idx_r * BLOCK_R - SEQ_LEN,
                    None,
                    idx_b,
                    idx_n,
                    idx_h,
                    S,
                    STRIDE_Q_B,
                    STRIDE_Q_N,
                    STRIDE_Q_S,
                    STRIDE_Q_H,
                    STRIDE_D_B,
                    STRIDE_D_N,
                    STRIDE_D_S,
                    BLOCK_R,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            offs_r_local = tl.arange(0, BLOCK_C)[:, None]
            offs_c_local = tl.arange(0, BLOCK_C)[None, :]
            chunk_idx_r = offs_r_local // BLOCK_SIZE
            chunk_idx_c = offs_c_local // BLOCK_SIZE
            block_mask_bool = chunk_idx_r >= chunk_idx_c
            block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6

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
                idx_c * BLOCK_C,
                block_mask,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_Q_B,
                STRIDE_Q_N,
                STRIDE_Q_S,
                STRIDE_Q_H,
                STRIDE_D_B,
                STRIDE_D_N,
                STRIDE_D_S,
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
                    idx_tile_r * BLOCK_C,
                    None,
                    idx_b,
                    idx_n,
                    idx_h,
                    S,
                    STRIDE_Q_B,
                    STRIDE_Q_N,
                    STRIDE_Q_S,
                    STRIDE_Q_H,
                    STRIDE_D_B,
                    STRIDE_D_N,
                    STRIDE_D_S,
                    BLOCK_C,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

            for idx_r in range(idx_c * BLOCK_C // BLOCK_R + 1, S // BLOCK_R):
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
                    idx_r * BLOCK_R,
                    None,
                    idx_b,
                    idx_n,
                    idx_h,
                    S,
                    STRIDE_Q_B,
                    STRIDE_Q_N,
                    STRIDE_Q_S,
                    STRIDE_Q_H,
                    STRIDE_D_B,
                    STRIDE_D_N,
                    STRIDE_D_S,
                    BLOCK_R,
                    LOW_TYPE,
                    HIGH_TYPE,
                )

        tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
        tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)


def diffusion_attention_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
):
    """
    Forward computation interface:
    Args:
        q: Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        k: Key tensor (K), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        v: Value tensor (V), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        mask: Attention mask, shape [SEQ, SEQ]
        scale: Scaling factor for QK product
    Returns:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        lse: LogSumExp tensor, shape [BSZ, Q_HEAD_NUM, SEQ]
    """
    # shape constraints
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]

    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] and k.shape[3] == v.shape[3] and q.shape[3] in {64, 128}

    o = torch.empty_like(q)
    fp32o = torch.empty_like(q, dtype=torch.float32)
    lse = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_cores, _ = get_device_properties()

    kernel_sda_fwd_up[(num_cores,)](
        q,
        k,
        v,
        o,
        fp32o,
        lse,
        scale,
        k.shape[1],
        q.shape[0],
        q.shape[1],
        q.shape[2],
        q.shape[3],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        BLOCK_SIZE=32,
    )
    kernel_sda_fwd_down[(num_cores,)](
        q,
        k,
        v,
        o,
        fp32o,
        lse,
        scale,
        k.shape[1],
        q.shape[0],
        q.shape[1],
        q.shape[2],
        q.shape[3],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        BLOCK_SIZE=32,
    )
    return o, fp32o, lse


def diffusion_attention_bwd_impl(
    fp32o: torch.Tensor,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
    eval_kernel: int = 0,
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
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4 and len(lse.shape) == 3
    assert q.shape == fp32o.shape and fp32o.shape == do.shape
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]
    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] and k.shape[3] == v.shape[3] and q.shape[3] in {64, 128}
    assert q.shape[0] == lse.shape[0] and q.shape[1] == lse.shape[1] and q.shape[2] == lse.shape[2]

    num_cores, _ = get_device_properties()
    d = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if eval_kernel == 0 or eval_kernel == 1:
        kernel_sda_bwd_d[(num_cores,)](
            fp32o,
            do,
            d,
            fp32o.shape[0],
            fp32o.shape[1],
            fp32o.shape[2],
            fp32o.shape[3],
            fp32o.stride(0),
            fp32o.stride(1),
            fp32o.stride(2),
            fp32o.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
        )
    if eval_kernel == 0 or eval_kernel == 2:
        kernel_sda_bwd_q_up[(num_cores,)](
            q,
            k,
            v,
            do,
            d,
            dq,
            lse,
            scale,
            k.shape[1],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            BLOCK_SIZE=32,
        )
        kernel_sda_bwd_q_down[(num_cores,)](
            q,
            k,
            v,
            do,
            d,
            dq,
            lse,
            scale,
            k.shape[1],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            BLOCK_SIZE=32,
        )
    if eval_kernel == 0 or eval_kernel == 3:
        kernel_sda_bwd_kv_left[(num_cores,)](
            q,
            k,
            v,
            do,
            d,
            dk,
            dv,
            lse,
            scale,
            k.shape[1],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            BLOCK_SIZE=32,
        )
        kernel_sda_bwd_kv_right[(num_cores,)](
            q,
            k,
            v,
            do,
            d,
            dk,
            dv,
            lse,
            scale,
            k.shape[1],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            BLOCK_SIZE=32,
        )
    return dq, dk, dv
