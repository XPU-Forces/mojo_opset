# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# ILU Triton grouped matmul (aligned with NPU group_gemm; launch uses ILU vector cores).

import torch
import triton
import triton.language as tl

from .utils import smart_triton_autotune

# Target upper bound on grid[0]; tiles per program ~= ceil(n / TARGET_PROGRAMS).
_TARGET_GRID_PROGRAMS = 256


@triton.jit
def grouped_launch_diagonal(pid, num_pid_m, num_pid_n, BLOCK_TRESHHOLD: tl.constexpr):
    if (num_pid_m >= BLOCK_TRESHHOLD) and (num_pid_n >= BLOCK_TRESHHOLD):
        curThresholdM = (
            BLOCK_TRESHHOLD
            if pid < (num_pid_m // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * num_pid_n
            else num_pid_m % BLOCK_TRESHHOLD
        )
        curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
        curThresholdN = (
            BLOCK_TRESHHOLD
            if pid % (num_pid_n * BLOCK_TRESHHOLD)
            < (curThresholdM * num_pid_n) // curThresholdM_thresholdN * curThresholdM_thresholdN
            else num_pid_n % BLOCK_TRESHHOLD
        )
        localRelativeBlock = pid % (BLOCK_TRESHHOLD * num_pid_n) % (BLOCK_TRESHHOLD * curThresholdM)
        task_m_idx = localRelativeBlock % curThresholdM + pid // (BLOCK_TRESHHOLD * num_pid_n) * BLOCK_TRESHHOLD
        x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
        while y != 0:
            x, y = y, x % y
        lcm = curThresholdM * curThresholdN // x
        task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + pid % (
            BLOCK_TRESHHOLD * num_pid_n
        ) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
    else:
        task_m_idx = pid // num_pid_n
        task_n_idx = pid % num_pid_n
    return task_m_idx, task_n_idx


def m_grouped_matmul_autotune_config():
    configs = []
    for BM, BN, nw in [
        (64, 64, 4), (64, 128, 4), (128, 64, 4),
        (128, 128, 8), (128, 256, 8), (256, 128, 8),
    ]:
        for BK in [32, 64, 128]:
            for ns in [2, 3]:
                configs.append(triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                    num_warps=nw, num_stages=ns,
                ))
    return configs


@smart_triton_autotune(configs=m_grouped_matmul_autotune_config(), selected_idx=0, key=["N", "K", "MAX_M"])
@triton.jit
def _m_grouped_matmul_kernel(
    A,
    B,
    C,
    group_offsets_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    MAX_M,
    stride_bg,
    strideBK,
    strideBN,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_tile_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    group_id = tl.program_id(2)

    if m_tile_id * BLOCK_M >= MAX_M:
        return

    group_start = tl.load(group_offsets_ptr + group_id).to(tl.int32)
    group_end = tl.load(group_offsets_ptr + group_id + 1).to(tl.int32)
    m_g = group_end - group_start

    if m_tile_id * BLOCK_M >= m_g:
        return

    offs_m = group_start + m_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_base = B + group_id * stride_bg
    if TRANS_B:
        b_ptrs = b_base + offs_n[:, None] * strideBN + offs_k[None, :] * strideBK
    else:
        b_ptrs = b_base + offs_k[:, None] * strideBK + offs_n[None, :] * strideBN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k0 * BLOCK_K)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < group_end) & k_mask[None, :], other=0.0)
        if TRANS_B:
            b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
            b = tl.trans(b)
        else:
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * strideBK

    c = acc.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < group_end) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def m_grouped_matmul_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    size_per_group: torch.Tensor,
    num_groups: int,
    M: int,
    N: int,
    K: int,
    strideBN: int,
    strideBK: int,
    trans_b: bool = False,
) -> torch.Tensor:
    cum = size_per_group.cumsum(0, dtype=torch.int32)
    group_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device=A.device)
    group_offsets[1:] = cum
    max_m = size_per_group.max().item()

    def grid(META):
        return (
            triton.cdiv(N, META["BLOCK_N"]),
            triton.cdiv(max_m, META["BLOCK_M"]),
            num_groups,
        )

    _m_grouped_matmul_kernel[grid](
        A,
        B,
        C,
        group_offsets,
        N,
        K,
        max_m,
        B.stride(0),
        strideBK,
        strideBN,
        trans_b,
    )
    return C


def k_grouped_matmul_autotune_config():
    configs = []
    for BM, BN, nw in [
        (64, 64, 4), (64, 128, 4), (128, 64, 4),
        (128, 128, 8), (128, 256, 8), (256, 128, 8),
    ]:
        for BK in [32, 64, 128]:
            for ns in [2, 3]:
                configs.append(triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                    num_warps=nw, num_stages=ns,
                ))
    return configs


@smart_triton_autotune(configs=k_grouped_matmul_autotune_config(), selected_idx=0, key=["M", "N"])
@triton.jit
def _k_grouped_matmul_kernel(
    A,
    B,
    C,
    group_offsets_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n_tile_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    group_id = tl.program_id(2)

    group_start = tl.load(group_offsets_ptr + group_id).to(tl.int32)
    group_end = tl.load(group_offsets_ptr + group_id + 1).to(tl.int32)
    k_g = group_end - group_start

    offs_m = m_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    msk_m = offs_m < M
    msk_n = offs_n < N

    # A: [total_K, M] row-major, load [BLOCK_K, BLOCK_M] then transpose
    a_ptrs = A + (group_start + offs_k)[:, None] * M + offs_m[None, :]
    # B: [total_K, N] row-major, load [BLOCK_K, BLOCK_N]
    b_ptrs = B + (group_start + offs_k)[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iters = (k_g + BLOCK_K - 1) // BLOCK_K

    for k0 in tl.range(0, num_k_iters):
        k_mask = offs_k < (k_g - k0 * BLOCK_K)
        a = tl.load(a_ptrs, mask=k_mask[:, None] & msk_m[None, :], other=0.0)
        a = tl.trans(a)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & msk_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * M
        b_ptrs += BLOCK_K * N

    c = acc.to(C.dtype.element_ty)
    offs_cm = group_id * M + offs_m
    c_ptrs = C + offs_cm[:, None] * N + offs_n[None, :]
    c_mask = msk_m[:, None] & msk_n[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def k_grouped_matmul_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    size_per_group: torch.Tensor,
    num_groups: int,
    M: int,
    N: int,
) -> torch.Tensor:
    cum = size_per_group.cumsum(0, dtype=torch.int32)
    group_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device=A.device)
    group_offsets[1:] = cum

    def grid(META):
        return (
            triton.cdiv(N, META["BLOCK_N"]),
            triton.cdiv(M, META["BLOCK_M"]),
            num_groups,
        )

    _k_grouped_matmul_kernel[grid](A, B, C, group_offsets, M, N)
    return C
