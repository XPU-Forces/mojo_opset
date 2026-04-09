# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# ILU Triton grouped matmul (aligned with NPU group_gemm; launch uses ILU vector cores).

import torch
import triton
import triton.language as tl

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


@triton.jit
def _group_matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k0 * BLOCK_K)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
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
    """Per-group matmul via ``_group_matmul_kernel`` (Triton).

    ``trans_b`` matches ``TTXGroupGemm`` (``not trans_weight``): weight storage is ``(G, K, N)``
    when True and ``(G, N, K)`` when False. Each group uses ``B_g`` with shape ``(K, N)``.
    ``strideBN`` / ``strideBK`` match the NPU/TTX custom-op signature but are unused here
    (strides come from ``A_g`` / ``B_g`` slices).
    """
    _ = (strideBN, strideBK, num_groups, M)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    group_start = size_per_group.cumsum(0) - size_per_group
    group_end = size_per_group.cumsum(0)
    for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
        m_g = end - start
        if m_g <= 0:
            continue

        A_g = A.narrow(0, start, m_g)
        if trans_b:
            B_g = B[g]
        else:
            B_g = B[g].transpose(0, 1).contiguous()
        C_g = C.narrow(0, start, m_g)
        grid = (triton.cdiv(m_g, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _group_matmul_kernel[grid](
            A_g,
            B_g,
            C_g,
            m_g,
            N,
            K,
            A_g.stride(0),
            A_g.stride(1),
            B_g.stride(0),
            B_g.stride(1),
            C_g.stride(0),
            C_g.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    return C


@triton.jit
def _k_grouped_matmul_kernel(
    A,
    B,
    C,
    group_size_ptr,
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    total_cores = tl.num_programs(axis=0)
    core_idx = tl.program_id(axis=0)
    last_count = 0
    group_start = 0
    group_end = 0
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    blocks_per_group = num_block_m * num_block_n
    # group_size_k = tl.load(group_size_ptr + tl.arange(0, num_groups)).to(tl.int32)
    for group_idx in range(num_groups):
        # k = tl.extract_slice(group_size_k, [group_idx], [1], [1])
        tokens = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + tokens
        cur_count = last_count + blocks_per_group
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        for cur_block in range(cur_block, cur_count, total_cores):
            task_m_idx, task_n_idx = grouped_launch_diagonal(
                cur_block - last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD
            )
            # matmul begin
            offs_am = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = task_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = group_start + tl.arange(0, BLOCK_K)
            a_ptrs_base = A + offs_k[:, None] * M + offs_am[None, :]
            b_ptrs_base = B + offs_k[:, None] * N + offs_bn[None, :]
            msk_m = offs_am < M
            msk_n = offs_bn < N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in tl.range(0, tl.cdiv(tokens, BLOCK_K)):
                a_ptrs = a_ptrs_base + kk * BLOCK_K * M
                b_ptrs = b_ptrs_base + kk * BLOCK_K * N
                a = tl.load(a_ptrs, mask=(offs_k[:, None] < group_end - kk * BLOCK_K) and msk_m[None, :], other=0.0)
                aa = tl.trans(a)
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < group_end - kk * BLOCK_K) and msk_n[None, :], other=0.0)
                accumulator = tl.dot(aa, b, acc=accumulator)

            c = accumulator.to(C.dtype.element_ty)
            offs_cm = group_idx * M + task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = task_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = C + offs_cm[:, None] * N + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < (group_idx + 1) * M) and (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
            # matmul_end
            # cur_block = cur_block + total_cores
        last_count = cur_count % total_cores
        group_start = group_end


def k_grouped_matmul_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    size_per_group: torch.Tensor,
    num_groups: int,
    M: int,
    N: int,
) -> torch.Tensor:
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    BLOCK_TRESHHOLD = 4

    n_tiles = num_groups * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if n_tiles == 0:
        return C
    BLOCK = max(1, triton.cdiv(n_tiles, _TARGET_GRID_PROGRAMS))
    grid = (triton.cdiv(n_tiles, BLOCK),)

    _k_grouped_matmul_kernel[grid](
        A,
        B,
        C,
        size_per_group,
        num_groups,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
    )
    return C
