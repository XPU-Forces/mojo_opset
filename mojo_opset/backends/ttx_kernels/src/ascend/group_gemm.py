import torch
import triton
import triton.language as tl
import random
import pytest


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


def get_autotune_config():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 9}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 9}),
    ]


@triton.autotune(configs=get_autotune_config(), key=["N", "K"])
@triton.jit
def m_grouped_matmul_bKmajor_kernel(
    A,
    B,
    C,
    group_size_ptr,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    total_cores = tl.num_programs(axis=0)
    core_idx = tl.program_id(axis=0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    last_count = 0
    group_start = 0
    group_end = 0

    for group_idx in range(num_groups):
        m = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + m
        num_block_m = tl.cdiv(m, BLOCK_M)
        cur_count = last_count + num_block_m * num_block_n
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        for cur_block in range(cur_block, cur_count, total_cores):
            task_m_idx, task_n_idx = grouped_launch_diagonal(
                cur_block - last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD
            )
            offs_am = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_bn = (group_idx * N + task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs_base = A + (offs_am[:, None] * K + offs_k[None, :])
            b_ptrs_base = B + (offs_bn[:, None] * strideBN + offs_k[None, :])
            msk_m = offs_am < group_end
            msk_n = offs_bn < (group_idx + 1) * N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                a_ptrs = a_ptrs_base + k * BLOCK_K
                b_ptrs = b_ptrs_base + k * BLOCK_K
                a = tl.load(
                    a_ptrs,
                    mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0,
                )
                tl.compile_hint(a, "dot_pad_only_k")
                b = tl.load(
                    b_ptrs,
                    mask=msk_n[:, None] and (offs_k[None, :] < (K - k * BLOCK_K)),
                    other=0.0,
                )
                b = tl.trans(b)
                tl.compile_hint(b, "dot_pad_only_k")
                accumulator = tl.dot(a, b, acc=accumulator)

            c = accumulator.to(C.dtype.element_ty)
            offs_cm = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_cn = (task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < group_end) and (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
        last_count = cur_count % total_cores
        group_start = group_end


@triton.autotune(configs=get_autotune_config(), key=["N", "K"])
@triton.jit
def m_grouped_matmul_bNmajor_kernel(
    A,
    B,
    C,
    group_size_ptr,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    total_cores = tl.num_programs(axis=0)
    core_idx = tl.program_id(axis=0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    last_count = 0
    group_start = 0
    group_end = 0
    group_idx = 0
    # group_size_m = tl.load(group_size_ptr + tl.arange(0, num_groups)).to(tl.int32)
    # should use tl.static_range on NV
    for group_idx in range(num_groups):
        # m = tl.extract_slice(group_size_m, [group_idx], [1], [1])
        m = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + m
        num_block_m = tl.cdiv(m, BLOCK_M)
        cur_count = last_count + num_block_m * num_block_n
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        for cur_block in range(cur_block, cur_count, total_cores):
            task_m_idx, task_n_idx = grouped_launch_diagonal(
                cur_block - last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD
            )
            offs_am = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_bn = (task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            offs_ak = tl.arange(0, BLOCK_K)
            offs_bk = (group_idx * K) + tl.arange(0, BLOCK_K)
            a_ptrs_base = A + (offs_am[:, None] * K + offs_ak[None, :])
            b_ptrs_base = B + (offs_bk[:, None] * strideBK + offs_bn[None, :])
            msk_m = offs_am < group_end
            msk_n = offs_bn < N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                a_ptrs = a_ptrs_base + k * BLOCK_K
                b_ptrs = b_ptrs_base + k * BLOCK_K * strideBK
                a = tl.load(
                    a_ptrs,
                    mask=msk_m[:, None] and (offs_ak[None, :] < K - k * BLOCK_K),
                    other=0.0,
                )
                tl.compile_hint(a, "dot_pad_only_k")
                b = tl.load(
                    b_ptrs,
                    mask=(offs_bk[:, None] < (group_idx * K + K - k * BLOCK_K)) and msk_n[None, :],
                    other=0.0,
                )
                tl.compile_hint(b, "dot_pad_only_k")
                accumulator = tl.dot(a, b, acc=accumulator)

            c = accumulator.to(C.dtype.element_ty)
            offs_cm = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_cn = (task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < group_end) and (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
        last_count = cur_count % total_cores
        group_start = group_end


def m_grouped_matmul(
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
    num_cores = 48
    m_grouped_matmul_kernel = m_grouped_matmul_bKmajor_kernel if trans_b else m_grouped_matmul_bNmajor_kernel
    m_grouped_matmul_kernel[(num_cores,)](
        A,
        B,
        C,
        size_per_group,
        num_groups,
        M,
        N,
        K,
        strideBN,
        strideBK,
        multibuffer=True,
    )
    return C


def get_autotune_config_1():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
    ]


@triton.autotune(configs=get_autotune_config_1(), key=["M", "N"])
@triton.jit
def k_grouped_matmul_kernel(
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
                tl.compile_hint(aa, "dot_pad_only_k")
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < group_end - kk * BLOCK_K) and msk_n[None, :], other=0.0)
                tl.compile_hint(b, "dot_pad_only_k")
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


def k_grouped_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    size_per_group: torch.Tensor,
    num_groups: int,
    M: int,
    N: int,
) -> torch.Tensor:
    num_cores = 48

    def grid(META):
        assert M % META["BLOCK_M"] == 0, "Only support when M is a multiple of BLOCK_M"
        return (num_cores,)

    k_grouped_matmul_kernel[grid](A, B, C, size_per_group, num_groups, M, N, multibuffer=True)
    return C


from torch import Tensor


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tokens_per_expert):
        out = m_grouped_matmul(x, w, tokens_per_expert, trans_b=True)
        ctx.save_for_backward(x, w, tokens_per_expert)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w, tokens_per_expert = ctx.saved_tensors
        dx = m_grouped_matmul(grad_output, w, tokens_per_expert, trans_b=False)
        dw = k_grouped_matmul(grad_output, x, tokens_per_expert)
        return dx, dw, None


def ttx_grouped_gemm(x, w, tokens_per_expert):
    if x.shape[0] == 0:
        return torch.matmul(x, w[0].T)
    return GroupedGemm.apply(x, w, tokens_per_expert)


def ttx_k_grouped_gemm(A: Tensor, B: Tensor, size_per_group: torch.Tensor) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 2
    AK, M = A.shape
    BK, N = B.shape
    assert A.stride(-1) == 1, "Please make sure A is K-major"
    assert B.stride(-1) == 1, "Please make sure B is K-major"
    assert AK == BK, "Please make sure that A and B have the same seqlen"
    num_groups = size_per_group.shape[0]
    C = A.new_empty(num_groups, M, N)
    k_grouped_matmul(A, B, C, size_per_group, num_groups, M, N)
    return C


def ttx_m_grouped_gemm(A: Tensor, B: Tensor, size_per_group: torch.Tensor, trans_b: bool = False) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape
    assert A.stride(-1) == 1, "Please make sure A is K-major"
    if trans_b:
        num_groups, N, BK = B.shape
        strideBN, strideBK = B.stride(1), B.stride(2)
    else:
        num_groups, BK, N = B.shape
        strideBK, strideBN = B.stride(1), B.stride(2)
    assert BK == K, "K of A should be equal to K of B"
    assert num_groups == size_per_group.numel()
    C = A.new_empty(M, N)
    m_grouped_matmul(A, B, C, size_per_group, num_groups, M, N, K, strideBN, strideBK, trans_b)
    return C