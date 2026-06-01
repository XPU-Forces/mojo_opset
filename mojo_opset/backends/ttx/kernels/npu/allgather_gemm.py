import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend.algorithm import (
    dist_swizzle2d_Nz,
    gemm_swizzle2d_Nz,
)
from triton.language.extra.cann.extension import sub_vec_id

from .utils import get_num_cores


@triton.jit
def kernel_allgather_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    peer_mem_ptr,
    rank,
    rank_size,
    buffer_num,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    pvalue: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    COMM_BLOCK_SIZE_M: tl.constexpr,
    COMM_BLOCK_SIZE_K: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    dtype = tl.bfloat16 if IS_BF16 else tl.float16
    subblock_idx = sub_vec_id()
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    num_loops_m = tl.cdiv(M, BLOCK_SIZE_M * pvalue)
    num_loops_n = tl.cdiv(N, BLOCK_SIZE_N)
    buffer_row_size = BLOCK_SIZE_M * pvalue * rank_size
    for global_id_m in range(0, num_loops_m):
        buffer_id = global_id_m % buffer_num
        actual_block_size_m = BLOCK_SIZE_M * pvalue
        if global_id_m == num_loops_m - 1:
            actual_block_size_m = M - global_id_m * BLOCK_SIZE_M * pvalue
        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        comm_num_m_blocks = tl.cdiv(actual_block_size_m, COMM_BLOCK_SIZE_M)
        comm_num_k_blocks = tl.cdiv(K, COMM_BLOCK_SIZE_K)
        if subblock_idx == 0:
            for k in range(
                pid, comm_num_m_blocks * comm_num_k_blocks * rank_size, ncore
            ):
                block_id_m, block_id_k, target_rank, comm_row_shape, comm_col_shape = (
                    dist_swizzle2d_Nz(
                        k,
                        rank_size,
                        actual_block_size_m,
                        K,
                        COMM_BLOCK_SIZE_M,
                        COMM_BLOCK_SIZE_K,
                    )
                )
                remote_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                comm_offs_m = (
                    tl.arange(0, COMM_BLOCK_SIZE_M)
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + global_id_m * BLOCK_SIZE_M * pvalue
                )
                comm_offs_k = (
                    tl.arange(0, COMM_BLOCK_SIZE_K) + block_id_k * COMM_BLOCK_SIZE_K
                )
                a_ptrs = a_ptr + (
                    comm_offs_m[:, None] * stride_am + comm_offs_k[None, :] * stride_ak
                )
                peermem_comm_offs_m = (
                    buffer_id * buffer_row_size
                    + rank * BLOCK_SIZE_M * pvalue
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + tl.arange(0, COMM_BLOCK_SIZE_M)
                )
                remote_ptrs = remote_ptr + (
                    peermem_comm_offs_m[:, None] * stride_am
                    + comm_offs_k[None, :] * stride_ak
                )
                comm_msk_m = comm_offs_m[:, None] < M
                peermem_comm_msk_m = (
                    peermem_comm_offs_m[:, None]
                    < buffer_id * buffer_row_size
                    + BLOCK_SIZE_M * rank * pvalue
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + comm_row_shape
                )
                a = tl.load(
                    a_ptrs, mask=(comm_offs_k[None, :] < K) & comm_msk_m, other=0.0
                )
                tl.store(
                    remote_ptrs, a, mask=(comm_offs_k[None, :] < K) & peermem_comm_msk_m
                )
        libshmem_device.barrier_all()
        num_tiles_m = tl.cdiv(actual_block_size_m, BLOCK_SIZE_M)
        for block_id in range(pid, num_tiles_m * num_loops_n * rank_size, ncore):
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                block_id,
                rank_size * BLOCK_SIZE_M * num_tiles_m,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            rank_idx = block_id_m // num_tiles_m
            block_id_m = rank_idx * pvalue + (block_id_m % num_tiles_m)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            matmul_offs_am = (
                buffer_id * buffer_row_size
                + block_id_m * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            matmul_msk_am = matmul_offs_am[:, None] < (
                buffer_id * buffer_row_size
                + BLOCK_SIZE_M * rank_idx * pvalue
                + actual_block_size_m
            )
            offs_bn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            msk_n = offs_bn[None, :] < N
            for block_id_k in range(0, num_k_blocks):
                offs_k = tl.arange(0, BLOCK_SIZE_K) + block_id_k * BLOCK_SIZE_K
                a_ptrs = peer_mem_ptr + (
                    matmul_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
                )
                b_ptrs = b_ptr + (
                    offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
                )
                a = tl.load(
                    a_ptrs, mask=(offs_k[None, :] < K) & matmul_msk_am, other=0.0
                )
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & msk_n, other=0.0)
                accumulator += tl.dot(a, b)
            c = accumulator.to(dtype)
            offs_cm = (
                block_id_m // pvalue * M
                + global_id_m * BLOCK_SIZE_M * pvalue
                + (block_id_m % pvalue) * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            offs_cn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M * (block_id_m // pvalue + 1)) & (
                offs_cn[None, :] < N
            )
            tl.store(c_ptrs, c, mask=c_mask)


def allgather_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    peer_mem: torch.Tensor,
    rank: int,
    world_size: int,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 256,
    BLOCK_SIZE_K: int = 256,
    COMM_BLOCK_SIZE_M: int = 20,
    COMM_BLOCK_SIZE_K: int = 256,
    pvalue: int = 4,
    buffer_num: int = 2,
) -> torch.Tensor:
    M, K = input.shape
    _, N = weight.shape
    ncore = get_num_cores("cube")
    kernel_allgather_gemm[ncore, 1, 1](
        input,
        weight,
        output,
        peer_mem,
        rank,
        world_size,
        buffer_num,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        pvalue,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        COMM_BLOCK_SIZE_M,
        COMM_BLOCK_SIZE_K,
        input.dtype == torch.bfloat16,
    )
    return output


def allgather_gemm_peer_mem_size(K: int, world_size: int) -> int:
    """Return the flat peer_mem element count required for allgather_gemm kernel."""
    BLOCK_SIZE_M, pvalue, buffer_num, BLOCK_SIZE_K = 128, 4, 2, 256
    return BLOCK_SIZE_M * pvalue * world_size * buffer_num * max(K, BLOCK_SIZE_K)
