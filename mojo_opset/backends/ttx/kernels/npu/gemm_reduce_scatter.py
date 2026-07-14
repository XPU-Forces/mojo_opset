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
def kernel_gemm_reduce_scatter(
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
    COMM_BLOCK_SIZE_N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    dtype = tl.bfloat16 if IS_BF16 else tl.float16
    subblock_idx = sub_vec_id()
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    loop_num_per_comm = ncore * pvalue
    m_per_rank = M // rank_size
    num_loops_m = tl.cdiv(m_per_rank, BLOCK_SIZE_M) * rank_size
    num_loops_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_loops = num_loops_m * num_loops_n
    num_loops_comm = tl.cdiv(total_loops, loop_num_per_comm)
    buffer_row_size = BLOCK_SIZE_M * loop_num_per_comm
    for global_id in range(0, num_loops_comm):
        buffer_id = global_id % buffer_num
        actual_loop_num_per_comm = loop_num_per_comm
        output_block_offset_in_rank = global_id * loop_num_per_comm // rank_size
        if global_id == num_loops_comm - 1:
            actual_loop_num_per_comm = total_loops - global_id * loop_num_per_comm
        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        num_blocks_per_rank = actual_loop_num_per_comm // rank_size
        for block_id in range(pid, actual_loop_num_per_comm, ncore):
            block_id_in_rank = (
                output_block_offset_in_rank + block_id % num_blocks_per_rank
            )
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                block_id_in_rank,
                m_per_rank,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            rank_idx = block_id // num_blocks_per_rank
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            matmul_offs_am = (
                m_per_rank * rank_idx
                + block_id_m * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            matmul_msk_am = matmul_offs_am[:, None] < (m_per_rank * (rank_idx + 1))
            offs_bn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            msk_n = offs_bn[None, :] < N
            for block_id_k in range(0, num_k_blocks):
                offs_k = tl.arange(0, BLOCK_SIZE_K) + block_id_k * BLOCK_SIZE_K
                a_ptrs = a_ptr + (
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
            offs_peer_mem_m = (
                buffer_id * buffer_row_size
                + block_id * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            offs_peer_mem_n = tl.arange(0, BLOCK_SIZE_N)
            peer_mem_ptrs = (
                peer_mem_ptr
                + BLOCK_SIZE_N * offs_peer_mem_m[:, None]
                + offs_peer_mem_n[None, :]
            )
            tl.store(peer_mem_ptrs, c)
        libshmem_device.barrier_all()
        comm_problem_size_m_per_rank = tl.cdiv(
            actual_loop_num_per_comm * BLOCK_SIZE_M, rank_size
        )
        comm_block_num_per_rank = tl.cdiv(
            comm_problem_size_m_per_rank, COMM_BLOCK_SIZE_M
        ) * tl.cdiv(BLOCK_SIZE_N, COMM_BLOCK_SIZE_N)
        if subblock_idx == 0:
            for idx in range(pid, comm_block_num_per_rank * rank_size, ncore):
                block_id_m, block_id_n, target_rank, comm_row_shape, comm_col_shape = (
                    dist_swizzle2d_Nz(
                        idx,
                        rank_size,
                        comm_problem_size_m_per_rank,
                        BLOCK_SIZE_N,
                        COMM_BLOCK_SIZE_M,
                        COMM_BLOCK_SIZE_N,
                    )
                )
                remote_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                comm_offs_m = (
                    tl.arange(0, COMM_BLOCK_SIZE_M)
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + rank * comm_problem_size_m_per_rank
                    + buffer_id * loop_num_per_comm * BLOCK_SIZE_M
                )
                comm_offs_n = (
                    tl.arange(0, COMM_BLOCK_SIZE_N) + block_id_n * COMM_BLOCK_SIZE_N
                )
                remote_ptrs = remote_ptr + (
                    comm_offs_m[:, None] * BLOCK_SIZE_N + comm_offs_n[None, :]
                )
                block_id = block_id_m * COMM_BLOCK_SIZE_M // BLOCK_SIZE_M
                block_id_in_rank = output_block_offset_in_rank + block_id
                block_id_gemm_m, block_id_gemm_n = gemm_swizzle2d_Nz(
                    block_id_in_rank, m_per_rank, N, BLOCK_SIZE_M, BLOCK_SIZE_N
                )
                m_offset_in_block = (block_id_m * COMM_BLOCK_SIZE_M) % BLOCK_SIZE_M
                n_offset_in_block = (block_id_n * COMM_BLOCK_SIZE_N) % BLOCK_SIZE_N
                offs_cm = (
                    block_id_gemm_m * BLOCK_SIZE_M
                    + m_offset_in_block
                    + tl.arange(0, COMM_BLOCK_SIZE_M)
                )
                offs_cn = (
                    block_id_gemm_n * BLOCK_SIZE_N
                    + n_offset_in_block
                    + tl.arange(0, COMM_BLOCK_SIZE_N)
                )
                c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < m_per_rank) & (offs_cn[None, :] < N)
                c_temp = tl.load(remote_ptrs)
                tl.atomic_add(c_ptr + c_offs, c_temp, mask=c_mask)


def gemm_reduce_scatter_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    peer_mem: torch.Tensor,
    rank: int,
    world_size: int,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 256,
    BLOCK_SIZE_K: int = 256,
    COMM_BLOCK_SIZE_M: int = 8,
    COMM_BLOCK_SIZE_N: int = 256,
    pvalue: int = 4,
    buffer_num: int = 2,
) -> None:
    M, K = input.shape
    _, N = weight.shape
    ncore = get_num_cores("cube")
    kernel_gemm_reduce_scatter[ncore, 1, 1](
        input, weight, output, peer_mem,
        rank, world_size, buffer_num,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        pvalue, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        COMM_BLOCK_SIZE_M, COMM_BLOCK_SIZE_N, input.dtype == torch.bfloat16,
    )


def gemm_reduce_scatter_peer_mem_size() -> int:
    """Return the flat peer_mem element count required for gemm_reduce_scatter kernel."""
    BLOCK_SIZE_M, BLOCK_SIZE_N = 128, 256
    ncore = get_num_cores("cube")
    pvalue, buffer_num = 4, 2
    return BLOCK_SIZE_M * pvalue * ncore * buffer_num * BLOCK_SIZE_N
