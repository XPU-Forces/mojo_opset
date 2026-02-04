import torch
import triton
import triton.language as tl
from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


def lightning_indexer_impl(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
):
    B, M, H, K = query.shape
    N = key.shape[1]

    output = torch.zeros((B, M, N), dtype=torch.float32, device=query.device)
    num_cores = get_num_cores("cube")

    grid = (num_cores,)
    lightning_indexer_kernel[grid](
        query,
        key,
        query_scale,
        key_scale,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        query_scale.stride(0),
        query_scale.stride(1),
        query_scale.stride(2),
        key_scale.stride(0),
        key_scale.stride(1),
        key_scale.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        B,
        M,
        N,
        H,
        K,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 64}),
        triton.Config({"BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_N": 256}),
        triton.Config({"BLOCK_SIZE_N": 512}),
    ],
    key=["N", "H", "K"],
)
@libentry()
@triton.jit
def lightning_indexer_kernel(
    query_ptr,
    key_ptr,
    query_scale_ptr,
    key_scale_ptr,
    output_ptr,
    query_stride_b,
    query_stride_m,
    query_stride_h,
    query_stride_k,
    key_stride_b,
    key_stride_n,
    key_stride_k,
    query_scale_stride_b,
    query_scale_stride_m,
    query_scale_stride_h,
    key_scale_stride_b,
    key_scale_stride_n,
    key_scale_stride_k,
    output_stride_b,
    output_stride_m,
    output_stride_n,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Caculate lightning index score.

    Args:
        query_ptr (tl.tensor): Pointer to the Q.
        key_ptr (tl.tensor): Pointer to the K.
        output_ptr (tl.tensor): Pointer to the index score.
        query_scale_ptr (tl.tensor): Pointer to scaling factors for Q (float), or weights.
        B (tl.constexpr): Batch size.
        H (tl.constexpr): Number of Q heads.
        M (tl.constexpr): Q sequence length.
        N (tl.constexpr): K sequence length.
        K (tl.constexpr): dim length.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.

    Returns:
        None
    """

    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = B * M * NUM_BLOCKS_N

    # Current M-dimension block index
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        batch_idx = (block_idx // (M * NUM_BLOCKS_N)).to(tl.int64)
        m_idx = ((block_idx // NUM_BLOCKS_N) % M).to(tl.int64)
        n_idx = (block_idx % NUM_BLOCKS_N).to(tl.int64)

        offs_h = tl.arange(0, H)
        offs_k = tl.arange(0, K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        key_ptrs = (
            key_ptr
            + batch_idx * key_stride_b
            + n_idx * BLOCK_SIZE_N * key_stride_n
            + offs_n[:, None] * key_stride_n
            + offs_k[None, :] * key_stride_k
        )
        mask = n_idx * BLOCK_SIZE_N + offs_n < N
        k = tl.load(key_ptrs, mask=mask[:, None], other=0.0)

        key_scale_ptrs = (
            key_scale_ptr
            + batch_idx * key_scale_stride_b
            + n_idx * BLOCK_SIZE_N * key_scale_stride_n
            + offs_n[:, None] * key_scale_stride_n
            + offs_k[None, :] * key_scale_stride_k
        )
        k_scale = tl.load(key_scale_ptrs, mask=mask[:, None], other=0.0)

        k = k * k_scale

        query_ptrs = (
            query_ptr
            + batch_idx * query_stride_b
            + m_idx * query_stride_m
            + offs_h[:, None] * query_stride_h
            + offs_k[None, :] * query_stride_k
        )
        q = tl.load(query_ptrs)

        relu_qk = tl.maximum(tl.dot(q.to(k.dtype), tl.trans(k)), 0.0)

        query_scale_ptrs = (
            query_scale_ptr
            + batch_idx * query_scale_stride_b
            + m_idx * query_scale_stride_m
            + offs_h * query_scale_stride_h
        )
        q_scale = tl.load(query_scale_ptrs)

        o = tl.sum(relu_qk * q_scale[:, None], axis=0)

        output_ptrs = (
            output_ptr
            + batch_idx * output_stride_b
            + m_idx * output_stride_m
            + n_idx * BLOCK_SIZE_N
            + offs_n * output_stride_n
        )
        tl.store(output_ptrs, o, mask=mask)
