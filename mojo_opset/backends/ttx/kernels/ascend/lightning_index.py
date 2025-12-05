import torch
import triton
import triton.language as tl
from typing import Optional
from triton.runtime.libentry import libentry
        
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_N": 256}),
        triton.Config({"BLOCK_SIZE_N": 512}),
    ],
    key=["H", "N", "K"],
)
@libentry()
@triton.jit
def lightning_index_kernel(
    q_ptr,
    k_ptr,
    o_ptr,
    q_s_ptr,
    B,
    H: tl.constexpr,
    M,
    N,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        q_ptr (tl.tensor): Pointer to the Q.
        k_ptr (tl.tensor): Pointer to the K.
        o_ptr (tl.tensor): Pointer to the index score.
        q_s_ptr (tl.tensor): Pointer to scaling factors for Q (float), or weights.
        B : Batch size.
        H (tl.constexpr): Number of Q heads.
        M : Q sequence length.
        N : K sequence length.
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
        batch_idx = block_idx // (M * NUM_BLOCKS_N)
        bm_idx = block_idx // NUM_BLOCKS_N
        
        n_idx = block_idx % NUM_BLOCKS_N

        offs_h = tl.arange(0, H)
        offs_k = tl.arange(0, K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        q_ptrs = q_ptr + bm_idx * H * K + offs_h[:, None] * K + offs_k[None, :]
        q_s_ptrs = q_s_ptr + bm_idx * H + offs_h
        q = tl.load(q_ptrs)
        q_s = tl.load(q_s_ptrs)

        k_ptrs = (
            k_ptr
            + batch_idx * N * K
            + n_idx * BLOCK_SIZE_N * K
            + offs_n[:, None] * K
            + offs_k[None, :]
        )

        mask = n_idx * BLOCK_SIZE_N + offs_n < N
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)

        o = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * q_s[:, None], axis=0)

        o_ptrs = o_ptr + bm_idx * N + n_idx * BLOCK_SIZE_N + offs_n
        tl.store(o_ptrs, o, mask=mask)

def lightning_index(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
):
    assert query.is_contiguous() and key.is_contiguous(), "Input tensors must be contiguous"

    B, M, H, K = query.size()
    N = key.size(1)

    o = torch.zeros((B, M, N), dtype=torch.float32, device=query.device)
    try:
        import triton.runtime.driver as driver
        NUM_CORE = driver.active.utils.get_device_properties(torch.npu.current_device())[
            "num_vectorcore"
        ]
    except AttributeError:
        NUM_CORE = 48
        
    grid = (NUM_CORE // 2,)
    lightning_index_kernel[grid](query, key, o, query_scale, B, H, M, N, K)
    return o


def lightning_index_forward(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    topk: int,
    key_scale: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    index_score = lightning_index(query, query_scale, key, key_scale)

    if mask is not None:
        index_score += mask
    topk_indices = index_score.topk(topk, dim=-1)[1]
    return topk_indices,index_score
