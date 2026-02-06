import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver
from torch import Tensor

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


def matmul_impl(x: Tensor, w: Tensor, bias: Tensor = None) -> Tensor:

    input_shape = x.shape  # * [B, M, K]
    x = x.reshape(-1, w.shape[-1])  # * [BM, K]

    BM, K = x.shape
    N = w.shape[0]
    b_t = w.T.contiguous()

    mat_c = torch.empty(BM, N, dtype=x.dtype, device=x.device)  # * [BM, N]

    num_cores = get_num_cores("cube")
    # * 避免频繁调优，设置最小的基块
    base_block_size = 64
    TRESHHOLD_M = BM // base_block_size
    TRESHHOLD_N = N // base_block_size

    matmul_kernel[(num_cores,)](
        x,
        b_t,
        mat_c,
        BM,
        N,
        K,
        TRESHHOLD_M,
        TRESHHOLD_N,
        num_cores,
    )
    # TODO: fuse add
    if bias is not None:
        c += bias.unsqueeze(0)
    new_shape = input_shape[:-1] + (b_t.shape[-1],)

    return mat_c.reshape(new_shape)  # * [B, M, N]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "BLOCK_TRESHHOLD": 4}),  # * added
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_TRESHHOLD": 4}),  # * added
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
    ],
    key=["TRESHHOLD_M", "TRESHHOLD_N", "K"],  # * K 轴不会频繁变化，但是 M、N 轴的长度会频繁变化，不作为调优传参
)
@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N,
    K: tl.constexpr,
    TRESHHOLD_M: tl.constexpr,  # * autotune args
    TRESHHOLD_N: tl.constexpr,  # * autotune args
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    task_m_idx = 0
    task_n_idx = 0

    """
    Refer to https://gitcode.com/Ascend/triton-ascend/blob/master/ascend/examples/tutorials/13-matrix-multiplication-optimized.py
    """
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    # * 当任务量较多时，可以使能对角线分核策略进行优化
    if NUM_BLOCKS_M >= BLOCK_TRESHHOLD or NUM_BLOCKS_N >= BLOCK_TRESHHOLD:
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            # * 8 * 8 对角线分核代码实现
            curThresholdM = BLOCK_TRESHHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_TRESHHOLD
            curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
            curThresholdN = (
                BLOCK_TRESHHOLD
                if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN
                else NUM_BLOCKS_N % BLOCK_TRESHHOLD
            )
            localRelativeBlock = block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) % (BLOCK_TRESHHOLD * curThresholdM)
            task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
            # * 求最小公倍数，方便求基本块的坐标
            x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
            while y != 0:
                x, y = y, x % y
            lcm = curThresholdM * curThresholdN // x
            task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_TRESHHOLD

            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N

            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                tl.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                tl.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
    # if NUM_BLOCKS_M >= BLOCK_TRESHHOLD:
    #     pass
    else:
        # * 传统顺序分核
        pass
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N

            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                tl.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                tl.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
