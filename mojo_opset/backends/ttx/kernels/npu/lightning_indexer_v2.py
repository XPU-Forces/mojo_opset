import torch
import triton
import triton.language as tl
from triton.runtime.libentry import libentry

try:
    import triton.experimental.tle as tle
except ImportError:
    tle = None

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


pipe = tle.dsa.ascend.PIPE if tle is not None else None


def lightning_indexer_v2_impl(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
):
    if tle is None:
        raise NotImplementedError("lightning_indexer_v2 requires triton.experimental.tle")

    B, M, H, K = query.shape
    N = key.shape[1]

    output = torch.zeros((B, M, N), dtype=torch.float32, device=query.device)
    num_cores = get_num_cores("cube")

    max_block_size_n = 512
    wsp = torch.empty((2, num_cores, H, max_block_size_n), dtype=torch.float32, device=query.device)

    grid = (num_cores,)
    lightning_indexer_v2_kernel[grid](
        query,
        key,
        query_scale,
        key_scale,
        output,
        wsp,
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
        output.stride(0),
        output.stride(1),
        output.stride(2),
        wsp.stride(0),
        wsp.stride(1),
        wsp.stride(2),
        B,
        M,
        N,
        H,
        K,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 64, "disable_auto_inject_block_sync": True, "unit_flag": True}),
        triton.Config({"BLOCK_SIZE_N": 128, "disable_auto_inject_block_sync": True, "unit_flag": True}),
        triton.Config({"BLOCK_SIZE_N": 256, "disable_auto_inject_block_sync": True, "unit_flag": True}),
        triton.Config({"BLOCK_SIZE_N": 512, "disable_auto_inject_block_sync": True, "unit_flag": True}),
    ],
    key=["N", "H", "K"],
)
@libentry()
@triton.jit
def lightning_indexer_v2_kernel(
    query_ptr,
    key_ptr,
    query_scale_ptr,
    key_scale_ptr,
    output_ptr,
    wsp_ptr,
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
    output_stride_b,
    output_stride_m,
    output_stride_n,
    wsp_stride_db,
    wsp_stride_core,
    wsp_stride_h,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks = B * M * num_blocks_n
    half_n: tl.constexpr = BLOCK_SIZE_N // 2
    vec_id = tle.dsa.ascend.sub_vec_id()

    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    tle.dsa.ascend.sync_block_set("vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set("vector", "cube", 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)

    buf_id = 0

    for block_idx in range(pid, num_blocks, num_core):
        batch_idx = (block_idx // (M * num_blocks_n)).to(tl.int32)
        m_idx = ((block_idx // num_blocks_n) % M).to(tl.int32)
        n_idx = (block_idx % num_blocks_n).to(tl.int32)

        offs_h = tl.arange(0, H)
        offs_k = tl.arange(0, K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        mask = n_idx * BLOCK_SIZE_N + offs_n < N

        key_ptrs = (
            key_ptr
            + batch_idx * key_stride_b
            + n_idx * BLOCK_SIZE_N * key_stride_n
            + offs_n[:, None] * key_stride_n
            + offs_k[None, :] * key_stride_k
        )
        k = tl.load(key_ptrs, mask=mask[:, None], other=0.0)

        query_ptrs = (
            query_ptr
            + batch_idx * query_stride_b
            + m_idx * query_stride_m
            + offs_h[:, None] * query_stride_h
            + offs_k[None, :] * query_stride_k
        )
        q = tl.load(query_ptrs)

        qk = tl.dot(q.to(k.dtype), tl.trans(k))

        tle.dsa.ascend.sync_block_wait("vector", "cube", buf_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)

        wsp_ptrs = (
            wsp_ptr
            + buf_id * wsp_stride_db
            + pid * wsp_stride_core
            + offs_h[:, None] * wsp_stride_h
            + offs_n[None, :]
        )
        tl.store(wsp_ptrs, qk.to(tl.float32), mask=mask[None, :])

        tle.dsa.ascend.sync_block_set("cube", "vector", buf_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)

        key_scale_ptrs = (
            key_scale_ptr
            + batch_idx * key_scale_stride_b
            + n_idx * BLOCK_SIZE_N * key_scale_stride_n
            + vec_id * half_n
            + tl.arange(0, half_n) * key_scale_stride_n
        )
        k_scale = tl.load(
            key_scale_ptrs,
            mask=n_idx * BLOCK_SIZE_N + vec_id * half_n + tl.arange(0, half_n) < N,
            other=0.0,
        )

        query_scale_ptrs = (
            query_scale_ptr
            + batch_idx * query_scale_stride_b
            + m_idx * query_scale_stride_m
            + offs_h * query_scale_stride_h
        )
        q_scale = tl.load(query_scale_ptrs)

        tle.dsa.ascend.sync_block_wait("cube", "vector", buf_id, pipe.PIPE_FIX, pipe.PIPE_MTE2)

        wsp_vec_ptrs = (
            wsp_ptr
            + buf_id * wsp_stride_db
            + pid * wsp_stride_core
            + offs_h[:, None] * wsp_stride_h
            + (tl.arange(0, half_n) + vec_id * half_n)[None, :]
        )
        qk_vec = tl.load(
            wsp_vec_ptrs,
            mask=(n_idx * BLOCK_SIZE_N + vec_id * half_n + tl.arange(0, half_n) < N)[None, :],
            other=0.0,
        )

        relu_qk = tl.maximum(qk_vec * k_scale[None, :], 0.0)
        o = tl.sum(relu_qk * q_scale[:, None], axis=0)

        output_ptrs = (
            output_ptr
            + batch_idx * output_stride_b
            + m_idx * output_stride_m
            + n_idx * BLOCK_SIZE_N
            + vec_id * half_n
            + tl.arange(0, half_n) * output_stride_n
        )
        tl.store(output_ptrs, o, mask=n_idx * BLOCK_SIZE_N + vec_id * half_n + tl.arange(0, half_n) < N)

        tle.dsa.ascend.sync_block_set("vector", "cube", buf_id, pipe.PIPE_MTE2, pipe.PIPE_FIX)

        buf_id ^= 1

    tle.dsa.ascend.sync_block_wait("vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_wait("vector", "cube", 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)
