import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 64},
        ),
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["TOTAL_S", "STRIDE_D_N"])
def kernel_da_bwd_d(
    fp32o,
    do,
    d,
    TOTAL_S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N,
    BLOCK_R: tl.constexpr,
):
    tl.static_assert(STRIDE_O_H == 1)
    tl.static_assert(STRIDE_D_S == 1)
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(TOTAL_S, BLOCK_R)
    for task_id in range(pid, num_r * N, tl.num_programs(axis=0)):
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        offs_h = tl.arange(0, H)
        ptr_fp32o = (
            fp32o
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + offs_h[None, :] * STRIDE_O_H
        )
        ptr_do = (
            do
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + offs_h[None, :] * STRIDE_O_H
        )
        ptr_d = d + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        mask_o = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < TOTAL_S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < TOTAL_S

        block_o = tl.load(ptr_fp32o, mask=mask_o, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_o, other=0.0)
        block_d = tl.sum(block_do.to(tl.float32) * block_o, axis=1)
        tl.store(ptr_d, block_d, mask=mask_d)
