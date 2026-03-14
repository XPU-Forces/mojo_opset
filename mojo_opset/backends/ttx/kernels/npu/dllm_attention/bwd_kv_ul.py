import triton
import triton.language as tl

from .micro_kernel import micro_kernel_bwd_kv


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_R": 256, "BLOCK_C": 64},
        )
    ],
    key=["N", "H"],
)
@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def kernel_da_bwd_kv_ul(
    q,
    k,
    v,
    do,
    d,
    lse,
    dk,
    dv,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ul,
    GROUP_SIZE: tl.constexpr,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_c_st = 0
    NUM_GROUP = N // GROUP_SIZE

    offset_r_local = tl.arange(0, BLOCK_C)[:, None]
    offset_c_local = tl.arange(0, BLOCK_C)[None, :]
    block_mask_ul = tl.load(mask_ul + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_c_ed = offset_block_c_st + tl.cdiv(seq_ed - seq_st, BLOCK_C)
        for task_id in range(
            offset_block_c_st * NUM_GROUP + ((pid % pnum - offset_block_c_st * NUM_GROUP % pnum + pnum) % pnum),
            offset_block_c_ed * NUM_GROUP,
            pnum,
        ):
            idx_c = task_id // NUM_GROUP - offset_block_c_st
            idx_group = task_id % NUM_GROUP
            offs_h = tl.arange(0, H)

            ptr_k = (
                k
                + idx_group * STRIDE_K_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_group * STRIDE_V_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            ptr_dk = (
                dk
                + idx_group * STRIDE_K_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + offs_h[None, :] * STRIDE_K_H
            )
            ptr_dv = (
                dv
                + idx_group * STRIDE_V_N
                + (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + offs_h[None, :] * STRIDE_V_H
            )
            mask_kv = (seq_st + idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < seq_ed

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_dk = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)
            block_dv = tl.full([BLOCK_C, H], 0.0, dtype=tl.float32)

            block_k = tl.trans(block_k)
            block_v = tl.trans(block_v)

            for idx_ingroup in range(GROUP_SIZE):
                idx_n = idx_group * GROUP_SIZE + idx_ingroup

                boundary_mask = (
                    (seq_st + idx_c * BLOCK_C + offset_r_local < seq_ed)
                    & (seq_st + idx_c * BLOCK_C + offset_c_local < seq_ed)
                )

                block_dk, block_dv = micro_kernel_bwd_kv(
                    q,
                    block_k,
                    block_v,
                    do,
                    d,
                    block_dk,
                    block_dv,
                    lse,
                    scale,
                    seq_st + idx_c * BLOCK_C,
                    seq_ed,
                    block_mask_ul,
                    idx_n,
                    offs_h,
                    STRIDE_Q_S,
                    STRIDE_Q_N,
                    STRIDE_Q_H,
                    STRIDE_D_S,
                    STRIDE_D_N,
                    BLOCK_C,
                    boundary_mask=boundary_mask,
                )

            tl.store(ptr_dk, block_dk.to(tl.bfloat16), mask=mask_kv)
            tl.store(ptr_dv, block_dv.to(tl.bfloat16), mask=mask_kv)
        seq_st = seq_ed
        offset_block_c_st = offset_block_c_ed
