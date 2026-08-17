import itertools

from functools import cache
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl


@cache
def get_device_properties() -> Tuple[int, int]:
    device = torch.npu.current_device()
    device_properties: Dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(device)

    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


@triton.jit
def _sdpa_infer_inner(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask_base_ptr,
    start_m,
    qk_scale,  # Starting position of current query block, qk scale factor
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Block size constants
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  # Current stage flag, m and n offset indices
    SEQ: tl.constexpr,
    HAS_MASK: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # Iterate over all k, v blocks in the current stage and accumulate the output
    for start_n in range(0, SEQ, BLOCK_N):  # Process BLOCK_N columns at a time
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
        if HAS_MASK:
            mask_ptr = (
                mask_base_ptr
                + start_m * BLOCK_M * SEQ
                + start_n
                + tl.arange(0, BLOCK_M)[:, None] * SEQ
                + tl.arange(0, BLOCK_N)[None, :]
            )
        # -- Compute qk ----
        k = tl.load(K_block_ptr)
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)

        # NOTE(zhangjihang): tl.where will introduce ub overflow
        qk = qk * qk_scale
        if HAS_MASK:
            mask = tl.load(mask_ptr)
            # qk += (1 - mask.to(tl.float32)) * (-1e6)
            # qk = tl.where(mask, qk, float("-inf"))
            qk = tl.where(mask, qk, -1e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=tl.PropagateNan.ALL), propagate_nan=tl.PropagateNan.ALL)
        #m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
        qk = qk - m_ij[:, None]  # Stabilize

        # Softmax weights p = exp(qk)
        p = tl.math.exp(qk)

        p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr)  # Load corresponding V block
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        acc_ptr = acc_ptr * alpha[:, None]
        acc_ptr = tl.dot(p_cast, v, acc_ptr)
        tl.extra.cann.extension.compile_hint(acc_ptr, "hivm.tile_mix_cube_num",2)

        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


def get_autotune_config():
    configs = []

    BM_list = [64, 128]  # 64, 128, 256
    BN_list = [64, 128]  # 64, 128, 256, 512

    multibuffer_list = [True]  # [True, False]
    unit_flag_list = [True]  # [True, False]
    limit_auto_multi_buffer_only_for_local_buffer_list = [False]  # [True, False]
    limit_auto_multi_buffer_of_local_buffer_list = ["no-l0c"]  # ["no-limit", "no-l0c"]

    # These knobs are tuned only when limit_auto_multi_buffer_only_for_local_buffer=False
    set_workspace_multibuffer_list = [2, 4]  # [2, 4]
    enable_hivm_auto_cv_balance_list = [True]  # [True, False]
    tile_mix_vector_loop_num_list = [2, 4]  # [2, 4]
    tile_mix_cube_loop_num_list = [2, 4]  # [2, 4]

    for (
        BM,
        BN,
        multibuffer,
        unit_flag,
        limit_auto_multi_buffer_only_for_local_buffer,
        limit_auto_multi_buffer_of_local_buffer,
    ) in itertools.product(
        BM_list,
        BN_list,
        multibuffer_list,
        unit_flag_list,
        limit_auto_multi_buffer_only_for_local_buffer_list,
        limit_auto_multi_buffer_of_local_buffer_list,
    ):
        if limit_auto_multi_buffer_only_for_local_buffer:
            # Keep defaults when tuning doesn't make sense
            configs.append(
                triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN},
                    multibuffer=multibuffer,
                    unit_flag=unit_flag,
                    limit_auto_multi_buffer_only_for_local_buffer=limit_auto_multi_buffer_only_for_local_buffer,
                    limit_auto_multi_buffer_of_local_buffer=limit_auto_multi_buffer_of_local_buffer,
                )
            )
        else:
            # Fully expand tuning space
            for (
                set_workspace_multibuffer,
                enable_hivm_auto_cv_balance,
                tile_mix_vector_loop,
                tile_mix_cube_loop,
            ) in itertools.product(
                set_workspace_multibuffer_list,
                enable_hivm_auto_cv_balance_list,
                tile_mix_vector_loop_num_list,
                tile_mix_cube_loop_num_list,
            ):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": BM, "BLOCK_N": BN},
                        multibuffer=multibuffer,
                        unit_flag=unit_flag,
                        limit_auto_multi_buffer_only_for_local_buffer=limit_auto_multi_buffer_only_for_local_buffer,
                        limit_auto_multi_buffer_of_local_buffer=limit_auto_multi_buffer_of_local_buffer,
                        set_workspace_multibuffer=set_workspace_multibuffer,
                        enable_hivm_auto_cv_balance=enable_hivm_auto_cv_balance,
                        tile_mix_vector_loop=tile_mix_vector_loop,
                        tile_mix_cube_loop=tile_mix_cube_loop,
                    )
                )
    # print(f"configs: {configs}")
    return configs


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 512}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}),
    ],
    key=["BSZ", "Q_HEAD_NUM", "SEQ", "HEAD_DIM","HAS_MASK"],
)
@triton.jit
def _sdpa_infer_kernel(
    Q,
    K,
    V,
    mask,
    M,
    Out,
    scale,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BSZ: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    KV_HEAD_NUM: tl.constexpr,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = SEQ // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (BSZ) × number of attention heads (Q_HEAD_NUM)
    NUM_BLOCKS = NUM_BLOCKS_M * BSZ * Q_HEAD_NUM

    # Current M-dimension block index
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    for block_idx in range(pid, NUM_BLOCKS, core_step):
        task_bn_idx = block_idx // NUM_BLOCKS_M
        task_seq_idx = block_idx % NUM_BLOCKS_M

        bsz_offset = task_bn_idx // Q_HEAD_NUM
        q_head_num_offset = task_bn_idx % Q_HEAD_NUM
        kv_head_num_offset = q_head_num_offset // (Q_HEAD_NUM // KV_HEAD_NUM)
        q_bn_offset = bsz_offset.to(tl.int64) * stride_qz + q_head_num_offset.to(tl.int64) * stride_qh
        kv_bn_offset = bsz_offset.to(tl.int64) * stride_kz + kv_head_num_offset.to(tl.int64) * stride_kh
        # Create block pointers for Q, K, V, Output
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_seq_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_seq_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # Initialize offsets
        offs_m = task_seq_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        # Initialize accumulator
        acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)

        acc_ptr, l_i, m_i = _sdpa_infer_inner(
            acc_ptr,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            mask,
            task_seq_idx,
            scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
            SEQ,
            HAS_MASK,
            V.dtype.element_ty == tl.float8e5,
        )

        m_i += tl.math.log(l_i)
        accumulator = acc_ptr / l_i[:, None]

        # NOTE(zhangjihang): for training
        # m_ptrs = M + task_bn_idx * SEQ + offs_m
        # tl.store(m_ptrs, m_i)

        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


@triton.autotune(
    configs=[
        triton.Config({"multibuffer": True, "BLOCK_R": 128, "BLOCK_C": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 256}),
        triton.Config({"multibuffer": False, "BLOCK_R": 256, "BLOCK_C": 128}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 128}),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sdpa_fwd(
    q,
    k,
    v,
    o,
    lse,
    mask,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)

        ptr_q = (
            q
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_o = (
            o
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_lse = (
            lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_lse = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S

        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_o = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
        block_l = tl.full([BLOCK_R], 0.0, dtype=HIGH_TYPE)
        block_m = tl.full([BLOCK_R], -1e6, dtype=HIGH_TYPE)
        for idx_c in range(0, num_c):
            ptr_k = (
                k
                + idx_b * STRIDE_K_B
                + (idx_n // group_size) * STRIDE_K_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_b * STRIDE_V_B
                + (idx_n // group_size) * STRIDE_V_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )
            ptr_mask = (
                mask
                + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            )

            mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
            mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & (
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S
            )

            block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)
            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)

            block_s = tl.dot(block_q, tl.trans(block_k)) * scale
            block_s = tl.where(block_mask, block_s, -1e6)
            block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1,
                                                   propagate_nan=tl.PropagateNan.ALL),
                                   propagate_nan=tl.PropagateNan.ALL)
            block_s = tl.math.exp(block_s - block_m_1[:, None])
            block_l_1= tl.sum(block_s, axis=1)
            alpha = tl.math.exp(block_m - block_m_1)
            block_l = alpha * block_l + block_l_1
            block_o = block_o * alpha[:, None]
            block_o = tl.dot(block_s.to(LOW_TYPE), block_v, block_o).to(HIGH_TYPE)
            block_m = block_m_1


        block_o = block_o / block_l[:, None]
        block_lse = tl.log(block_l) + block_m
        tl.store(ptr_o, block_o.to(LOW_TYPE), mask=mask_q)
        tl.store(ptr_lse, block_lse, mask=mask_lse)


@triton.autotune(
    configs=[
        triton.Config({"multibuffer": True, "BLOCK_R": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 256}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128}),
        triton.Config({"multibuffer": False, "BLOCK_R": 64}),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sdpa_bwd_d(
    o,
    do,
    d,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_O_B: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    for task_id in range(pid, num_r * B * N, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)
        ptr_o = (
            o
            + idx_b * STRIDE_O_B
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_do = (
            do
            + idx_b * STRIDE_O_B
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        mask_o = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S

        block_o = tl.load(ptr_o, mask=mask_o, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_o, other=0.0)
        block_d = tl.sum(block_do.to(HIGH_TYPE) * block_o.to(HIGH_TYPE), axis=1)
        tl.store(ptr_d, block_d, mask=mask_d)


@triton.autotune(
    configs=[
        triton.Config({"multibuffer": True, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 128}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 64, "BLOCK_C": 128}),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sdpa_bwd_q(
    q,
    k,
    v,
    do,
    d,
    dq,
    lse,
    mask,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)

        ptr_q = (
            q
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_do = (
            do
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_dq = (
            dq
            + idx_b * STRIDE_Q_B
            + idx_n * STRIDE_Q_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
            + idx_h[None, :] * STRIDE_Q_H
        )
        ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        ptr_lse = (
            lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S
        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
        block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
        block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
        block_dq = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)

        for idx_c in range(0, num_c):
            ptr_k = (
                k
                + idx_b * STRIDE_K_B
                + (idx_n // group_size) * STRIDE_K_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v
                + idx_b * STRIDE_V_B
                + (idx_n // group_size) * STRIDE_V_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )
            ptr_mask = (
                mask
                + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            )

            mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
            mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & (
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S
            )

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)

            block_s = tl.dot(block_q, block_k.T) * scale
            block_p = tl.exp(block_s - block_lse[:, None])
            block_p = tl.where(block_mask, block_p, 0.0)
            block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
            block_ds = block_p * (block_dp - block_d[:, None]) * scale
            block_dq += tl.dot(block_ds.to(LOW_TYPE), block_k).to(HIGH_TYPE)

        tl.store(ptr_dq, block_dq.to(LOW_TYPE), mask=mask_q)


@triton.autotune(
    configs=[
        triton.Config({"multibuffer": True, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 128}),
        triton.Config({"multibuffer": True, "BLOCK_R": 64, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 128}),
        triton.Config({"multibuffer": False, "BLOCK_R": 128, "BLOCK_C": 64}),
        triton.Config({"multibuffer": False, "BLOCK_R": 64, "BLOCK_C": 128}),
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sdpa_bwd_kv(
    q,
    k,
    v,
    do,
    d,
    dk,
    dv,
    lse,
    mask,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
        idx_b = task_id // (num_group * num_c)
        idx_group = task_id // num_c % num_group
        idx_c = task_id % num_c
        idx_h = tl.arange(0, H)

        ptr_k = (
            k
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_v = (
            v
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )
        ptr_dk = (
            dk
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_dv = (
            dv
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )

        mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
        block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
        block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
        block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
        block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

        for idx_ingroup in range(group_size):
            idx_n = idx_group * group_size + idx_ingroup
            for idx_r in range(0, num_r):
                ptr_q = (
                    q
                    + idx_b * STRIDE_Q_B
                    + idx_n * STRIDE_Q_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                    + idx_h[None, :] * STRIDE_Q_H
                )
                ptr_do = (
                    do
                    + idx_b * STRIDE_Q_B
                    + idx_n * STRIDE_Q_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                    + idx_h[None, :] * STRIDE_Q_H
                )
                ptr_d = (
                    d
                    + idx_b * STRIDE_D_B
                    + idx_n * STRIDE_D_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
                )
                ptr_lse = (
                    lse
                    + idx_b * STRIDE_D_B
                    + idx_n * STRIDE_D_N
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
                )
                ptr_mask = (
                    mask
                    + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S
                    + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
                )

                mask_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
                mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S
                mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & (
                    (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S
                )

                block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
                block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
                block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
                block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
                block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)

                block_s = tl.dot(block_q, block_k.T) * scale
                block_p = tl.exp(block_s - block_lse[:, None])
                block_p = tl.where(block_mask, block_p, 0.0)
                block_dv += tl.dot(block_p.to(LOW_TYPE).T, block_do).to(HIGH_TYPE)
                block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
                block_ds = block_p * (block_dp - block_d[:, None]) * scale
                block_dk += tl.dot(block_ds.to(LOW_TYPE).T, block_q).to(HIGH_TYPE)

        tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
        tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)


def sdpa_infer_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    """
    Forward computation interface:
    Args:
        q: Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        k: Key tensor (K), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        v: Value tensor (V), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        mask: Attention mask, shape [SEQ, SEQ]
        scale: Scaling factor for QK product
    Returns:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
    """
    # shape constraints
    assert q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1]
    head_dim = q.shape[-1]
    assert head_dim in {64, 128}
    assert q.shape[-2] == k.shape[-2] and k.shape[-2] == v.shape[-2]
    seq_length = q.shape[-2]
    has_mask = mask is not None
    if has_mask:
        assert len(mask.shape) == 2 and mask.shape[0] == seq_length and mask.shape[1] == seq_length
        assert mask.dtype == torch.bool

    if not enable_gqa:
        assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    q_head_num = q.shape[1]
    kv_head_num = k.shape[1]

    if scale is None:
        scale = 1.0
    o = torch.empty_like(q)

    extra_kern_args = {}
    cube_num, vector_num = get_device_properties()
    num_cores = cube_num
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    # The pointer is unused by the HAS_MASK=False specialization.
    mask_arg = mask if has_mask else q

    # mask = 1 - mask.to(torch.int8)
    # mask = (1.0 - mask.to(torch.float32)) * (-1e6)
    _sdpa_infer_kernel[(num_cores,)](
        q,
        k,
        v,
        mask_arg,
        M,
        o,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        BSZ=q.shape[0],
        Q_HEAD_NUM=q_head_num,
        KV_HEAD_NUM=kv_head_num,
        SEQ=seq_length,
        HEAD_DIM=head_dim,
        HAS_MASK=has_mask,
        enable_ubuf_saving=True,
        enable_hivm_auto_cv_balance=True,
        multibuffer=True,  # 控制开double_buffer
        unit_flag=True,  # cube搬出的一个优化项
        limit_auto_multi_buffer_only_for_local_buffer=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
        set_workspace_multibuffer=4,
        tile_mix_vector_loop=8,
        tile_mix_cube_loop=4,
        limit_auto_multi_buffer_buffer="only-cube",
        enable_dynamic_cv_pipeline=True,
        enable_cube_block_merge=True,
        hfusion_enable_multiple_consumer_fusion=True,
        intra_cache_num=3,
        inter_cache_num=2,
        **extra_kern_args,
    )
    return o

    # head_num 8:1 16K 32 block_size
    # sdpa -> flash_attention_score ~ 7000+us
    # triton 20000+us -> 14000+us flex_attention(triton + torch.compile)


def sdpa_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
):
    """
    Forward computation interface:
    Args:
        q: Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        k: Key tensor (K), shape [BSZ, KV_HEAD_NUM, SEQ, HEAD_DIM]
        v: Value tensor (V), shape [BSZ, KV_HEAD_NUM, SEQ, HEAD_DIM]
        mask: Attention mask, shape [SEQ, SEQ]
        scale: Scaling factor for QK product
    Returns:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        lse: LogSumExp tensor, shape [BSZ, Q_HEAD_NUM, SEQ]
    """
    # shape constraints
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]

    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] and k.shape[3] == v.shape[3] and q.shape[3] in {64, 128}

    o = torch.empty_like(q)
    lse = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_cores, _ = get_device_properties()

    kernel_sdpa_fwd[(num_cores,)](
        q,
        k,
        v,
        o,
        lse,
        mask,
        scale,
        k.shape[1],
        q.shape[0],
        q.shape[1],
        q.shape[2],
        q.shape[3],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        limit_auto_multi_buffer_buffer="no-limit",
        enable_dynamic_cv_pipeline=True,
        enable_cube_block_merge=True,
        hfusion_enable_multiple_consumer_fusion=True,
        intra_cache_num=3,
        inter_cache_num=2,
    )

    return o, lse



@triton.autotune(
    configs=[
        triton.Config({"multibuffer": False, "BLOCK_R": r, "BLOCK_C": c})
        for r in [64, 128]
        for c in [64, 128]
    ],
    key=["N", "S", "H"],
    reset_to_zero=["dq", "dk", "dv"],
)
@triton.jit
def kernel_sdpa_bwd_qkv(
    q,
    k,
    v,
    do,
    d,
    dq,
    dk,
    dv,
    lse,
    mask,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    group_num = N // group_size
    idx_h = tl.arange(0, H)
    start_id = pid % 2
    for idx_br in range(pid // 2 , B * num_r * N, n_programs // 2):
        idx_b = idx_br // num_r // N
        idx_r = idx_br // N % num_r
        idx_n = idx_br % N
        tasks = num_c
        ptr_dq = (
                dq
                + idx_b * STRIDE_Q_B
                + idx_n * STRIDE_Q_N
                + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S
                + idx_h[None, :] * STRIDE_Q_H
        )
        dq_mask = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S

        q_block_ptr = tl.make_block_ptr(
            base=q + idx_b * STRIDE_Q_B + idx_n * STRIDE_Q_N,
            shape=(S, H),
            strides=(STRIDE_Q_S, STRIDE_Q_H),
            offsets=((idx_r * BLOCK_R).to(tl.int32), 0),
            block_shape=(BLOCK_R, H),
            order=(1, 0),
        )
        block_q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        do_block_ptr = tl.make_block_ptr(
            base=do + idx_b * STRIDE_Q_B + idx_n * STRIDE_Q_N,
            shape=(S, H),
            strides=(STRIDE_Q_S, STRIDE_Q_H),
            offsets=((idx_r * BLOCK_R).to(tl.int32), 0),
            block_shape=(BLOCK_R, H),
            order=(1, 0),
        )
        block_do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero")

        block_dq = tl.zeros((BLOCK_R, H), dtype=tl.float32)
        for idx_c in range(start_id, tasks, 2):
            idx_group = idx_n // group_size

            k_block_ptr = tl.make_block_ptr(
                base=k + idx_b * STRIDE_K_B + idx_group * STRIDE_K_N,
                shape=(S, H),
                strides=(STRIDE_K_S, STRIDE_K_H),
                offsets=((idx_c * BLOCK_C).to(tl.int32), 0),
                block_shape=(BLOCK_C, H),
                order=(1, 0),
            )

            v_block_ptr = tl.make_block_ptr(
                base=v + idx_b * STRIDE_V_B + idx_group * STRIDE_V_N,
                shape=(S, H),
                strides=(STRIDE_V_S, STRIDE_V_H),
                offsets=((idx_c * BLOCK_C).to(tl.int32), 0),
                block_shape=(BLOCK_C, H),
                order=(1, 0),
            )

            q_offs = idx_r * BLOCK_R + tl.arange(0, BLOCK_R)
            mask_d = q_offs < S
            ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + q_offs * STRIDE_D_S
            ptr_lse = lse + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + q_offs * STRIDE_D_S
            block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
            block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)

            ptr_mask = (
                mask
                + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            )
            mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & (
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S
            )
            block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)


            ptr_dk = (
                dk
                + pid // 2 * STRIDE_K_B * B
                + idx_b * STRIDE_K_B
                + idx_group * STRIDE_K_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
                + idx_h[None, :] * STRIDE_K_H
            )
            dk_mask = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S

            ptr_dv = (
                dv
                + pid // 2 * STRIDE_K_B * B
                + idx_b * STRIDE_V_B
                + idx_group * STRIDE_V_N
                + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
                + idx_h[None, :] * STRIDE_V_H
            )

            block_k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            k_T = tl.trans(block_k)
            qk = tl.dot(block_q, k_T)
            qk = qk * scale
            p = tl.math.exp(qk - block_lse[:, None])

            p = tl.where(block_mask, p, 0.0)
            # dv
            block_dv = tl.dot(tl.trans(p.to(block_k.dtype)), block_do)
            tl.atomic_add(ptr_dv, block_dv, mask=dk_mask)
            # dp
            block_v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_T = tl.trans(block_v)
            dp = tl.dot(block_do, v_T)
            # ds
            ds = p * (dp - block_d[:, None]) * scale
            ds_cast = ds.to(block_q.dtype)

            # dk
            block_dk = tl.dot(tl.trans(ds_cast), block_q)
            tl.atomic_add(ptr_dk, block_dk, dk_mask)

            # dq
            block_dq = tl.dot(ds_cast, block_k, block_dq)
            # tl.atomic_add(ptr_dq, block_dq.to(ptr_dq.dtype.element_ty), dq_mask)

        tl.atomic_add(ptr_dq, block_dq, mask=dq_mask)
        # tl.store(ptr_dq, block_dq.to(ptr_dq.dtype.element_ty), mask=dq_mask)


_BWD_ATOMIC_PRESSURE_THRESHOLD = 256


def _select_bwd_branch(seq_len, q_head_num, kv_head_num, block_r=128):
    """
    Return True for the fused (QKV) kernel, False for the split (Q + KV) kernels.
    Two backward strategies exist:

      "fused"  – kernel_sdpa_bwd_qkv computes dq, dk, dv in a single kernel.
                 dq is accumulated locally and written once; dk/dv must use
                 atomic_add into a per-core workspace (later reduced with sum).

      "split"  – kernel_sdpa_bwd_q (dq) + kernel_sdpa_bwd_kv (dk, dv).
                 Each kernel picks its own optimal parallelisation dimension,
                 so NO atomics and NO workspace are needed, but the "front-end"
                 work (Q@K^T, softmax, do@V^T) is computed twice.

    The dominant cost driver of the fused path is the number of atomic_add
    writes that land on each KV-block:

        atomic_pressure = num_r * group_size
                        = (seq_len / BLOCK_R) * (q_head_num / kv_head_num)

    When atomic_pressure is small the fused kernel wins because it avoids the
    redundant front-end computation.  When atomic_pressure is large the
    atomic + workspace + reduction overhead explodes and the split kernels win.

    The threshold below was calibrated on Ascend NPU.  Use force_branch in the
    benchmark script to sweep shapes and refine it.

    """
    num_r = (seq_len + block_r - 1) // block_r
    group_size = q_head_num // kv_head_num
    atomic_pressure = num_r * group_size
    return atomic_pressure < _BWD_ATOMIC_PRESSURE_THRESHOLD


def sdpa_bwd_impl(
    o: torch.Tensor,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
    force_branch: Optional[str] = None,
):
    """
    Backward computation interface:
    Args:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        do: Gradient tensor for o, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        q: Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        k: Key tensor (K), shape [BSZ, KV_HEAD_NUM, SEQ, HEAD_DIM]
        v: Value tensor (V), shape [BSZ, KV_HEAD_NUM, SEQ, HEAD_DIM]
        lse: Logsumexp tensor, shape [BSZ, Q_HEAD_NUM, SEQ]
        mask: Attention mask, shape [SEQ, SEQ]
        scale: Scaling factor for QK product
    Returns:
        dq: Gradient tensor for Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        dk: Gradient tensor for Key tensor (K), shape [BSZ, KV_HEAD_NUM, SEQ, HEAD_DIM]
        dv: Gradient tensor for Value tensor (V), shape [BSZ, KV_HEAD_NUM, SEQ, HEAD_DIM]
    """
    # shape constraints
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4 and len(lse.shape) == 3
    assert q.shape == o.shape and o.shape == do.shape
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]
    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] and k.shape[3] == v.shape[3] and q.shape[3] in {64, 128}
    assert q.shape[0] == lse.shape[0] and q.shape[1] == lse.shape[1] and q.shape[2] == lse.shape[2]

    num_cores, num_vec_cores = get_device_properties()
    d = torch.empty_like(lse)
    dq = torch.zeros_like(q)

    kernel_sdpa_bwd_d[(num_vec_cores,)](
        o,
        do,
        d,
        o.shape[0],
        o.shape[1],
        o.shape[2],
        o.shape[3],
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        d.stride(0),
        d.stride(1),
        d.stride(2),
    )

    # -- Select backward strategy -------------------------------------------
    # force_branch: "fused" | "split" | None (auto-select via heuristic)
    if force_branch is not None:
        use_fused = force_branch == "fused"
    else:
        use_fused = _select_bwd_branch(q.shape[2], q.shape[1], k.shape[1])

    if use_fused:
        bsz, kv_head, seq_len, head_dim = k.shape
        # Per-core workspace for dk/dv to reduce atomic_add conflicts.
        # The fused kernel indexes this dimension with (pid // 2), so we
        # need (num_cores + 1) // 2 slots.
        num_workspace = (num_cores + 1) // 2
        core_dk = torch.zeros((num_workspace, bsz, kv_head, seq_len, head_dim), dtype=torch.float32, device=k.device)
        core_dv = torch.zeros((num_workspace, bsz, kv_head, seq_len, head_dim), dtype=torch.float32, device=v.device)
        kernel_sdpa_bwd_qkv[(num_cores,)](
            q,
            k,
            v,
            do,
            d,
            dq,
            core_dk,
            core_dv,
            lse,
            mask,
            scale,
            k.shape[1],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            unit_flag=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=2,
            inter_cache_num=2,
        )
        dk = torch.sum(core_dk, 0)
        dv = torch.sum(core_dv, 0)
    else:
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        kernel_sdpa_bwd_q[(num_cores,)](
            q,
            k,
            v,
            do,
            d,
            dq,
            lse,
            mask,
            scale,
            k.shape[1],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            unit_flag=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=1,
        )

        kernel_sdpa_bwd_kv[(num_cores,)](
            q,
            k,
            v,
            do,
            d,
            dk,
            dv,
            lse,
            mask,
            scale,
            k.shape[1],
            q.shape[0],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            d.stride(0),
            d.stride(1),
            d.stride(2),
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            unit_flag=True,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            intra_cache_num=3,
            inter_cache_num=2,
        )
    return dq, dk.to(k.dtype), dv.to(v.dtype)