import itertools
import os

from functools import cache
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

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


torch.npu.set_device(0)
# ========== 全局变量和常量 ==========
DEVICE = "npu"
TEST_DATA_DIR = "/home/rjw/test_data"
RESULT_DIR = "./test_results"
RESULT_DIR = "./test_results_batch"
# os.makedirs(RESULT_DIR, exist_ok=True)

os.environ["TRITON_BENCH_METHOD"] = "npu"  # 设置为 NPU 测试方法
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # 打印自动调优信息

test_results = []  # 全局结果存储
valid_fields = [
    "B",
    "N1",
    "S1",
    "D",
    "causal",
    "dtype",
    "BM",
    "BN",
    "From",
    "Testcase Name",
    "sparse mode",
]
dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}

# D 泛化列表, GPU 仅支持 D 为 2 的幂次方
D_FANHUA_LIST = [64, 128]


## this version support HEAD_DIM > 128 and golden baseline is ascendC
@triton.jit
def _attn_fwd_inner(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    start_m,
    qk_scale,  # Starting position of current query block, qk scale factor
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Block size constants
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  # Current stage flag, m and n offset indices
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):  # Total context length, whether to enable FP8 for value precision
    # Set the processing range [lo, hi) for the current stage (in column block units)
    # causal = true
    # stage = 1
    # Causal attention, as the name implies, restricts the flow of information during computation,
    # only allowing the model to see the current and previous positions.
    # In other words, the output at the current position can only depend on the input at or before this position,
    # and cannot access information from future positions.
    # Causal attention ensures sequential order and prevents "leakage of future information."
    # But the following logic will also be triggered
    if STAGE == 1:
        # Stage 1: process all tokens before the query block
        # tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Stage 2: process the current query block
        # tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # Align starting position
    # causal = False (no need for masking)
    else:
        lo, hi = 0, N_CTX  # Process the entire context

    # Adjust K and V block pointers to the starting position `lo`
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))  # K is [HEAD_DIM, N_CTX], shift along the second dim by lo
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))  # V is [N_CTX, HEAD_DIM], shift along the first dim by lo

    # Index mapping for the accumulator , used for slicing when HEAD_DIM >= 256
    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
    block2d_acc = row * HEAD_DIM + col_head_dim

    # Iterate over all k, v blocks in the current stage and accumulate the output
    for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
        # -- Compute qk ----
        k = tl.load(K_block_ptr)
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # Apply causal mask for STAGE 2
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])  # Construct upper triangular mask
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)  # Set invalid positions to -∞
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Update m_ij = max(m_i, max(qk))

            # 0.9x full attention
            # tl.wlere(cond, a, b)
            # ture -> 1,0
            # positive * a + negative * b
            qk -= m_ij[:, None]  # Subtract max for softmax stability
        else:
            qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
            qk = qk - m_ij[:, None]  # Stabilize

        # Softmax weights p = exp(qk)
        p = tl.math.exp(qk)

        # Convert softmax weight type depending on FP8 usage
        if fp8_v:
            p_cast = p.to(tl.float8e5)  # Convert to FP8 format (save memory)
        else:
            p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr)  # Load corresponding V block
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        if HEAD_DIM < 256:
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)
        else:
            # 1. Load current slice of accumulator
            acc = tl.load(acc_ptr + block2d_acc)
            # 2. Update in slices (split by 1/4 of BLOCK_M to avoid ub overflow)
            for i in range(4):
                # Calculate start/end rows for current slice
                offset = i * (BLOCK_M // 4)
                # Extract slice data
                acc_i = tl.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                alpha_i = tl.extract_slice(alpha, [offset], [BLOCK_M // 4], [1])
                pv_i = tl.extract_slice(pv, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                # Incrementally update slice: acc = acc * alpha + pv
                acc_i = acc_i * alpha_i[:, None] + pv_i
                # Write updated slice back to accumulator
                acc = tl.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            # 3. updated accumulator
            tl.store(acc_ptr + block2d_acc, acc)

        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
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
    print(f"configs: {configs}")
    return configs


# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['Z', 'H', 'N_CTX', 'HEAD_DIM'],  # 加入 shape 相关的关键参数
# )
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    M,
    Out,
    acc,
    sm_scale,
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
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Current M-dimension block index
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    for block_idx in range(pid, NUM_BLOCKS, core_step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # Create block pointers for Q, K, V, Output
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # Initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        # Initialize accumulator
        if HEAD_DIM < 256:
            acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        else:
            acc_offset = (
                off_z.to(tl.int64) * stride_qz // stride_qm * HEAD_DIM
                + off_h.to(tl.int64) * stride_qh // stride_qm * HEAD_DIM
                + task_m_idx * BLOCK_M * HEAD_DIM
            )
            acc_ptr = acc + acc_offset

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)

        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_inner(
                acc_ptr,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                task_m_idx,
                sm_scale,  #
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,  #
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,  #
            )
        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc_ptr, l_i, m_i = _attn_fwd_inner(
                acc_ptr,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                task_m_idx,
                sm_scale,  #
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,  #
                2,
                offs_m,
                offs_n,
                N_CTX,
                V.dtype.element_ty == tl.float8e5,  #
            )

        m_i += tl.math.log(l_i)
        if HEAD_DIM < 256:
            accumulator = acc_ptr / l_i[:, None]
        else:
            row = tl.arange(0, BLOCK_M)[:, None]
            col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
            block2d_acc = row * HEAD_DIM + col_head_dim
            accumulator = tl.load(acc_ptr + block2d_acc)
            accumulator = accumulator / l_i[:, None]

        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


def attention_with_any_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
):
    """
        Forward computation interface:
        Args:
            ctx: Context object
            q: Query tensor (Q), shape [Z, H, N_CTX, HEAD_DIM]
            k: Key tensor (K), shape [Z, H, N_CTX, HEAD_DIM]
            v: Value tensor (V), shape [Z, H, N_CTX, HEAD_DIM]
            sm_scale: Scaling factor for QK product
        Returns:
            o: Attention output tensor, shape [Z, H, N_CTX, HEAD_DIM]
        """
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {64, 128, 256}  # 注释用于泛化测试 HEAD_DIM_K

        o = torch.empty_like(q)

        stage = 1
        extra_kern_args = {}
        # Tuning for AMD target
        # if is_hip():
        #     waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        #     extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        cube_num, vector_num = get_device_properties()
        num_cores = cube_num
        acc = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K),
            dtype=torch.float32,
            device=q.device,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        _attn_fwd[(num_cores,)](
            q,
            k,
            v,
            M,
            o,
            acc,
            sm_scale,
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
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
            BLOCK_M=128,
            BLOCK_N=512,
            multibuffer=True,  # autotune config, 控制开double buffer
            unit_flag=True,  # autotune config, cube搬出的一个优化项
            limit_auto_multi_buffer_only_for_local_buffer=False,  # autotune config, 是否开启cube和vector的并行，false表示开启
            set_workspace_multibuffer=4,  # autotune config, 表示同时cube和vector有几个并行，【2,4】，仅limit_auto_multi_buffer_only_for_local_buffer=False 时生效
            enable_hivm_auto_cv_balance=True,
            tile_mix_vector_loop=2,  # 中间vector切分； 1:2
            tile_mix_cube_loop=4,  # (128, 128) * (128, 512); (M, N)大的切分
            **extra_kern_args,
        )

        # )  # set_workspace_multibuffer: 2, tile_mix_vector_loop: 4, tile_mix_cube_loop: 2
        # 以下参数用于autotune
        # BLOCK_M=BM,
        # BLOCK_N=BN,
        # multibuffer=True, # autotune config, 控制开double buffer
        # unit_flag=True, # autotune config, cube搬出的一个优化项
        # limit_auto_multi_buffer_only_for_local_buffer=False, # autotune config, 是否开启cube和vector的并行，false表示开启
        # set_workspace_multibuffer=4, # autotune config, 表示同时cube和vector有几个并行，【2,4】，仅limit_auto_multi_buffer_only_for_local_buffer=False 时生效
        # enable_hivm_auto_cv_balance=True,
        # tile_mix_vector_loop=2,  # 中间vector切分； 1:2
        # tile_mix_cube_loop=4,    # (128, 128) * (128, 512); (M, N)大的切分
        # **extra_kern_args)
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o