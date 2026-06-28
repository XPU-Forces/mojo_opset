import math

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_chunk_indices


def load_balance_sorted_stride(
        cu_q_lens: list,
        kvlens: list,
        num_q_heads: int,
        n_programs: int,
        BLOCK_M: int,
        BLOCK_N: int,
):
    """
    Sorted Round-Robin load balancing:
    1. Enumerate all tasks and compute each task's workload
    2. Sort tasks by workload in descending order
    3. Round-robin assignment: sorted position i -> core i % n_programs

    After sorting, adjacent tasks have similar workload. Round-robin ensures
    every core gets a uniform mix of high/low workload tasks, achieving better
    balance than the original unsorted stride (opt).

    Returns:
        task_tensor: (num_tasks, 5) [core_id, q_head_id, b_id, q_block_id, workload]
        core_range_tensor: (n_programs, 2) [start_idx, end_idx]
    """
    num_seqs = len(cu_q_lens) - 1

    # ---- Step 1: enumerate all tasks and compute workload ----
    tasks = []  # (workload, q_head_id, b_id, q_block_id)
    for b_id in range(num_seqs):
        q_seq_len = cu_q_lens[b_id + 1] - cu_q_lens[b_id]
        kv_seq_len = kvlens[b_id]
        kv_computed_len = kv_seq_len - q_seq_len
        num_q_chunks = (q_seq_len + BLOCK_M - 1) // BLOCK_M
        for q_block_id in range(num_q_chunks):
            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            num_total_blocks = (q_block_start + kv_computed_len + q_block_len + BLOCK_N - 1) // BLOCK_N
            for q_head_id in range(num_q_heads):
                tasks.append((num_total_blocks, q_head_id, b_id, q_block_id))

    total_tasks = len(tasks)

    if total_tasks == 0:
        task_tensor = torch.empty(0, 5, dtype=torch.long)
        core_range_tensor = torch.empty(n_programs, 2, dtype=torch.long)
        return task_tensor, core_range_tensor

    # ---- Step 2: sort by workload in descending order ----
    tasks.sort(key=lambda x: x[0], reverse=True)

    # ---- Step 3: round-robin assignment (sorted stride) ----
    core_tasks = [[] for _ in range(n_programs)]
    for i, (workload, q_head_id, b_id, q_block_id) in enumerate(tasks):
        core_id = i % n_programs
        core_tasks[core_id].append([core_id, q_head_id, b_id, q_block_id, workload])

    # ---- Step 4: pack into contiguous tensors ----
    task_list = []
    core_ranges = []
    current_idx = 0
    for core_id in range(n_programs):
        start_idx = current_idx
        for task in core_tasks[core_id]:
            task_list.append(task)
            current_idx += 1
        end_idx = current_idx
        core_ranges.append([start_idx, end_idx])

    task_tensor = torch.tensor(task_list, dtype=torch.long)
    core_range_tensor = torch.tensor(core_ranges, dtype=torch.long)

    # ---- Print Load Balance Info ----
    # core_workloads = [sum(t[4] for t in core_tasks[i]) for i in range(n_programs)]
    # core_counts = [len(core_tasks[i]) for i in range(n_programs)]
    # max_wl = max(core_workloads)
    # min_wl = min(core_workloads)
    # avg_wl = sum(core_workloads) / n_programs
    # print(f"[SortedStride] Total task number: {total_tasks}, 核数: {n_programs}")
    # print(f"[SortedStride] Task number per core: min={min(core_counts)}, max={max(core_counts)}")
    # print(f"[SortedStride] Load per core: min={min_wl}, max={max_wl}, avg={avg_wl:.1f}, "
    #       f"diff={max_wl - min_wl} ({(max_wl - min_wl) / avg_wl * 100:.1f}%)")

    return task_tensor.npu(), core_range_tensor.npu()


@triton.jit
def causal_mask_fn(mask_ptr, mask_size, mask_stride_m, mask_stride_n, q_start, kv_start, Q_BLOCK, KV_BLOCK):
    offset_causal = min(max(kv_start - q_start, -mask_size), mask_size)
    offsets_mask_causal = (
        (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m
        + (mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]) * mask_stride_n
    )
    mask_causal = tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)

    return mask_causal


@triton.jit
def _sdpa_infer_single_block(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_T_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    qk_scale,
    mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    # -- Compute qk ----

    # Load (transposed) K block
    k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, k_T)
    # tl.compile_hint(qk, "tile_cube_loop")

    qk = qk * qk_scale
    if mask is not None:
        qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)    # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k_T.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, 1)
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)
    # tl.compile_hint(acc_ptr, "tile_cube_loop")

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def paged_prefill_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    batch_size,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_mask_m,
    stride_mask_n,
    softmax_scale,
    core_tasks,  # (num_tasks, 5): [core_id, q_head_id, b_id, q_block_id, workload]
    core_tasks_ranges,  # (n_programs, 2): [start_idx, end_idx]
    AUX_MASK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)

    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")

    task_start = tl.load(core_tasks_ranges + pid * 2)
    task_end = tl.load(core_tasks_ranges + pid * 2 + 1)

    for task_id in range(task_start, task_end):
        q_head_id = tl.load(core_tasks + task_id * 5 + 1)
        b_id = tl.load(core_tasks + task_id * 5 + 2)
        q_block_id = tl.load(core_tasks + task_id * 5 + 3)
        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc

        if seqlens_kv_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
        kv_cache_len = kv_seq_len - q_seq_len

        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
        q_block_end_in_seq = min(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
        q_block_len = q_block_end_in_seq - q_block_start_in_seq

        Q_block_ptr = tl.make_block_ptr(
            base=q_ptr + (q_start_loc + q_block_start_in_seq) * stride_qt + q_head_id * stride_qh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_qt, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=o_ptr + (q_start_loc + q_block_start_in_seq) * stride_ot + q_head_id * stride_oh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_ot, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)
        for kv_block_id in range(0, num_kv_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=key_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=value_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            mask = causal_mask_fn(
                aux_mask_ptr,
                AUX_MASK_SIZE,
                stride_mask_m,
                stride_mask_n,
                kv_cache_len + q_block_start_in_seq,
                kv_block_start_in_seq,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )

            acc, l_i, m_i = _sdpa_infer_single_block(
                acc,
                l_i,
                m_i,
                q,
                K_T_block_ptr,
                V_block_ptr,
                softmax_scale,
                mask,
                HEAD_DIM,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_D,
                value_cache_ptr.dtype.element_ty == tl.float8e5,
            )

        m_i += tl.math.log(l_i)
        accumulator = acc / l_i[:, None]

        # NOTE(zhangjihang): for training
        # m_ptrs = M + task_bn_idx * sub_kv_len + offs_m
        # tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def paged_prefill_page_aggregation_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    batch_size,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_mask_m,
    stride_mask_n,
    softmax_scale,
    core_tasks,  # (num_tasks, 5): [core_id, q_head_id, b_id, q_block_id, workload]
    core_tasks_ranges,  # (n_programs, 2): [start_idx, end_idx]
    AUX_MASK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    PAGE_AGGREGATION_NUM: tl.constexpr,
):
    pid = tl.program_id(0)

    task_start = tl.load(core_tasks_ranges + pid * 2)
    task_end = tl.load(core_tasks_ranges + pid * 2 + 1)

    for task_id in range(task_start, task_end):
        q_head_id = tl.load(core_tasks + task_id * 5 + 1)
        b_id = tl.load(core_tasks + task_id * 5 + 2)
        q_block_id = tl.load(core_tasks + task_id * 5 + 3)
        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc

        if seqlens_kv_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
        kv_cache_len = kv_seq_len - q_seq_len

        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
        q_block_end_in_seq = min(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
        q_block_len = q_block_end_in_seq - q_block_start_in_seq

        Q_block_ptr = tl.make_block_ptr(
            base=q_ptr + (q_start_loc + q_block_start_in_seq) * stride_qt + q_head_id * stride_qh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_qt, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=o_ptr + (q_start_loc + q_block_start_in_seq) * stride_ot + q_head_id * stride_oh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_ot, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)
        for kv_block_id in range(0, num_kv_blocks, PAGE_AGGREGATION_NUM):
            mask = causal_mask_fn(
                aux_mask_ptr,
                AUX_MASK_SIZE,
                stride_mask_m,
                stride_mask_n,
                kv_cache_len + q_block_start_in_seq,
                kv_block_id * BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N * PAGE_AGGREGATION_NUM,
            )


            # Load (transposed) K block
            k = tl.zeros((PAGE_AGGREGATION_NUM * BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=key_cache_ptr.dtype.element_ty)
            for page_iter in range(PAGE_AGGREGATION_NUM):
                kv_block_start = (kv_block_id + page_iter) * BLOCK_SIZE_N
                kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = max(kv_block_end - kv_block_start, 0)
                logical_page_id = min(kv_block_start // PAGE_SIZE, stride_bt_batch - 1)
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )
                cur_k_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_k_blksz, stride_k_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                k_slice = tl.load(cur_k_block_ptr, boundary_check=(
                    0, 1), padding_option="zero")
                k = tl.extra.cann.extension.insert_slice(k, k_slice, offsets=(page_iter * BLOCK_SIZE_N, 0),
                                                         sizes=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                                                         strides=(1, 1))
            k_T = tl.trans(k)
            qk = tl.dot(q, k_T)
            # tl.compile_hint(qk, "tile_cube_loop")

            qk = qk * softmax_scale
            if mask is not None:
                qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)  # Scaled max
            qk = qk - m_ij[:, None]  # Stabilize

            # Softmax weights p = exp(qk)
            p = tl.math.exp(qk)

            p_cast = p.to(k_T.dtype)

            # Softmax denominator (sum of each row)
            l_ij = tl.sum(p, 1)
            # -- Update m_i and l_i
            alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
            l_i = l_i * alpha + l_ij  # Update softmax denominator
            # -- Update output accumulator --
            acc = acc * alpha[:, None]
            # Load corresponding V block
            v = tl.zeros((PAGE_AGGREGATION_NUM * BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=value_cache_ptr.dtype.element_ty)
            for page_iter in range(PAGE_AGGREGATION_NUM):
                kv_block_start = (kv_block_id + page_iter) * BLOCK_SIZE_N
                kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = max(kv_block_end - kv_block_start, 0)
                logical_page_id = min(kv_block_start // PAGE_SIZE, stride_bt_batch - 1)
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v_slice = tl.load(cur_v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                v = tl.extra.cann.extension.insert_slice(v, v_slice, offsets=(page_iter * BLOCK_SIZE_N, 0),
                                                         sizes=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                                                         strides=(1, 1))
            acc = tl.dot(p_cast, v, acc)
            # tl.compile_hint(acc_ptr, "tile_cube_loop")

            # Update current block max
            m_i = m_ij

        m_i += tl.math.log(l_i)
        accumulator = acc / l_i[:, None]

        # NOTE(zhangjihang): for training
        # m_ptrs = M + task_bn_idx * sub_kv_len + offs_m
        # tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def paged_attention_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    aux_mask: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_total_seq_len: Optional[int] = None,
) -> torch.Tensor:
    _, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = key_cache.shape
    batch_size = cu_q_lens.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if aux_mask is None:
        aux_mask = torch.ones(1024, 1024 * 3, device="npu", dtype=torch.bool).tril(1024)

    # Note(chenyifan):
    #   In general, this paged attention kernel works in a `split-q` style.
    #   "bsz * query * q_head" is splited into tasks of shape [BLOCK_SIZE_M, HEAD_DIM]
    #   and then attributed to one program.
    #
    #   Currently, we chunk the queries manually according to a magic CHUNK_SIZE to split queries
    #   It should be better with a autotuned BLOCK_SIZE_M and a pre-configured max_seq_len

    o = torch.empty_like(q)

    CHUNK_SIZE = 128
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))
    cube_num = get_num_cores("cube")
    grid = (cube_num,)
    core_task, core_range_tensor = load_balance_sorted_stride(
        cu_q_lens.tolist(),
        seqlens_kv.tolist(),
        num_q_heads,
        cube_num,
        CHUNK_SIZE,
        BLOCK_SIZE_N,
    )
    if not (page_size < 128 and 128 % page_size == 0):
        paged_prefill_kernel[grid](
            q,
            key_cache,
            value_cache,
            o,
            aux_mask,
            batch_size,
            cu_q_lens,
            seqlens_kv,
            block_tables.to(torch.int32),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            aux_mask.stride(0),
            aux_mask.stride(1),
            softmax_scale,
            core_task,
            core_range_tensor,
            aux_mask.shape[0],
            page_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            BLOCK_SIZE_M=CHUNK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_D=head_dim,
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            enable_dynamic_cv_flow_opt=True,
            intra_cache_num=3,
            inter_cache_num=2,
        )
    else:
        PAGE_AGGREGATION_NUM = 128 // page_size
        paged_prefill_page_aggregation_kernel[grid](
            q,
            key_cache,
            value_cache,
            o,
            aux_mask,
            batch_size,
            cu_q_lens,
            seqlens_kv,
            block_tables.to(torch.int32),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            aux_mask.stride(0),
            aux_mask.stride(1),
            softmax_scale,
            core_task,
            core_range_tensor,
            aux_mask.shape[0],
            page_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            BLOCK_SIZE_M=CHUNK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_D=head_dim,
            PAGE_AGGREGATION_NUM=PAGE_AGGREGATION_NUM,
            enable_dynamic_cv_pipeline=True,
            enable_cube_block_merge=True,
        )

    return o


def _should_use_flash_decode(
    batch_size: int,
    num_kv_heads: int,
    group_size: int,
    max_kv_len: int,
    cube_num: int,
) -> bool:
    """
    FD is triggered when B*N_KV tasks are too few to saturate all cube cores
    and the KV sequence is long enough to benefit from S2 splitting.
    """
    _FD_BN_RATIO = 0.4
    if max_kv_len < 256:
        return False
    loop_times = batch_size * num_kv_heads
    if loop_times >= _FD_BN_RATIO * cube_num:
        return False
    # MHA / MQA: always FD once loop_times threshold is met
    if group_size == 1:
        return True
    # GQA: additionally require long-enough KV to amortise workspace overhead
    return max_kv_len >= 2048


def _compute_kv_split_parts(
    batch_size: int,
    num_kv_heads: int,
    max_kv_len: int,
    cube_num: int,
) -> int:
    """
    Start with aicNum/loopTimes and reduce until each split covers at least
    KV_SPLIT_LIMIT tokens (experience value matching sInnerFactor_=128 path).
    """
    KV_SPLIT_LIMIT = 256
    loop_times = batch_size * num_kv_heads
    max_by_cores = cube_num // loop_times
    max_by_len = max_kv_len // KV_SPLIT_LIMIT
    return max(1, min(max_by_cores, max_by_len))


# ---------------------------------------------------------------------------
# Flash Decode Phase-1 kernel: Cube cores compute partial attention per split
#
# Core assignment:
#   Each program handles one (batch, kv_head, s2_split) triple and processes
#   all GROUP_SIZE Q-heads simultaneously.
#   Tasks are: total = B * N_KV * KV_SPLIT_PARTS, cycled across cube_num cores.
#
# Workspace written per program:
#   acc_ws  [ws_task_idx, g, d]  – locally-normalised partial output (float32)
#   lse_ws  [ws_task_idx, g]     – m_i + log(l_i) per G-head (float32)
# ---------------------------------------------------------------------------

@triton.jit
def paged_decode_fd_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    seqlens_ptr,
    block_tables_ptr,
    acc_ws_ptr,
    lse_ws_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_bt_batch,
    stride_bt_block,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    softmax_scale,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_fd_tasks = BATCH_SIZE * NUM_KV_HEADS * KV_SPLIT_PARTS

    for fd_task_id in range(pid, total_fd_tasks, n_progs):
        split_idx = fd_task_id % KV_SPLIT_PARTS
        kv_task = fd_task_id // KV_SPLIT_PARTS
        b_id = kv_task // NUM_KV_HEADS
        kv_head_id = kv_task % NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        # Partition the KV sequence evenly across splits, aligning chunk_size to
        # PAGE_SIZE so that each split starts at a page boundary.  This avoids KV
        # blocks that cross page boundaries, which would produce incorrect loads
        # because each page maps to a different physical block via block_tables.
        raw_chunk = tl.cdiv(kv_seq_len, KV_SPLIT_PARTS)
        chunk_size = tl.cdiv(raw_chunk, PAGE_SIZE) * PAGE_SIZE
        kv_start = split_idx * chunk_size
        kv_end = tl.minimum(kv_start + chunk_size, kv_seq_len)

        # Workspace slot for this (b, kv_head, split)
        ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)

        # Load Q for all G-heads: [G, D]
        q_ptrs = (
            q_ptr
            + b_id * stride_qb
            + q_head_ids[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        # Iterate over KV blocks within [kv_start, kv_end)
        num_kv_blocks = tl.cdiv(kv_end - kv_start, BLOCK_SIZE_N)

        for kv_block_id in range(num_kv_blocks):
            kv_block_start = kv_start + kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_end)
            kv_block_len = kv_block_end - kv_block_start

            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr
                + b_id * stride_bt_batch
                + logical_page_id * stride_bt_block
            )

            K_T_block_ptr = tl.make_block_ptr(
                base=(
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz
                ),
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=(
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz
                ),
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)  # [G, BLOCK_N]
            qk = qk * softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(
                m_i, tl.max(qk, 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL,
            )
            qk = qk - m_ij[:, None]
            p = tl.math.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + tl.dot(p.to(k_T.dtype), v)
            m_i = m_ij

        # Locally normalise acc by this split's softmax denominator (l_i).
        # The reduce kernel will re-weight each split's contribution using lse_i.
        # Empty splits (kv_start >= kv_seq_len) have l_i=0; guard against div-0.
        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_i_safe[:, None]
        lse_i = tl.where(l_i > 0, m_i + tl.math.log(l_i), float("-inf"))

        # Write lse to workspace
        lse_ptrs = (
            lse_ws_ptr
            + ws_task_idx * stride_lse_task
            + g_offsets * stride_lse_g
        )
        tl.store(lse_ptrs, lse_i)

        # Write acc to workspace
        acc_ptrs = (
            acc_ws_ptr
            + ws_task_idx * stride_aws_task
            + g_offsets[:, None] * stride_aws_g
            + offs_d[None, :] * stride_aws_d
        )
        tl.store(acc_ptrs, acc, mask=offs_d[None, :] < HEAD_DIM)


# ---------------------------------------------------------------------------
# Flash Decode Phase-2 kernel: Vector cores merge partial results
#
# Each program handles one (batch, kv_head) pair and merges KV_SPLIT_PARTS
# partial outputs into the final attention output.
#
# Merge formula (online softmax correction):
#   lse_max_g  = max_i(lse_i_g)          per G-head
#   w_i_g      = exp(lse_i_g - lse_max_g)
#   W_g        = sum_i(w_i_g)
#   out_g      = sum_i (w_i_g / W_g) * acc_ws[i, g, :]
# ---------------------------------------------------------------------------

@triton.jit
def paged_decode_fd_reduce_kernel(
    acc_ws_ptr,
    lse_ws_ptr,
    o_ptr,
    seqlens_ptr,
    stride_aws_task,
    stride_aws_g,
    stride_aws_d,
    stride_lse_task,
    stride_lse_g,
    stride_ob,
    stride_oh,
    stride_od,
    BATCH_SIZE,
    KV_SPLIT_PARTS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_reduce_tasks = BATCH_SIZE * NUM_KV_HEADS

    for reduce_task_id in range(pid, total_reduce_tasks, n_progs):
        b_id = reduce_task_id // NUM_KV_HEADS
        kv_head_id = reduce_task_id % NUM_KV_HEADS

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)

        # Pass 1: find per-G-head max lse across all splits (numerical stability)
        lse_max = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx
            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + g_offsets * stride_lse_g
            )
            lse_max = tl.maximum(lse_max, tl.load(lse_ptrs))

        # Pass 2: weighted accumulation of partial outputs
        out = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)
        exp_sum = tl.zeros((GROUP_SIZE,), dtype=tl.float32)

        for split_idx in tl.static_range(KV_SPLIT_PARTS):
            ws_task_idx = (b_id * NUM_KV_HEADS + kv_head_id) * KV_SPLIT_PARTS + split_idx

            lse_ptrs = (
                lse_ws_ptr
                + ws_task_idx * stride_lse_task
                + g_offsets * stride_lse_g
            )
            lse = tl.load(lse_ptrs)
            w = tl.math.exp(lse - lse_max)  # [G,]; 0 for empty splits (lse=-inf)
            exp_sum += w

            acc_ptrs = (
                acc_ws_ptr
                + ws_task_idx * stride_aws_task
                + g_offsets[:, None] * stride_aws_g
                + offs_d[None, :] * stride_aws_d
            )
            acc_split = tl.load(acc_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)
            out += w[:, None] * acc_split

        # Normalise and write final output
        exp_sum_safe = tl.where(exp_sum > 0, exp_sum, 1.0)
        out = out / exp_sum_safe[:, None]

        o_ptrs = (
            o_ptr
            + b_id * stride_ob
            + q_head_ids[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)


@triton.jit
def paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_TOTAL_BLOCKS,
    MAX_NUM_BLOCKS_PER_SEQ,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be less than BLOCK_SIZE_D")
    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        kv_head_id = kv_task_id % NUM_KV_HEADS
        b_id = kv_task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        # Compute q_head_ids for this kv_head group
        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        # Load q for all heads in the group: [GROUP_SIZE, D]
        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_ids[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        # Initialize softmax state as vectors [GROUP_SIZE,]
        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

        for kv_block_id in range(0, num_kv_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
            kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block)

            # Load K transposed: [D, BLOCK_N] for tl.dot(q, k_T)
            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            # Load V: [BLOCK_N, D] for tl.dot(p, v)
            V_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)

            qk *= softmax_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            qk = qk - m_ij[:, None]

            p = tl.math.exp(qk)
            p_cast = p.to(k_T.dtype)

            pv = tl.dot(p_cast, v)

            # Softmax denominator and update (Vector: parallel with Cube)
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv

            m_i = m_ij

        m_i += tl.math.log(l_i)
        if kv_seq_len > 0:
            # avoid division by zero
            acc = acc / l_i[:, None]

        # Store output for all heads in the group
        o_ptrs = o_ptr + b_id * stride_ob + q_head_ids[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, page_size, head_dim_cache = key_cache.shape
    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)

    cube_num = get_num_cores("cube")
    vector_num = get_num_cores("vector")
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))
    group_size  = num_q_heads // num_kv_heads
    max_kv_len  = int(seqlens.max().item())

    if _should_use_flash_decode(batch_size, num_kv_heads, group_size, max_kv_len, cube_num):
        kv_split_parts = _compute_kv_split_parts(
            batch_size, num_kv_heads, max_kv_len, cube_num
        )

        # Workspace: one slot per (batch, kv_head, split)
        num_ws_tasks = batch_size * num_kv_heads * kv_split_parts
        acc_ws = torch.empty(
            (num_ws_tasks, group_size, head_dim),
            dtype=torch.float32, device=q.device,
        )
        lse_ws = torch.full(
            (num_ws_tasks, group_size),
            float("-inf"), dtype=torch.float32, device=q.device,
        )

        # Phase 1 – Cube cores: partial attention per (b, kv_head, split)
        paged_decode_fd_kernel[(cube_num,)](
            q, key_cache, value_cache, seqlens, block_tables,
            acc_ws, lse_ws,
            q.stride(0), q.stride(1), q.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            block_tables.stride(0), block_tables.stride(1),
            acc_ws.stride(0), acc_ws.stride(1), acc_ws.stride(2),
            lse_ws.stride(0), lse_ws.stride(1),
            softmax_scale,
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        # Phase 2 – Vector cores: online-softmax merge across splits
        paged_decode_fd_reduce_kernel[(vector_num,)](
            acc_ws, lse_ws, o, seqlens,
            acc_ws.stride(0), acc_ws.stride(1), acc_ws.stride(2),
            lse_ws.stride(0), lse_ws.stride(1),
            o.stride(0), o.stride(1), o.stride(2),
            batch_size,
            KV_SPLIT_PARTS=kv_split_parts,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
        return o

    # -----------------------------------------------------------------------
    # Non-FD path: original single-kernel decode
    # -----------------------------------------------------------------------
    paged_decode_kernel[(cube_num,)](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        block_tables,
        batch_size,
        num_total_blocks,
        max_num_blocks_per_seq,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        softmax_scale,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        page_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return o