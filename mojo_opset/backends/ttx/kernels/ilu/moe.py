from __future__ import annotations

import torch
import triton
import triton.language as tl

from .group_gemm import m_grouped_matmul_impl
from .utils import libentry, smart_triton_autotune


@libentry()
@triton.jit
def _moe_gating_fused_kernel(
    x_ptr,
    w_ptr,
    out_values_ptr,
    out_indices_ptr,
    stride_x_row,
    stride_w_row,
    stride_out_row,
    H: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per token row: ``logits = x @ W`` then softmax, top-``TOP_K``, renormalize.

    Tie-break: among tied maxima on valid experts, pick the smallest expert index. This matches
    ``torch.topk`` when probabilities are pairwise distinct; with exact ties it may differ slightly.
    """
    row_id = tl.program_id(0).to(tl.int64)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N_EXPERTS

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        x_vec = tl.load(
            x_ptr + row_id * stride_x_row + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        w_ptrs = w_ptr + k_offsets[:, None] * stride_w_row + n_offsets[None, :]
        w_tile = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(x_vec[:, None] * w_tile, axis=0)

    logits = tl.where(n_mask, acc, float("-inf"))
    row_max = tl.max(logits, axis=0)
    # If every expert is -inf, row_max is -inf and logits - row_max is NaN (-inf - -inf).
    shift = tl.where(row_max == float("-inf"), 0.0, row_max)
    row_exp = tl.exp(logits - shift)
    row_exp = tl.where(n_mask, row_exp, 0.0)
    row_sum = tl.sum(row_exp, axis=0)
    row_sum_safe = tl.where(row_sum > 0.0, row_sum, 1.0)
    probs = row_exp / row_sum_safe
    probs = tl.where(row_sum > 0.0, probs, 0.0)

    vals_base = out_values_ptr + row_id * stride_out_row
    idxs_base = out_indices_ptr + row_id * stride_out_row
    topk_sum = 0.0

    for _k in range(TOP_K):
        cur_max = tl.max(probs, axis=0)
        is_max = (probs == cur_max) & n_mask
        tied_idx = tl.where(is_max, n_offsets, BLOCK_N)
        best_idx = tl.min(tied_idx, axis=0)
        # Sentinel BLOCK_N when no valid expert wins; clamp to [0, N_EXPERTS).
        best_idx = tl.minimum(best_idx, N_EXPERTS - 1)

        tl.store(vals_base + _k, cur_max)
        tl.store(idxs_base + _k, best_idx.to(tl.int32))
        topk_sum += cur_max

        probs = tl.where(n_offsets == best_idx, float("-inf"), probs)

    rcp_sum = 1.0 / tl.where(topk_sum > 0.0, topk_sum, 1.0)
    rcp_sum = tl.where(topk_sum > 0.0, rcp_sum, 0.0)
    for _k in range(TOP_K):
        v = tl.load(vals_base + _k)
        tl.store(vals_base + _k, v.to(tl.float32) * rcp_sum)


def dense_moe_gating_forward(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single kernel: ``x @ W``, softmax, top-k, renormalize (matches MojoMoEGating)."""
    assert gate_weight.dtype == torch.float32
    if hidden_states.dim() != 2:
        raise ValueError("hidden_states must be 2D")
    num_tokens, h = hidden_states.shape
    h_w, num_experts = gate_weight.shape
    if h != h_w:
        raise ValueError("hidden / gate dim mismatch")

    hs = hidden_states.float() if hidden_states.dtype != torch.float32 else hidden_states
    gw = gate_weight

    BLOCK_N = triton.next_power_of_2(num_experts)
    BLOCK_K = min(128, triton.next_power_of_2(h))
    num_warps = 4 if BLOCK_N <= 64 else 8

    top_k_gates = hs.new_empty(num_tokens, top_k)
    top_k_indices = torch.empty(num_tokens, top_k, dtype=torch.int32, device=hidden_states.device)

    _moe_gating_fused_kernel[(num_tokens,)](
        hs,
        gw,
        top_k_gates,
        top_k_indices,
        hs.stride(0),
        gw.stride(0),
        top_k_gates.stride(0),
        H=h,
        N_EXPERTS=num_experts,
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return top_k_indices, top_k_gates


def _moe_swiglu_autotune_config():
    configs = []
    for BM, BN, nw in [
        (64, 64, 4), (64, 128, 4), (128, 64, 4),
        (128, 128, 8), (128, 256, 8), (256, 128, 8),
    ]:
        for BK in [32, 64, 128]:
            for ns in [2, 3]:
                configs.append(triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                    num_warps=nw, num_stages=ns,
                ))
    return configs


@smart_triton_autotune(configs=_moe_swiglu_autotune_config(), selected_idx=0, key=["HALF_N", "K", "MAX_M"])
@triton.jit
def _moe_grouped_matmul_swiglu_kernel(
    A,
    B,
    C,
    group_offsets_ptr,
    HALF_N: tl.constexpr,
    K: tl.constexpr,
    MAX_M,
    stride_bg,
    strideBK,
    strideBN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped matmul with fused SwiGLU: C = silu(A @ B_gate) * (A @ B_up).

    B shape [num_groups, 2*HALF_N, K] (row-major, transposed load).
    B[:, :HALF_N, :] = gate weights, B[:, HALF_N:, :] = up weights.
    Output C has HALF_N columns per token.
    """
    n_tile_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    group_id = tl.program_id(2)

    if m_tile_id * BLOCK_M >= MAX_M:
        return

    group_start = tl.load(group_offsets_ptr + group_id).to(tl.int32)
    group_end = tl.load(group_offsets_ptr + group_id + 1).to(tl.int32)
    m_g = group_end - group_start

    if m_tile_id * BLOCK_M >= m_g:
        return

    offs_m = group_start + m_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m[:, None] < group_end

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_base = B + group_id * stride_bg

    offs_n_up = offs_n + HALF_N
    b_ptrs = b_base + offs_n[:, None] * strideBN + offs_k[None, :] * strideBK
    b_ptrs_up = b_base + offs_n_up[:, None] * strideBN + offs_k[None, :] * strideBK

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    N: tl.constexpr = HALF_N * 2
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k0 * BLOCK_K)
        a = tl.load(a_ptrs, mask=m_mask & k_mask[None, :], other=0.0)

        b = tl.load(b_ptrs, mask=(offs_n[:, None] < HALF_N) & k_mask[None, :], other=0.0)
        b = tl.trans(b)
        acc = tl.dot(a, b, acc=acc)

        bu = tl.load(b_ptrs_up, mask=(offs_n_up[:, None] < N) & k_mask[None, :], other=0.0)
        bu = tl.trans(bu)
        acc_up = tl.dot(a, bu, acc=acc_up)

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * strideBK
        b_ptrs_up += BLOCK_K * strideBK

    result = (acc * tl.sigmoid(acc)) * acc_up

    c = result.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * HALF_N + offs_n[None, :]
    c_mask = m_mask & (offs_n[None, :] < HALF_N)
    tl.store(c_ptrs, c, mask=c_mask)


def _moe_grouped_matmul_swiglu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    size_per_group: torch.Tensor,
    num_groups: int,
    N: int,
    K: int,
    strideBN: int,
    strideBK: int,
) -> torch.Tensor:
    half_n = N // 2
    cum = size_per_group.cumsum(0, dtype=torch.int32)
    group_offsets = torch.empty(num_groups + 1, dtype=torch.int32, device=A.device)
    group_offsets[0] = 0
    group_offsets[1:] = cum
    max_m = size_per_group.max().item()

    def grid(META):
        return (
            triton.cdiv(half_n, META["BLOCK_N"]),
            triton.cdiv(max_m, META["BLOCK_M"]),
            num_groups,
        )

    _moe_grouped_matmul_swiglu_kernel[grid](
        A,
        B,
        C,
        group_offsets,
        half_n,
        K,
        max_m,
        B.stride(0),
        strideBK,
        strideBN,
    )
    return C


def dense_moe_experts_grouped_forward(
    sorted_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Grouped up linear + SwiGLU + grouped down linear (two ``m_grouped_matmul_impl``, ``swiglu_fwd_impl``)."""
    if sorted_hidden_states.dim() != 2:
        raise ValueError("sorted_hidden_states must be 2D")
    num_experts, n_up, k_in = up_proj_weight.shape
    e2, h_out, k_inter = down_proj_weight.shape
    if e2 != num_experts or k_in != sorted_hidden_states.shape[1]:
        raise ValueError("shape mismatch in dense_moe_experts_grouped_forward")
    inter = k_inter
    if n_up != 2 * inter:
        raise ValueError("up_proj must be 2 * intermediate_size on last dim")

    dtype = sorted_hidden_states.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported activation dtype {dtype}")
    if not sorted_hidden_states.is_contiguous():
        raise ValueError("sorted_hidden_states must be contiguous")

    t_tokens = sorted_hidden_states.shape[0]

    activated = torch.empty(t_tokens, inter, device=sorted_hidden_states.device, dtype=dtype)
    _moe_grouped_matmul_swiglu(
        sorted_hidden_states,
        up_proj_weight,
        activated,
        tokens_per_expert,
        num_experts,
        n_up,
        k_in,
        up_proj_weight.stride(1),
        up_proj_weight.stride(2),
    )

    out = torch.empty(t_tokens, h_out, device=sorted_hidden_states.device, dtype=dtype)
    m_grouped_matmul_impl(
        activated,
        down_proj_weight,
        out,
        tokens_per_expert,
        num_experts,
        t_tokens,
        h_out,
        inter,
        down_proj_weight.stride(1),
        down_proj_weight.stride(2),
        True,
    )
    return out


@libentry()
@triton.jit
def _moe_combine_atomic_fp32_kernel(
    expert_ptr,
    gate_ptr,
    tok_ptr,
    out_fp_ptr,
    R,
    H,
    stride_er,
    stride_eh,
    stride_gr,
    stride_tok,
    out_stride0,
    out_stride1,
    MULTIPLY_GATES: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per (expert_row, hidden_tile): load expert row, optional gate multiply, ``atomic_add`` into ``out_fp_ptr``."""
    r = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1).to(tl.int64)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    v = tl.load(expert_ptr + r * stride_er + offs_h * stride_eh, mask=mask_h, other=0.0).to(tl.float32)
    if MULTIPLY_GATES:
        g = tl.load(gate_ptr + r * stride_gr).to(tl.float32)
        v = v * g
    tok = tl.load(tok_ptr + r * stride_tok).to(tl.int64)

    out_ptrs = out_fp_ptr + tok * out_stride0 + offs_h * out_stride1
    tl.atomic_add(out_ptrs, v, mask=mask_h)


def dense_moe_combine_triton_forward(
    output_buffer: torch.Tensor,
    expert_outputs: torch.Tensor,
    sorted_gates: torch.Tensor,
    token_indices: torch.Tensor,
    multiply_by_gates: bool,
) -> torch.Tensor:
    """Launch ``_moe_combine_atomic_fp32_kernel`` on a fresh fp32 buffer, cast, ``output_buffer.copy_``."""
    num_tokens, h = output_buffer.shape
    r, h2 = expert_outputs.shape
    if h != h2:
        raise ValueError("hidden dim mismatch in combine")
    if sorted_gates.shape[0] != r or token_indices.shape[0] != r:
        raise ValueError("length mismatch in combine")

    if r == 0:
        return output_buffer

    out_fp = torch.empty((num_tokens, h), device=output_buffer.device, dtype=torch.float32)
    out_fp.zero_()

    BLOCK_H = 128
    grid = (int(r), triton.cdiv(int(h), BLOCK_H))
    _moe_combine_atomic_fp32_kernel[grid](
        expert_outputs,
        sorted_gates,
        token_indices,
        out_fp,
        int(r),
        int(h),
        expert_outputs.stride(0),
        expert_outputs.stride(1),
        sorted_gates.stride(0),
        token_indices.stride(0),
        out_fp.stride(0),
        out_fp.stride(1),
        MULTIPLY_GATES=multiply_by_gates,
        BLOCK_H=BLOCK_H,
    )

    reduced = out_fp.to(output_buffer.dtype)
    output_buffer.copy_(reduced)
    return output_buffer


def _moe_dispatch_argsort_expert_stable(flat_expert: torch.Tensor) -> torch.Tensor:
    """Stable argsort by expert id — keeps original token order within same expert.

    Why torch.argsort instead of a Triton kernel:
      argsort is a global operation (all r elements must participate in comparison).
      Triton programs are independent tiles with no cross-program synchronisation,
      so a single-kernel sort is infeasible for arbitrary r. torch.argsort delegates
      to CUB radix sort on GPU which is already highly optimised for integer keys.
      The downstream gather is the part that benefits from Triton fusion.
    """
    return torch.argsort(flat_expert, dim=0, stable=True)


@libentry()
@triton.jit
def _moe_dispatch_gather_fused_kernel(
    sorted_hidden_ptr,
    sorted_gates_ptr,
    token_indices_ptr,
    hidden_ptr,
    flat_gates_ptr,
    sort_indices_ptr,
    R,
    H,
    stride_h0,
    stride_h1,
    stride_s0,
    stride_s1,
    stride_flat_g,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1).to(tl.int64)
    if r >= R:
        return
    flat_i = tl.load(sort_indices_ptr + r).to(tl.int64)
    tok = flat_i // TOP_K
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    vals = tl.load(
        hidden_ptr + tok * stride_h0 + offs * stride_h1,
        mask=mask,
        other=0.0,
    )
    tl.store(sorted_hidden_ptr + r * stride_s0 + offs * stride_s1, vals, mask=mask)
    if tl.program_id(1) == 0:
        g = tl.load(flat_gates_ptr + flat_i * stride_flat_g)
        tl.store(sorted_gates_ptr + r, g)
        tl.store(token_indices_ptr + r, tok.to(tl.int32))


def dense_moe_dispatch_triton_forward(
    hidden_states: torch.Tensor,
    top_k_gates: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, h = hidden_states.shape
    top_k = top_k_indices.shape[-1]
    r = num_tokens * top_k
    device = hidden_states.device

    flat_expert = top_k_indices.reshape(-1)
    counts = torch.bincount(flat_expert.long(), minlength=num_experts).to(torch.int32)
    sort_indices = _moe_dispatch_argsort_expert_stable(flat_expert)

    flat_gates = top_k_gates.reshape(-1, 1)

    sorted_hidden = torch.empty((r, h), dtype=hidden_states.dtype, device=device)
    sorted_gates = torch.empty((r, 1), dtype=flat_gates.dtype, device=device)
    token_indices = torch.empty(r, dtype=torch.int32, device=device)

    sort_i32 = sort_indices.to(torch.int32)

    BLOCK_H = 128
    grid_h = (r, triton.cdiv(h, BLOCK_H))
    _moe_dispatch_gather_fused_kernel[grid_h](
        sorted_hidden,
        sorted_gates,
        token_indices,
        hidden_states,
        flat_gates,
        sort_i32,
        r,
        h,
        hidden_states.stride(0),
        hidden_states.stride(1),
        sorted_hidden.stride(0),
        sorted_hidden.stride(1),
        flat_gates.stride(0),
        TOP_K=top_k,
        BLOCK_H=BLOCK_H,
    )
    return sorted_hidden, counts, sorted_gates, token_indices


def moe_gating_impl(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
):
    """Delegates to ``dense_moe_gating_forward`` (same math as ``MojoMoEGating.forward``)."""
    return dense_moe_gating_forward(hidden_states, gate_weight, top_k)


def moe_dispatch_impl(
    hidden_states: torch.Tensor,
    top_k_gates: torch.Tensor,
    top_k_indices: torch.Tensor,
    num_experts: int,
):
    if not hidden_states.is_cuda:
        raise RuntimeError("moe_dispatch_impl (ILU TTX) requires CUDA tensors; got non-CUDA hidden_states.")
    return dense_moe_dispatch_triton_forward(
        hidden_states,
        top_k_gates,
        top_k_indices,
        num_experts,
    )

def moe_experts_impl(
    sorted_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
):
    return dense_moe_experts_grouped_forward(
        sorted_hidden_states,
        tokens_per_expert,
        up_proj_weight,
        down_proj_weight,
    )

def moe_combine_impl(
    output_buffer: torch.Tensor,
    expert_outputs: torch.Tensor,
    sorted_gates: torch.Tensor,
    token_indices: torch.Tensor,
    multiply_by_gates: bool = True,
):
    return dense_moe_combine_triton_forward(
        output_buffer,
        expert_outputs,
        sorted_gates,
        token_indices,
        multiply_by_gates,
    )
