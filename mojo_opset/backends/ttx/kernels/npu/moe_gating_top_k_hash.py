# Copyright 2026, The FlagOS Contributors.

"""TTX Triton kernels for hash-based MoE gating on Ascend."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .utils import get_num_cores, libentry

try:
    import triton.experimental.tle as tle

    HAS_TLE = True
except Exception:
    HAS_TLE = False


NORM_SOFTMAX: tl.constexpr = tl.constexpr(0)
NORM_SIGMOID: tl.constexpr = tl.constexpr(1)
NORM_SQRT_SOFTPLUS: tl.constexpr = tl.constexpr(2)

_NUM_VECTORCORE = get_num_cores()


def _moe_gating_grid(n_rows: int) -> tuple[int]:
    return (max(1, min(n_rows, _NUM_VECTORCORE)),)


def _moe_gating_block_e(n_experts: int) -> int:
    return triton.next_power_of_2(n_experts)


def _moe_gating_block_k(k: int) -> int:
    return triton.next_power_of_2(k)


@libentry()
@triton.jit
def _moe_gating_top_k_hash_kernel(
    x_ptr,
    input_ids_ptr,
    tid2eid_ptr,
    y_ptr,
    expert_idx_ptr,
    norm_out_ptr,
    row_count: tl.constexpr,
    expert_count: tl.constexpr,
    k: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    eps: tl.constexpr,
    norm_type: tl.constexpr,
    write_norm_out: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """General path for softmax or requests that need the full norm output."""
    pid = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k

    for row in range(pid, row_count, tl.num_programs(0)):
        expert_offsets = tl.arange(0, BLOCK_E)
        expert_mask = expert_offsets < expert_count
        logits = tl.load(
            x_ptr + row * expert_count + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)

        if norm_type == NORM_SOFTMAX:
            logits_for_max = tl.where(expert_mask, logits, float("-inf"))
            row_max = tl.max(
                logits_for_max,
                axis=0,
                propagate_nan=tl.PropagateNan.ALL,
            )
            exp_logits = tl.exp(tl.where(expert_mask, logits - row_max, float("-inf")))
            sum_exp = tl.sum(exp_logits, axis=0)
            scores = exp_logits / sum_exp
        elif norm_type == NORM_SIGMOID:
            scores = tl.sigmoid(logits)
        else:
            scores = tl.sqrt(tl.log(1.0 + tl.exp(logits)))

        if write_norm_out:
            tl.store(
                norm_out_ptr + row * expert_count + expert_offsets,
                scores,
                mask=expert_mask,
            )

        token_id = tl.load(input_ids_ptr + row).to(tl.int64)
        selected_experts = tl.load(
            tid2eid_ptr + token_id * k + k_offsets,
            mask=k_mask,
            other=0,
        ).to(tl.int32)
        selected_logits = tl.load(
            x_ptr + row * expert_count + selected_experts,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        if norm_type == NORM_SOFTMAX:
            selected_scores = tl.exp(selected_logits - row_max) / sum_exp
        elif norm_type == NORM_SIGMOID:
            selected_scores = tl.sigmoid(selected_logits)
        else:
            selected_scores = tl.sqrt(tl.log(1.0 + tl.exp(selected_logits)))

        selected_scores = tl.where(k_mask, selected_scores, 0.0)
        if norm_type != NORM_SOFTMAX:
            selected_scores /= tl.sum(selected_scores, axis=0) + eps
        selected_scores *= routed_scaling_factor

        tl.store(y_ptr + row * k + k_offsets, selected_scores, mask=k_mask)
        tl.store(expert_idx_ptr + row * k + k_offsets, selected_experts, mask=k_mask)


@libentry()
@triton.jit
def _moe_gating_dsa_k_kernel(
    x_ptr,
    input_ids_ptr,
    tid2eid_ptr,
    y_ptr,
    expert_idx_ptr,
    row_count: tl.constexpr,
    expert_count: tl.constexpr,
    k: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    eps: tl.constexpr,
    norm_type: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """TLE path: copy a contiguous logit row to UB, then gather only K values."""
    pid = tl.program_id(0)
    row_offsets = tl.arange(0, BLOCK_E)
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k

    x_ub = tle.dsa.alloc(
        [BLOCK_E],
        dtype=x_ptr.dtype.element_ty,
        mem_addr_space=tle.dsa.ascend.UB,
    )
    for row in range(pid, row_count, tl.num_programs(0)):
        tle.dsa.copy(x_ptr + row * expert_count + row_offsets, x_ub, [expert_count])

        token_id = tl.load(input_ids_ptr + row).to(tl.int64)
        selected_experts = tl.load(
            tid2eid_ptr + token_id * k + k_offsets,
            mask=k_mask,
            other=0,
        ).to(tl.int32)
        row_logits = tle.dsa.to_tensor(x_ub).to(tl.float32)
        selected_logits = tl.gather(row_logits, selected_experts, axis=0)

        if norm_type == NORM_SIGMOID:
            selected_scores = tl.sigmoid(selected_logits)
        else:
            selected_scores = tl.sqrt(tl.log(1.0 + tl.exp(selected_logits)))

        selected_scores = tl.where(k_mask, selected_scores, 0.0)
        selected_scores /= tl.sum(selected_scores, axis=0) + eps
        selected_scores *= routed_scaling_factor

        tl.store(y_ptr + row * k + k_offsets, selected_scores, mask=k_mask)
        tl.store(expert_idx_ptr + row * k + k_offsets, selected_experts, mask=k_mask)


@libentry()
@triton.jit
def _moe_gating_k_kernel_non_tle(
    x_ptr,
    input_ids_ptr,
    tid2eid_ptr,
    y_ptr,
    expert_idx_ptr,
    row_count: tl.constexpr,
    expert_count: tl.constexpr,
    k: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    eps: tl.constexpr,
    norm_type: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Non-TLE path: gather K selected logits directly from global memory."""
    pid = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k

    for row in range(pid, row_count, tl.num_programs(0)):
        token_id = tl.load(input_ids_ptr + row).to(tl.int64)
        selected_experts = tl.load(
            tid2eid_ptr + token_id * k + k_offsets,
            mask=k_mask,
            other=0,
        ).to(tl.int32)
        selected_logits = tl.load(
            x_ptr + row * expert_count + selected_experts,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        if norm_type == NORM_SIGMOID:
            selected_scores = tl.sigmoid(selected_logits)
        else:
            selected_scores = tl.sqrt(tl.log(1.0 + tl.exp(selected_logits)))

        selected_scores = tl.where(k_mask, selected_scores, 0.0)
        selected_scores /= tl.sum(selected_scores, axis=0) + eps
        selected_scores *= routed_scaling_factor

        tl.store(y_ptr + row * k + k_offsets, selected_scores, mask=k_mask)
        tl.store(expert_idx_ptr + row * k + k_offsets, selected_experts, mask=k_mask)


def moe_gating_top_k_hash_infer_impl(
    x: torch.Tensor,
    k: int,
    *,
    input_ids: torch.Tensor = None,
    tid2eid: torch.Tensor = None,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    norm_type: int = 1,
    out_flag: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run hash-based MoE gating with an optional TLE DSA fast path."""
    if x.device.type != "npu":
        raise ValueError(f"x must be on NPU, got {x.device}")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("x must be float16, bfloat16, or float32")
    if k <= 0 or k > 64 or k > x.shape[1]:
        raise ValueError(f"k={k} must be in [1, min(64, expert_count={x.shape[1]})]")
    if norm_type not in (NORM_SOFTMAX, NORM_SIGMOID, NORM_SQRT_SOFTPLUS):
        raise ValueError(
            f"norm_type={norm_type} must be 0 (softmax), 1 (sigmoid), or 2 (sqrt_softplus)"
        )

    row_count, expert_count = x.shape
    if input_ids is None or tid2eid is None:
        raise NotImplementedError(
            "Non-hash top-k selection is not implemented; provide input_ids and tid2eid."
        )
    if input_ids.ndim != 1 or input_ids.shape[0] != row_count:
        raise ValueError(
            f"input_ids must have shape ({row_count},), got {input_ids.shape}"
        )
    if input_ids.dtype != torch.int64 or not input_ids.is_contiguous():
        raise ValueError("input_ids must be a contiguous int64 tensor")
    if input_ids.device != x.device:
        raise ValueError(f"input_ids device {input_ids.device} != x device {x.device}")
    if tid2eid.ndim != 2 or tid2eid.dtype != torch.int32 or not tid2eid.is_contiguous():
        raise ValueError("tid2eid must be a contiguous 2D int32 tensor")
    if tid2eid.device != x.device:
        raise ValueError(f"tid2eid device {tid2eid.device} != x device {x.device}")
    if tid2eid.shape[1] != k:
        raise ValueError(f"tid2eid.shape[1]={tid2eid.shape[1]} must equal k={k}")

    y = torch.empty(row_count, k, dtype=x.dtype, device=x.device)
    expert_idx = torch.empty(row_count, k, dtype=torch.int32, device=x.device)
    grid = _moe_gating_grid(row_count)
    block_k = _moe_gating_block_k(k)

    if norm_type != NORM_SOFTMAX and not out_flag:
        if HAS_TLE:
            _moe_gating_dsa_k_kernel[grid](
                x,
                input_ids,
                tid2eid,
                y,
                expert_idx,
                row_count=row_count,
                expert_count=expert_count,
                k=k,
                routed_scaling_factor=routed_scaling_factor,
                eps=eps,
                norm_type=norm_type,
                BLOCK_E=_moe_gating_block_e(expert_count),
                BLOCK_K=block_k,
                num_warps=1,
                num_stages=1,
            )
        else:
            _moe_gating_k_kernel_non_tle[grid](
                x,
                input_ids,
                tid2eid,
                y,
                expert_idx,
                row_count=row_count,
                expert_count=expert_count,
                k=k,
                routed_scaling_factor=routed_scaling_factor,
                eps=eps,
                norm_type=norm_type,
                BLOCK_K=block_k,
                num_warps=1,
                num_stages=1,
            )
        return y, expert_idx, None

    norm_out = torch.empty(
        row_count,
        expert_count,
        dtype=torch.float32,
        device=x.device,
    )
    _moe_gating_top_k_hash_kernel[grid](
        x,
        input_ids,
        tid2eid,
        y,
        expert_idx,
        norm_out,
        row_count=row_count,
        expert_count=expert_count,
        k=k,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
        norm_type=norm_type,
        write_norm_out=out_flag,
        BLOCK_E=_moe_gating_block_e(expert_count),
        BLOCK_K=block_k,
        num_warps=1,
        num_stages=1,
    )
    return y, expert_idx, norm_out if out_flag else None
