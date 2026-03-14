from functools import cache
from typing import Any
from typing import Dict
from typing import Tuple

import torch
import triton

from .micro_kernel import packed_bool_to_i8
from .fwd_u import kernel_da_fwd_u
from .fwd_d import kernel_da_fwd_d
from .bwd_d import kernel_da_bwd_d
from .bwd_q_u import kernel_da_bwd_q_u
from .bwd_q_d import kernel_da_bwd_q_d
from .bwd_kv_ul import kernel_da_bwd_kv_ul
from .bwd_kv_ur import kernel_da_bwd_kv_ur
from .bwd_kv_r import kernel_da_bwd_kv_r


@cache
def get_device_properties() -> Tuple[int, int]:
    device = torch.npu.current_device()
    device_properties: Dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(device)

    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


def dllm_attention_up_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float = 1.0,
    BLOCK_SIZE: int = 8,
):
    """
    Forward computation interface:
    Args:
        q: Query tensor (Q), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        k: Key tensor (K), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        v: Value tensor (V), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        cu_seqlen: Cumulative sequence lengths, shape [BSZ], with no leading zero.
        scale: Scaling factor for QK product
    Returns:
        o: Attention output tensor in fp32, shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM]
        lse: LogSumExp tensor, shape [TOTAL_SEQ, NUM_HEAD]
    """

    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16

    assert len(q.shape) == 3 and len(k.shape) == 3 and len(v.shape) == 3
    assert (
        q.shape[0] == k.shape[0]
        and k.shape[0] == v.shape[0]
        and q.shape[0] >= cu_seqlen[cu_seqlen.shape[0] - 1].cpu().item()
    )
    assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] in {64, 128}

    o = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.zeros((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32).T
    num_cores, _ = get_device_properties()

    if (not hasattr(dllm_attention_up_fwd_impl, "masks")):
        dllm_attention_up_fwd_impl.masks = {}
    if BLOCK_SIZE not in dllm_attention_up_fwd_impl.masks:
        BLOCK_MASK = 64
        offset_r_local = torch.arange(0, BLOCK_MASK)[:, None]
        offset_c_local = torch.arange(0, BLOCK_MASK)[None, :]
        chunk_idx_r = offset_r_local // BLOCK_SIZE
        chunk_idx_c = offset_c_local // BLOCK_SIZE
        mask_ul_i8 = packed_bool_to_i8((chunk_idx_r == chunk_idx_c))
        mask_ur_i8 = packed_bool_to_i8((chunk_idx_r > chunk_idx_c))
        dllm_attention_up_fwd_impl.masks[BLOCK_SIZE] = (mask_ul_i8.to(q.device), mask_ur_i8.to(q.device))
    mask_ul, mask_ur = dllm_attention_up_fwd_impl.masks[BLOCK_SIZE]

    kernel_da_fwd_u[(num_cores,)](
        q,
        k,
        v,
        o,
        lse,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ul,
        mask_ur,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        mask_ul.stride(0),
    )

    return o, lse


def dllm_attention_up_bwd_impl(
    fp32o: torch.Tensor,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float = 1.0,
    BLOCK_SIZE: int = 8,
):
    """
    Backward computation interface:
    Args:
        fp32o: Attention output tensor in fp32, shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM]
        do: Gradient tensor, shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        q: Query tensor (Q), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        k: Key tensor (K), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        v: Value tensor (V), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        lse: Logsumexp tensor, shape [TOTAL_SEQ, NUM_HEAD]
        cu_seqlen: Cumulative sequence lengths, shape [BSZ]
        scale: Scaling factor for QK product
    Returns:
        dq, dk, dv: Gradient tensors
    """

    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert do.dtype == torch.bfloat16 and fp32o.dtype == torch.float32

    assert len(q.shape) == 3 and len(k.shape) == 3 and len(v.shape) == 3
    assert (
        q.shape[0] == k.shape[0]
        and k.shape[0] == v.shape[0]
        and q.shape[0] >= cu_seqlen[cu_seqlen.shape[0] - 1].cpu().item()
    )
    assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] in {64, 128}
    assert q.shape[0] == lse.shape[0] and q.shape[1] == lse.shape[1]

    num_cores, num_vectorcore = get_device_properties()
    d = torch.empty(q.shape[1], q.shape[0], device=q.device, dtype=torch.float32).T
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if (not hasattr(dllm_attention_up_bwd_impl, "masks")):
        dllm_attention_up_bwd_impl.masks = {}
    if BLOCK_SIZE not in dllm_attention_up_bwd_impl.masks:
        BLOCK_MASK = 64
        offset_r_local = torch.arange(0, BLOCK_MASK)[:, None]
        offset_c_local = torch.arange(0, BLOCK_MASK)[None, :]
        chunk_idx_r = offset_r_local // BLOCK_SIZE
        chunk_idx_c = offset_c_local // BLOCK_SIZE
        mask_ul_i8 = packed_bool_to_i8((chunk_idx_r == chunk_idx_c))
        mask_ur_i8 = packed_bool_to_i8((chunk_idx_r > chunk_idx_c))
        dllm_attention_up_bwd_impl.masks[BLOCK_SIZE] = (mask_ul_i8.to(q.device), mask_ur_i8.to(q.device))
    mask_ul, mask_ur = dllm_attention_up_bwd_impl.masks[BLOCK_SIZE]

    kernel_da_bwd_d[(num_vectorcore,)](
        fp32o,
        do,
        d,
        fp32o.shape[0],
        fp32o.shape[1],
        fp32o.shape[2],
        fp32o.stride(0),
        fp32o.stride(1),
        fp32o.stride(2),
        d.stride(0),
        d.stride(1),
    )
    kernel_da_bwd_q_u[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dq,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ur,
        mask_ul,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        mask_ul.stride(0),
    )
    kernel_da_bwd_kv_ul[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dk,
        dv,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ul,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        mask_ul.stride(0),
    )
    kernel_da_bwd_kv_ur[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dk,
        dv,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ur,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        mask_ur.stride(0),
    )

    return dq, dk, dv


def dllm_attention_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float = 1.0,
    BLOCK_SIZE: int = 8,
):
    """
    Forward computation interface:
    Args:
        q: Query tensor (Q), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        k: Key tensor (K), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        v: Value tensor (V), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        cu_seqlen: Cumulative sequence lengths, shape [BSZ], with no leading zero.
        scale: Scaling factor for QK product
    Returns:
        o: Attention output tensor in fp32, shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM]
        lse: LogSumExp tensor, shape [TOTAL_SEQ, NUM_HEAD]
    """

    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16

    assert len(q.shape) == 3 and len(k.shape) == 3 and len(v.shape) == 3
    assert (
        q.shape[0] == k.shape[0]
        and k.shape[0] == v.shape[0]
        and q.shape[0] >= cu_seqlen[cu_seqlen.shape[0] - 1].cpu().item()
    )
    assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] in {64, 128}

    o = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.zeros((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32).T
    num_cores, _ = get_device_properties()

    if (not hasattr(dllm_attention_fwd_impl, "masks")):
        dllm_attention_fwd_impl.masks = {}
    if BLOCK_SIZE not in dllm_attention_fwd_impl.masks:
        BLOCK_MASK = 64
        offset_r_local = torch.arange(0, BLOCK_MASK)[:, None]
        offset_c_local = torch.arange(0, BLOCK_MASK)[None, :]
        chunk_idx_r = offset_r_local // BLOCK_SIZE
        chunk_idx_c = offset_c_local // BLOCK_SIZE
        mask_ul_i8 = packed_bool_to_i8((chunk_idx_r == chunk_idx_c))
        mask_ur_i8 = packed_bool_to_i8((chunk_idx_r > chunk_idx_c))
        mask_dr_i8 = packed_bool_to_i8((chunk_idx_r >= chunk_idx_c))
        mask_ul = mask_ul_i8.to(q.device)
        mask_ur = mask_ur_i8.to(q.device)
        mask_dr = mask_dr_i8.to(q.device)
        dllm_attention_fwd_impl.masks[BLOCK_SIZE] = (mask_ul, mask_ur, mask_dr)
    else:
        mask_ul, mask_ur, mask_dr = dllm_attention_fwd_impl.masks[BLOCK_SIZE]

    kernel_da_fwd_u[(num_cores,)](
        q,
        k,
        v,
        o,
        lse,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ul,
        mask_ur,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        mask_ul.stride(0),
    )
    kernel_da_fwd_d[(num_cores,)](
        q,
        k,
        v,
        o,
        lse,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_dr,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        mask_dr.stride(0),
    )

    return o, lse


def dllm_attention_bwd_impl(
    fp32o: torch.Tensor,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float = 1.0,
    BLOCK_SIZE: int = 8,
):
    """
    Backward computation interface:
    Args:
        fp32o: Attention output tensor in fp32, shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM]
        do: Gradient tensor, shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        q: Query tensor (Q), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        k: Key tensor (K), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        v: Value tensor (V), shape [TOTAL_SEQ, NUM_HEAD, HEAD_DIM], bf16
        lse: LogSumExp tensor, shape [TOTAL_SEQ, NUM_HEAD]
        cu_seqlen: Cumulative sequence lengths, shape [BSZ]
        scale: Scaling factor for QK product
    Returns:
        dq, dk, dv: Gradient tensors
    """

    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert do.dtype == torch.bfloat16 and fp32o.dtype == torch.float32

    assert len(q.shape) == 3 and len(k.shape) == 3 and len(v.shape) == 3
    assert (
        q.shape[0] == k.shape[0]
        and k.shape[0] == v.shape[0]
        and q.shape[0] >= cu_seqlen[cu_seqlen.shape[0] - 1].cpu().item()
    )
    assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] in {64, 128}
    assert q.shape[0] == lse.shape[0] and q.shape[1] == lse.shape[1]

    num_cores, num_vectorcore = get_device_properties()
    d = torch.empty(q.shape[1], q.shape[0], device=q.device, dtype=torch.float32).T
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if (not hasattr(dllm_attention_bwd_impl, "masks")):
        dllm_attention_bwd_impl.masks = {}
    if BLOCK_SIZE not in dllm_attention_bwd_impl.masks:
        BLOCK_MASK = 64
        offset_r_local = torch.arange(0, BLOCK_MASK)[:, None]
        offset_c_local = torch.arange(0, BLOCK_MASK)[None, :]
        chunk_idx_r = offset_r_local // BLOCK_SIZE
        chunk_idx_c = offset_c_local // BLOCK_SIZE
        mask_ul_i8 = packed_bool_to_i8((chunk_idx_r == chunk_idx_c))
        mask_ur_i8 = packed_bool_to_i8((chunk_idx_r > chunk_idx_c))
        mask_dr_i8 = packed_bool_to_i8((chunk_idx_r >= chunk_idx_c))
        mask_ul = mask_ul_i8.to(q.device)
        mask_ur = mask_ur_i8.to(q.device)
        mask_dr = mask_dr_i8.to(q.device)
        dllm_attention_bwd_impl.masks[BLOCK_SIZE] = (mask_ul, mask_ur, mask_dr)
    else:
        mask_ul, mask_ur, mask_dr = dllm_attention_bwd_impl.masks[BLOCK_SIZE]

    kernel_da_bwd_d[(num_vectorcore,)](
        fp32o,
        do,
        d,
        fp32o.shape[0],
        fp32o.shape[1],
        fp32o.shape[2],
        fp32o.stride(0),
        fp32o.stride(1),
        fp32o.stride(2),
        d.stride(0),
        d.stride(1),
    )
    kernel_da_bwd_q_u[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dq,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ur,
        mask_ul,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        mask_ul.stride(0),
    )
    kernel_da_bwd_q_d[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dq,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_dr,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        mask_dr.stride(0),
    )
    kernel_da_bwd_kv_ul[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dk,
        dv,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ul,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        mask_ul.stride(0),
    )
    kernel_da_bwd_kv_r[(num_cores,)](
        q,
        k,
        v,
        do,
        d,
        lse,
        dk,
        dv,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ur,
        mask_dr,
        q.shape[1] // k.shape[1],
        q.shape[0] // 2,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        d.stride(0),
        d.stride(1),
        mask_ur.stride(0),
    )

    return dq, dk, dv
