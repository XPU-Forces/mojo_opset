from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def micro_kernel_fwd(
    block_q,
    k,
    v,
    block_o,
    block_m,
    block_l,
    scale,
    offset_c,
    offset_c_ed,
    block_mask,
    idx_n,
    offs_h,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    boundary_mask=None,
):
    tl.static_assert(STRIDE_K_H == 1)
    tl.static_assert(STRIDE_V_H == 1)
    ptr_k = (
        k
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + offs_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + offs_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < offset_c_ed
    block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
    block_k = tl.trans(block_k)
    block_s = tl.dot(block_q, block_k) * scale
    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    if boundary_mask is not None:
        block_s += ((boundary_mask.to(tl.float32) - 1.0) * 1e6)
    if block_mask is not None:
        block_s = tl.where(block_mask, block_s, -1.0e6)
        tl.compile_hint(block_s, "bitwise_mask")
    block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
    block_s = tl.exp(block_s - block_m_1[:, None])
    block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)
    block_o = tl.exp(block_m - block_m_1)[:, None] * block_o
    block_o = block_o + tl.dot(block_s.to(tl.bfloat16), block_v)

    return block_o, block_m_1, block_l_1


@triton.jit
def micro_kernel_bwd_q(
    block_q,
    k,
    v,
    block_do,
    block_d,
    block_dq,
    block_lse,
    scale,
    offset_c,
    offset_c_ed,
    block_mask,
    idx_n,
    offs_h,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    boundary_mask=None,
):
    tl.static_assert(STRIDE_K_H == 1)
    tl.static_assert(STRIDE_V_H == 1)
    ptr_k = (
        k
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + offs_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + offs_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < offset_c_ed
    block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
    block_s = tl.dot(block_q, block_k.T) * scale
    if boundary_mask is not None:
        block_s += ((boundary_mask.to(tl.float32) - 1.0) * 1e6)
    if block_mask is not None:
        block_s = tl.where(block_mask, block_s, -1.0e6)
        tl.compile_hint(block_s, "bitwise_mask")
    block_p = tl.exp(block_s - block_lse[:, None])
    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    block_dp = tl.dot(block_do, block_v.T)
    block_ds = block_p * (block_dp - block_d[:, None])
    block_dq += tl.dot(block_ds.to(tl.bfloat16), block_k) * scale
    return block_dq


@triton.jit
def micro_kernel_bwd_kv(
    q,
    block_k,
    block_v,
    do,
    d,
    block_dk,
    block_dv,
    lse,
    scale,
    offset_r,
    offset_r_ed,
    block_mask,
    idx_n,
    offs_h,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N,
    BLOCK_R: tl.constexpr,
    boundary_mask=None,
):
    tl.static_assert(STRIDE_D_S == 1)
    tl.static_assert(STRIDE_Q_H == 1)
    ptr_q = (
        q + idx_n * STRIDE_Q_N + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
    )
    ptr_do = (
        do + idx_n * STRIDE_Q_N + (offset_r + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + offs_h[None, :] * STRIDE_Q_H
    )
    ptr_d = d + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
    ptr_lse = lse + idx_n * STRIDE_D_N + (offset_r + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S

    mask_q = (offset_r + tl.arange(0, BLOCK_R))[:, None] < offset_r_ed
    mask_d = (offset_r + tl.arange(0, BLOCK_R))[:] < offset_r_ed

    block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
    block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
    block_s = tl.dot(block_q, block_k) * scale
    if boundary_mask is not None:
        block_s += ((boundary_mask.to(tl.float32) - 1.0) * 1e6)
    if block_mask is not None:
        block_s = tl.where(block_mask, block_s, -1.0e6)
        tl.compile_hint(block_s, "bitwise_mask")
    block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
    block_p = tl.exp(block_s - block_lse[:, None])
    block_dv += tl.dot(block_p.to(tl.bfloat16).T, block_do)
    block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
    block_dp = tl.dot(block_do, block_v)
    block_ds = block_p * (block_dp - block_d[:, None])
    block_dk += tl.dot(block_ds.to(tl.bfloat16).T, block_q) * scale

    return block_dk, block_dv


def packed_bool_to_i8(
        bool_mask: torch.Tensor,
        block_num: Optional[int] = 1
) -> torch.Tensor:
    orig_shape = bool_mask.shape
    assert orig_shape[-1] % 8 == 0

    assert (bool_mask.numel() // orig_shape[-1]) % block_num == 0, f"{bool_mask.numel() = } {orig_shape=}"
    bool_mask_blocked = bool_mask.reshape(block_num, -1, orig_shape[-1])

    result = []
    for i in range(block_num):
        flat_mask = bool_mask_blocked[i].flatten()

        flat_len = flat_mask.numel()
        num_packed = flat_len // 8
        mask_8group = flat_mask.reshape(num_packed, 8)

        weights = torch.tensor(
            [1 << i for i in range(0, 8)],
            dtype=torch.uint8,
            device=bool_mask.device
        )
        packed_vals = (mask_8group.to(torch.uint8) * weights).sum(dim=-1, dtype=torch.uint8)

        padded_flat = torch.cat([
            packed_vals,
            torch.zeros(flat_len - num_packed, dtype=torch.uint8, device=bool_mask.device)
        ], dim=0)
        result.append(padded_flat)
    padded_flat = torch.cat(result, dim=0)

    return padded_flat.reshape(orig_shape)
