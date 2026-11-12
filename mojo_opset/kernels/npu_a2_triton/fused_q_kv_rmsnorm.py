"""Ascend A2 (910B) Triton fused Q/KV RMSNorm.

Structure relative to the vllm original (grid-y task axis, one shared
``BLOCK=next_pow2(max(q_size, kv_size))`` for both halves):

1. Q and KV are launched as two kernels with their own ``BLOCK``, so each side
   runs at its own width and ``SIZE`` stays a compile-time constant.
2. ``RPP`` rows are merged per program, tile ``[RPP, BLOCK]``; the norm
   reduction runs along the tile's column axis and yields ``RPP`` inverse RMS
   values at once.
3. Below ``_SPLIT_MIN_ROWS`` rows the two launches cost more than the structure
   saves, so the original single-kernel layout is used instead.
4. An empty batch returns empty outputs without launching.

The split crossover and ``RPP=8`` come from a cold-cache calibration on this
arch; ``RPP=16`` exceeds the per-core UB budget ("ub overflow").

A2 has no FP8 (the DAV_2201 platform ini carries no ``DT_FLOAT8_*`` and Triton's
``f32 -> fp8`` cast aborts in ``arith::TruncFOp`` lowering), so
``fused_q_kv_rmsnorm_quant`` is not registered for this arch; only the norm
entry is provided.
"""

import torch
import triton
import triton.language as tl

_RPP = 8
_SPLIT_MIN_ROWS = 64
_NUM_WARPS = 8


@triton.jit
def _fused_q_kv_rmsnorm_native(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_rows(
    x_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    in_stride,
    out_stride,
    eps,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    RPP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RPP + tl.arange(0, RPP)
    row_mask = rows < num_rows
    cols = tl.arange(0, BLOCK)
    col_mask = cols < SIZE
    m = row_mask[:, None] & col_mask[None, :]
    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + rows[:, None] * in_stride + cols[None, :], mask=m, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / SIZE
    rrms = tl.rsqrt(variance + eps)
    y = x * rrms[:, None] * w[None, :]
    tl.store(
        out_ptr + rows[:, None] * out_stride + cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=m,
    )


def _fused_q_kv_rmsnorm_impl(qr, kv, q_weight, kv_weight, eps):
    q_out = torch.empty(qr.shape, dtype=qr.dtype, device=qr.device)
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    num_tokens = qr.shape[0]
    if num_tokens == 0:
        return q_out, kv_out
    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    if num_tokens < _SPLIT_MIN_ROWS:
        block_size = triton.next_power_of_2(max(q_size, kv_size))
        _fused_q_kv_rmsnorm_native[(num_tokens, 2)](
            qr, q_out, q_weight, qr.stride(0), q_out.stride(0),
            kv, kv_out, kv_weight, kv.stride(0), kv_out.stride(0),
            eps,
            Q_SIZE=q_size, KV_SIZE=kv_size, BLOCK_SIZE=block_size,
            num_warps=8 if block_size >= 2048 else 4,
        )
        return q_out, kv_out
    _rmsnorm_rows[(triton.cdiv(num_tokens, _RPP),)](
        qr, q_weight, q_out, num_tokens, qr.stride(0), q_out.stride(0), eps,
        SIZE=q_size, BLOCK=triton.next_power_of_2(q_size), RPP=_RPP, num_warps=_NUM_WARPS,
    )
    _rmsnorm_rows[(triton.cdiv(num_tokens, _RPP),)](
        kv, kv_weight, kv_out, num_tokens, kv.stride(0), kv_out.stride(0), eps,
        SIZE=kv_size, BLOCK=triton.next_power_of_2(kv_size), RPP=_RPP, num_warps=_NUM_WARPS,
    )
    return q_out, kv_out


@torch.library.custom_op("mojo_npu_triton_a2::fused_q_kv_rmsnorm_fwd", mutates_args=())
def fused_q_kv_rmsnorm_fwd(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fused_q_kv_rmsnorm_impl(
        qr.contiguous(), kv.contiguous(), q_weight, kv_weight, eps
    )


@fused_q_kv_rmsnorm_fwd.register_fake
def _fused_q_kv_rmsnorm_fake(qr, kv, q_weight, kv_weight, eps):
    return torch.empty_like(qr, memory_format=torch.contiguous_format), torch.empty_like(
        kv, memory_format=torch.contiguous_format
    )
