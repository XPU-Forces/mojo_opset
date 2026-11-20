"""Fused ``sigmoid(x @ W_gate^T) * shared_out`` for A5 Triton.

The gate is a GEMV (N=1).  Padding the weight to 16 columns lets the dot
product run on the cube unit as ``[BLOCK_M, BLOCK_K] @ [BLOCK_K, 16]`` with
only column 0 kept, so ``MatMul + Sigmoid + Mul`` collapse into one kernel
with no ``[tokens, 1]`` intermediate.  The second pass (sigmoid *
shared_out) stays on the vector pipe.

Tiling (measured on Ascend950PR, bf16/fp16, cold cache):

* ``BLOCK_K=256`` is the interior optimum; 128 and 512 are slower everywhere.
* ``BLOCK_M`` is shape-dependent and chosen on cold-cache timings (L2 flushed
  before each launch): 32 at 4096 tokens with d >= 5120, 64 otherwise at
  4096/8192 tokens, 128 at 8192 tokens with d <= 2048 and at >= 16384 tokens.
  256 is worse everywhere.  A ``@triton.autotune`` over 32/64/128 was rejected
  because its ``do_bench`` picks flipped between runs (8192x2048 chose 64 once
  and 128 the next time, a 2x spread), so the tile is derived deterministically
  from the shape.
* ``num_stages=2`` enables the Ascend multibuffer (MTE2 / vector overlap).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_expert_gate_kernel(
    X, WP, S, OUT, NROWS, D,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program owns ``BLOCK_M`` tokens.  ``WP`` is the gate weight padded to
    # ``BLOCK_N`` columns (only column 0 is meaningful), so the score is a real
    # matrix product on the cube unit.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < NROWS
    n_idx = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        kmask = ks < D
        x = tl.load(
            X + rows[:, None] * D + ks[None, :],
            mask=rmask[:, None] & kmask[None, :], other=0.0,
        )
        wp = tl.load(
            WP + ks[:, None] * BLOCK_N + n_idx[None, :],
            mask=kmask[:, None], other=0.0,
        )
        # fp32 must not fall back to TF32 on CUDA (4.5e-2 error); the flag is
        # ignored for bf16/fp16 inputs.
        acc = tl.dot(x, wp, acc, input_precision="ieee")

    score = tl.sum(tl.where(n_idx[None, :] == 0, acc, 0.0), axis=1)
    g = 1.0 / (1.0 + tl.exp(-score))

    for k0 in range(0, D, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask = rmask[:, None] & (ks[None, :] < D)
        s = tl.load(S + rows[:, None] * D + ks[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        y = g[:, None] * s
        tl.store(OUT + rows[:, None] * D + ks[None, :],
                 y.to(OUT.dtype.element_ty), mask=mask)


def shared_expert_gate_fwd_impl(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    shape = shared_out.shape
    x2 = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    s2 = shared_out.reshape(-1, shared_out.shape[-1]).contiguous()
    w = gate_weight.reshape(-1).contiguous()
    n, d = x2.shape
    out = torch.empty_like(s2)
    # The cube needs at least 16 columns; only column 0 carries the weight.
    wp = torch.zeros(d, 16, device=x2.device, dtype=x2.dtype)
    wp[:, 0] = w
    # fp32 doubles the UB footprint of a tile, so halve the K tile for it.
    block_k = 128 if x2.dtype == torch.float32 else 256
    # Cold-cache (L2 flushed before every launch) scan over BLOCK_M =
    # 32/64/128/256, same process, both dtypes:
    #   4096 tokens : d >= 5120 -> 32, else 64
    #   8192 tokens : d <= 2048 -> 128, else 64
    #   >= 16384    : 128
    if n >= 16384 or (n >= 8192 and d <= 2048):
        block_m = 128
    elif n <= 4096 and d >= 5120:
        block_m = 32
    else:
        block_m = 64
    grid = (triton.cdiv(n, block_m),)
    _shared_expert_gate_kernel[grid](
        x2, wp, s2, out, n, d,
        BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_N=16, num_stages=2,
    )
    return out.reshape(shape)


@torch.library.custom_op("mojo_npu_triton_a5::shared_expert_gate_fwd",
                         mutates_args=())
def shared_expert_gate_fwd(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    return shared_expert_gate_fwd_impl(hidden_states, gate_weight, shared_out)


@shared_expert_gate_fwd.register_fake
def _shared_expert_gate_fwd_fake(hidden_states, gate_weight, shared_out):
    return torch.empty_like(shared_out, memory_format=torch.contiguous_format)
