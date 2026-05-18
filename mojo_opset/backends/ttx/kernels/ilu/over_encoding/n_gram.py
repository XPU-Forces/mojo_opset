"""ILU Triton: N-gram index computation for Over-Encoding.

For each output position (token, gram_slot g):
    result = input_ids[token]
    carry  = ori_vocab_size
    for i in 1 .. gram[g]-1:
        if seq_pos >= i:
            ctx = input_ids[token - i]          # same sequence, earlier position
        else:
            ctx = oe_history[batch, H - i + seq_pos]   # from history buffer
        result = (result + ctx * carry) % oe_vocab_sizes[g]
        carry  = carry * ori_vocab_size % oe_vocab_sizes[g]
    result += oe_vocab_offsets[g]

Two entry points are provided:
  n_gram_decode_impl  – batched decode:  input_ids [B, S]   -> [B, S, G]
  n_gram_prefill_impl – packed prefill:  input_ids [T]      -> [T, G]
                          (T = sum of per-sequence lengths)
"""

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.utils import input_guard

from ..utils import libentry


# ---------------------------------------------------------------------------
# Decode kernel  (input_ids [B, S], oe_history [B, H])
# ---------------------------------------------------------------------------

@libentry()
@triton.jit
def _n_gram_decode_kernel(
    input_ids_ptr,         # [B, S] int64, row-major
    oe_history_ptr,        # [B, H] int64, row-major
    oe_vocab_sizes_ptr,    # [G] int64
    oe_vocab_offsets_ptr,  # [G] int64
    n_grams_ptr,           # [G] int64
    out_ptr,               # [B, S, G] int64, row-major - stride (S*G, G, 1)
    B,
    S,
    G,
    H,
    ori_vocab_size,
    MAX_GRAM: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    b = pid // S
    s = pid % S

    g_offsets = tl.arange(0, BLOCK_G).to(tl.int64)
    g_mask = g_offsets < tl.cast(G, tl.int64)

    gram_vals  = tl.load(n_grams_ptr + g_offsets, mask=g_mask, other=1)
    oe_vs      = tl.load(oe_vocab_sizes_ptr + g_offsets, mask=g_mask, other=1)
    oe_offsets = tl.load(oe_vocab_offsets_ptr + g_offsets, mask=g_mask, other=0)

    input_id = tl.load(input_ids_ptr + b * S + s)

    c_vocab = tl.cast(ori_vocab_size, tl.int64)
    result = input_id + tl.zeros([BLOCK_G], dtype=tl.int64)
    carry  = c_vocab + tl.zeros([BLOCK_G], dtype=tl.int64)

    c_zero = tl.cast(0, tl.int64)
    c_h_minus_1 = tl.cast(H - 1, tl.int64)
    c_H = tl.cast(H, tl.int64)

    for i in range(1, MAX_GRAM):
        ctx_pos = s - tl.cast(i, tl.int64)
        use_ids = ctx_pos >= c_zero

        safe_ids_pos = tl.where(use_ids, ctx_pos, c_zero)
        ctx_from_ids = tl.load(input_ids_ptr + b * S + safe_ids_pos)

        hist_pos = c_H + ctx_pos
        safe_hist_pos = tl.where(use_ids, c_h_minus_1, hist_pos)
        ctx_from_hist = tl.load(oe_history_ptr + b * H + safe_hist_pos)

        ctx = tl.where(use_ids, ctx_from_ids, ctx_from_hist)

        is_active = tl.cast(i, tl.int64) < gram_vals

        new_result = (result + ctx * carry) % oe_vs
        result = tl.where(is_active, new_result, result)

        new_carry = carry * c_vocab % oe_vs
        carry = tl.where(is_active, new_carry, carry)

    out_base = pid * tl.cast(G, tl.int64)
    tl.store(out_ptr + out_base + g_offsets, result + oe_offsets, mask=g_mask)


@input_guard(make_contiguous=True, auto_to_device=True)
def n_gram_decode_impl(
    input_ids: torch.Tensor,          # [B, S] int64
    oe_history_inputs: torch.Tensor,  # [B, H] int64
    oe_vocab_sizes: torch.Tensor,     # [G] int64
    oe_vocab_offsets: torch.Tensor,   # [G] int64
    n_grams: torch.Tensor,            # [G] int64
    vocab_size: int,
) -> torch.Tensor:                    # [B, S, G] int64
    assert input_ids.dim() == 2, f"n_gram_decode expects 2-D input_ids, got {input_ids.shape}"
    B, S = input_ids.shape
    G    = int(n_grams.size(0))
    H    = int(oe_history_inputs.size(1))

    block_g = 1
    while block_g < G:
        block_g <<= 1

    out = torch.empty(B, S, G, dtype=torch.int64, device=input_ids.device)
    if B == 0 or S == 0 or G == 0:
        return out

    grid = (B * S,)
    _n_gram_decode_kernel[grid](
        input_ids, oe_history_inputs,
        oe_vocab_sizes, oe_vocab_offsets, n_grams,
        out,
        B=B, S=S, G=G, H=H,
        ori_vocab_size=vocab_size,
        MAX_GRAM=H+1,
        BLOCK_G=block_g,
    )
    return out


# ---------------------------------------------------------------------------
# Prefill kernel  (input_ids [T], packed / variable-length sequences)
# ---------------------------------------------------------------------------

@libentry()
@triton.jit
def _n_gram_prefill_kernel(
    input_ids_ptr,         # [T] int64 - all sequences packed flat
    cu_seq_lens_ptr,       # [B+1] int64 - cumulative sequence lengths, cu_seq_lens[0]=0
    oe_history_ptr,        # [B, H] int64, row-major
    oe_vocab_sizes_ptr,    # [G] int64
    oe_vocab_offsets_ptr,  # [G] int64
    n_grams_ptr,           # [G] int64
    out_ptr,               # [T, G] int64, row-major
    T,
    G,
    H,
    B,
    ori_vocab_size,
    MAX_GRAM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    LOG2B: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t = pid

    g_offsets = tl.arange(0, BLOCK_G).to(tl.int64)
    g_mask = g_offsets < tl.cast(G, tl.int64)

    gram_vals  = tl.load(n_grams_ptr + g_offsets, mask=g_mask, other=1)
    oe_vs      = tl.load(oe_vocab_sizes_ptr + g_offsets, mask=g_mask, other=1)
    oe_offsets = tl.load(oe_vocab_offsets_ptr + g_offsets, mask=g_mask, other=0)

    c_zero = tl.cast(0, tl.int64)
    lo = c_zero
    hi = tl.cast(B, tl.int64)
    for _ in range(LOG2B):
        mid = (lo + hi) >> 1
        val = tl.load(cu_seq_lens_ptr + mid)
        lo = tl.where(val <= t, mid + 1, lo)
        hi = tl.where(val <= t, hi, mid)
    b = lo - 1
    s = t - tl.load(cu_seq_lens_ptr + b)

    input_id = tl.load(input_ids_ptr + t)

    c_vocab = tl.cast(ori_vocab_size, tl.int64)
    result = input_id + tl.zeros([BLOCK_G], dtype=tl.int64)
    carry  = c_vocab + tl.zeros([BLOCK_G], dtype=tl.int64)

    c_h_minus_1 = tl.cast(H - 1, tl.int64)
    c_H = tl.cast(H, tl.int64)

    for i in range(1, MAX_GRAM):
        ctx_pos = s - tl.cast(i, tl.int64)
        use_ids = ctx_pos >= c_zero

        safe_t_minus_i = tl.where(use_ids, t - tl.cast(i, tl.int64), c_zero)
        ctx_from_ids   = tl.load(input_ids_ptr + safe_t_minus_i)

        hist_pos      = c_H + ctx_pos
        safe_hist_pos = tl.where(use_ids, c_h_minus_1, hist_pos)
        ctx_from_hist = tl.load(oe_history_ptr + b * H + safe_hist_pos)

        ctx = tl.where(use_ids, ctx_from_ids, ctx_from_hist)

        is_active = tl.cast(i, tl.int64) < gram_vals

        new_result = (result + ctx * carry) % oe_vs
        result = tl.where(is_active, new_result, result)

        new_carry = carry * c_vocab % oe_vs
        carry = tl.where(is_active, new_carry, carry)

    out_base = t * tl.cast(G, tl.int64)
    tl.store(out_ptr + out_base + g_offsets, result + oe_offsets, mask=g_mask)


@input_guard(make_contiguous=True, auto_to_device=True)
def n_gram_prefill_impl(
    input_ids: torch.Tensor,          # [T] int64 – packed tokens from all sequences
    q_lens: torch.Tensor,             # [B] int64
    oe_history_inputs: torch.Tensor,  # [B, H] int64
    oe_vocab_sizes: torch.Tensor,     # [G] int64
    oe_vocab_offsets: torch.Tensor,   # [G] int64
    n_grams: torch.Tensor,            # [G] int64
    vocab_size: int,
) -> torch.Tensor:                    # [T, G] int64
    assert input_ids.dim() == 1, f"n_gram_prefill expects 1-D input_ids, got {input_ids.shape}"
    T = int(input_ids.size(0))
    B = int(q_lens.size(0))
    G = int(n_grams.size(0))
    H = int(oe_history_inputs.size(1))

    block_g = 1
    while block_g < G:
        block_g <<= 1

    out = torch.empty(T, G, dtype=torch.int64, device=input_ids.device)
    if T == 0 or G == 0:
        return out

    device = input_ids.device

    # Build cumulative sequence lengths for the kernel's binary search.
    q_lens_i64 = q_lens.to(dtype=torch.int64, device=device)
    cu_seq_lens = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_seq_lens[1:] = q_lens_i64.cumsum(0)

    grid = (T,)
    _n_gram_prefill_kernel[grid](
        input_ids, cu_seq_lens,
        oe_history_inputs,
        oe_vocab_sizes, oe_vocab_offsets, n_grams,
        out,
        T=T, G=G, H=H, B=B,
        ori_vocab_size=vocab_size,
        MAX_GRAM=H+1,
        BLOCK_G=block_g,
        LOG2B=B.bit_length(),
    )
    return out
