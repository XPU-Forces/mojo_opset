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
  n_gram_decode_impl  – batched decode:  input_ids [B, S]   → [B, S, G]
  n_gram_prefill_impl – packed prefill:  input_ids [T]      → [T, G]
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
    out_ptr,               # [B, S, G] int64, row-major – stride (S*G, G, 1)
    B,
    S,
    G,
    H,
    ori_vocab_size,
    MAX_GRAM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    g          = pid % G
    token_idx  = pid // G          # flat token index in [0, B*S)
    b          = token_idx // S
    s          = token_idx % S

    gram      = tl.load(n_grams_ptr + g)
    oe_vs     = tl.load(oe_vocab_sizes_ptr + g)
    oe_offset = tl.load(oe_vocab_offsets_ptr + g)

    result = tl.load(input_ids_ptr + b * S + s)
    carry  = tl.cast(ori_vocab_size, tl.int64)

    for i in range(1, MAX_GRAM):
        ctx_pos   = s - i                        # tl.int64; negative when s < i
        is_active = tl.cast(i, tl.int64) < gram  # whether this gram step applies
        use_ids   = ctx_pos >= tl.cast(0, tl.int64)

        # --- load context from input_ids (safe index: clamp to 0 when out-of-range) ---
        safe_ids_pos = tl.where(use_ids, ctx_pos, tl.cast(0, tl.int64))
        ctx_from_ids = tl.load(input_ids_ptr + b * S + safe_ids_pos)

        # --- load context from history (valid only when use_ids is False) ---
        hist_pos      = tl.cast(H, tl.int64) + ctx_pos  # = H - i + s; in [0,H) iff s < i
        safe_hist_pos = tl.where(use_ids, tl.cast(H - 1, tl.int64), hist_pos)
        ctx_from_hist = tl.load(oe_history_ptr + b * H + safe_hist_pos)

        ctx = tl.where(use_ids, ctx_from_ids, ctx_from_hist)

        new_result = (result + ctx * carry) % oe_vs
        result = tl.where(is_active, new_result, result)

        new_carry = carry * tl.cast(ori_vocab_size, tl.int64) % oe_vs
        carry = tl.where(is_active, new_carry, carry)

    tl.store(out_ptr + token_idx * G + g, result + oe_offset)


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

    max_gram = int(n_grams.max().item())

    out = torch.empty(B, S, G, dtype=torch.int64, device=input_ids.device)
    if B == 0 or S == 0 or G == 0:
        return out

    grid = (B * S * G,)
    _n_gram_decode_kernel[grid](
        input_ids, oe_history_inputs,
        oe_vocab_sizes, oe_vocab_offsets, n_grams,
        out,
        B=B, S=S, G=G, H=H,
        ori_vocab_size=vocab_size,
        MAX_GRAM=max_gram,
    )
    return out


# ---------------------------------------------------------------------------
# Prefill kernel  (input_ids [T], packed / variable-length sequences)
# ---------------------------------------------------------------------------

@libentry()
@triton.jit
def _n_gram_prefill_kernel(
    input_ids_ptr,         # [T] int64 – all sequences packed flat
    batch_ids_ptr,         # [T] int64 – which batch sequence each token belongs to
    seq_positions_ptr,     # [T] int64 – position of each token within its sequence
    oe_history_ptr,        # [B, H] int64, row-major
    oe_vocab_sizes_ptr,    # [G] int64
    oe_vocab_offsets_ptr,  # [G] int64
    n_grams_ptr,           # [G] int64
    out_ptr,               # [T, G] int64, row-major
    T,
    G,
    H,
    ori_vocab_size,
    MAX_GRAM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    g = pid % G
    t = pid // G           # flat token index in [0, T)

    gram      = tl.load(n_grams_ptr + g)
    oe_vs     = tl.load(oe_vocab_sizes_ptr + g)
    oe_offset = tl.load(oe_vocab_offsets_ptr + g)

    b = tl.load(batch_ids_ptr + t)     # batch index for token t
    s = tl.load(seq_positions_ptr + t)  # position within its sequence

    result = tl.load(input_ids_ptr + t)
    carry  = tl.cast(ori_vocab_size, tl.int64)

    for i in range(1, MAX_GRAM):
        ctx_pos   = s - i
        is_active = tl.cast(i, tl.int64) < gram
        use_ids   = ctx_pos >= tl.cast(0, tl.int64)

        # --- load context from packed input_ids ---
        # When seq_pos >= i the context token is at flat index t - i (same sequence).
        safe_t_minus_i = tl.where(use_ids, t - i, tl.cast(0, tl.int64))
        ctx_from_ids   = tl.load(input_ids_ptr + safe_t_minus_i)

        # --- load context from history ---
        hist_pos      = tl.cast(H, tl.int64) + ctx_pos   # H - i + s; in [0,H) iff s < i
        safe_hist_pos = tl.where(use_ids, tl.cast(H - 1, tl.int64), hist_pos)
        ctx_from_hist = tl.load(oe_history_ptr + b * H + safe_hist_pos)

        ctx = tl.where(use_ids, ctx_from_ids, ctx_from_hist)

        new_result = (result + ctx * carry) % oe_vs
        result = tl.where(is_active, new_result, result)

        new_carry = carry * tl.cast(ori_vocab_size, tl.int64) % oe_vs
        carry = tl.where(is_active, new_carry, carry)

    tl.store(out_ptr + t * G + g, result + oe_offset)


@input_guard(make_contiguous=True, auto_to_device=True)
def n_gram_prefill_impl(
    input_ids: torch.Tensor,          # [T] int64 – packed tokens from all sequences
    seq_lens: torch.Tensor,           # [B] int64
    oe_history_inputs: torch.Tensor,  # [B, H] int64
    oe_vocab_sizes: torch.Tensor,     # [G] int64
    oe_vocab_offsets: torch.Tensor,   # [G] int64
    n_grams: torch.Tensor,            # [G] int64
    vocab_size: int,
) -> torch.Tensor:                    # [T, G] int64
    assert input_ids.dim() == 1, f"n_gram_prefill expects 1-D input_ids, got {input_ids.shape}"
    T = int(input_ids.size(0))
    B = int(seq_lens.size(0))
    G = int(n_grams.size(0))
    H = int(oe_history_inputs.size(1))

    max_gram = int(n_grams.max().item())

    out = torch.empty(T, G, dtype=torch.int64, device=input_ids.device)
    if T == 0 or G == 0:
        return out

    device = input_ids.device

    # Precompute per-token metadata on the same device so the kernel
    # can look them up with a single scalar load per program.
    seq_lens_i64 = seq_lens.to(dtype=torch.int64, device=device)
    batch_ids = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int64, device=device),
        seq_lens_i64,
    )  # [T]
    cu_seq_lens = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_seq_lens[1:] = seq_lens_i64.cumsum(0)
    seq_positions = (
        torch.arange(T, dtype=torch.int64, device=device) - cu_seq_lens[batch_ids]
    )  # [T]

    grid = (T * G,)
    _n_gram_prefill_kernel[grid](
        input_ids, batch_ids, seq_positions,
        oe_history_inputs,
        oe_vocab_sizes, oe_vocab_offsets, n_grams,
        out,
        T=T, G=G, H=H,
        ori_vocab_size=vocab_size,
        MAX_GRAM=max_gram,
    )
    return out
