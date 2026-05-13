"""ILU: Over-Encoding decode for NF4-quantised embeddings (single fused kernel).

Entry point:
    over_encoding_decode_impl – decode path: input_ids [B, S] → [B, S, G, D]
"""

import torch
import triton
import triton.language as tl

from mojo_opset.core.operators.over_encoding import get_nf4_codebook

from ..utils import libentry


@libentry()
@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(
            min(int(args["embedding_dim"]), 256)
        ),
    }
)
@triton.jit
def _over_encoding_decode_kernel(
    out_ptr,               # [B*S*G, D] – output_dtype
    input_ids_ptr,         # [B, S] int64
    oe_history_ptr,        # [B, H] int64
    oe_vocab_sizes_ptr,    # [G] int64
    oe_vocab_offsets_ptr,  # [G] int64
    n_grams_ptr,           # [G] int64
    qweight_ptr,           # [V, D/2] int8, packed NF4
    scale_ptr,             # [V, D/GROUP_SIZE] float32
    mean_ptr,              # [V, D/GROUP_SIZE] float32
    codebook_ptr,          # [16] float16 – NF4 codebook
    B,
    S,
    G,
    H,
    ori_vocab_size,
    embedding_dim,
    qweight_stride_0,
    qweight_stride_1,
    scale_stride_0,
    scale_stride_1,
    mean_stride_0,
    mean_stride_1,
    out_stride_0,
    out_stride_1,
    vocab_start_id,
    vocab_size,
    MAX_GRAM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 2-D grid: axis 0 = flat (b,s,g) index, axis 1 = embedding-dim chunks.
    pid_bsg = tl.program_id(0).to(tl.int64)
    pid_d   = tl.program_id(1).to(tl.int64)

    # Decompose pid_bsg → (b, s, g)
    g         = pid_bsg % G
    token_idx = pid_bsg // G     # flat token in [0, B*S)
    b         = token_idx // S
    s         = token_idx % S

    # ----- N-gram index computation ----------------------------------------
    gram      = tl.load(n_grams_ptr + g)
    oe_vs     = tl.load(oe_vocab_sizes_ptr + g)
    oe_offset = tl.load(oe_vocab_offsets_ptr + g)

    result = tl.load(input_ids_ptr + b * S + s)
    carry  = tl.cast(ori_vocab_size, tl.int64)

    for i in range(1, MAX_GRAM):
        ctx_pos   = s - i                         # negative when s < i
        is_active = tl.cast(i, tl.int64) < gram
        use_ids   = ctx_pos >= tl.cast(0, tl.int64)

        safe_ids_pos = tl.where(use_ids, ctx_pos, tl.cast(0, tl.int64))
        ctx_from_ids = tl.load(input_ids_ptr + b * S + safe_ids_pos)

        hist_pos      = tl.cast(H, tl.int64) + ctx_pos   # = H - i + s
        safe_hist_pos = tl.where(use_ids, tl.cast(H - 1, tl.int64), hist_pos)
        ctx_from_hist = tl.load(oe_history_ptr + b * H + safe_hist_pos)

        ctx = tl.where(use_ids, ctx_from_ids, ctx_from_hist)

        new_result = (result + ctx * carry) % oe_vs
        result = tl.where(is_active, new_result, result)

        new_carry = carry * tl.cast(ori_vocab_size, tl.int64) % oe_vs
        carry = tl.where(is_active, new_carry, carry)

    vocab_id = result + oe_offset

    # ----- NF4 dequant embedding lookup ------------------------------------
    local_id = vocab_id - tl.cast(vocab_start_id, tl.int64)
    valid    = (local_id >= tl.cast(0, tl.int64)) & (local_id < tl.cast(vocab_size, tl.int64))
    safe_id  = tl.where(valid, local_id, tl.cast(0, tl.int64))

    # BLOCK_D output elements → BLOCK_D/2 packed bytes
    d_start    = pid_d * BLOCK_D
    half_start = d_start // 2

    half_offs = tl.arange(0, BLOCK_D // 2)
    half_idx  = half_start + half_offs
    half_mask = valid & (half_idx < (embedding_dim // 2))

    packed = tl.load(
        qweight_ptr + safe_id * qweight_stride_0 + half_idx * qweight_stride_1,
        mask=half_mask,
        other=0,
    ).to(tl.int32)

    # Unpack: even dim ← low nibble, odd dim ← high nibble
    low_nf4  = packed & 0x0F
    high_nf4 = (packed >> 4) & 0x0F

    low_out_idx  = d_start + half_offs * 2     # d_start, d_start+2, …
    high_out_idx = low_out_idx + 1              # d_start+1, d_start+3, …

    low_mask  = valid & (low_out_idx  < embedding_dim)
    high_mask = valid & (high_out_idx < embedding_dim)

    # NF4 codebook lookup → float32
    low_val  = tl.load(codebook_ptr + low_nf4,  mask=low_mask,  other=0.0).to(tl.float32)
    high_val = tl.load(codebook_ptr + high_nf4, mask=high_mask, other=0.0).to(tl.float32)

    # Per-group scale and mean
    low_group  = low_out_idx  // GROUP_SIZE
    high_group = high_out_idx // GROUP_SIZE

    low_scale = tl.load(
        scale_ptr + safe_id * scale_stride_0 + low_group * scale_stride_1,
        mask=low_mask, other=0.0,
    ).to(tl.float32)
    low_mean = tl.load(
        mean_ptr + safe_id * mean_stride_0 + low_group * mean_stride_1,
        mask=low_mask, other=0.0,
    ).to(tl.float32)
    high_scale = tl.load(
        scale_ptr + safe_id * scale_stride_0 + high_group * scale_stride_1,
        mask=high_mask, other=0.0,
    ).to(tl.float32)
    high_mean = tl.load(
        mean_ptr + safe_id * mean_stride_0 + high_group * mean_stride_1,
        mask=high_mask, other=0.0,
    ).to(tl.float32)

    # Dequantize: val * scale + mean
    low_out  = low_val  * low_scale  + low_mean
    high_out = high_val * high_scale + high_mean

    base = pid_bsg * out_stride_0
    tl.store(out_ptr + base + low_out_idx  * out_stride_1, low_out,  mask=low_mask)
    tl.store(out_ptr + base + high_out_idx * out_stride_1, high_out, mask=high_mask)


def over_encoding_decode_impl(
    input_ids: torch.Tensor,            # [B, S] int64
    oe_history_inputs: torch.Tensor,    # [B, H] int64
    oe_vocab_sizes: torch.Tensor,       # [G] int64
    oe_vocab_offsets: torch.Tensor,     # [G] int64
    n_grams: torch.Tensor,              # [G] int64
    LUT_qweight: torch.Tensor,          # [V, D/2] int8
    LUT_scale: torch.Tensor,            # [V, D/group_size] float32
    LUT_mean: torch.Tensor,             # [V, D/group_size] float32
    *,
    group_size: int = 1,
    codebook: torch.Tensor = None,
    ori_vocab_size: int = None,
    mega_vocab_start_id: int = 0,
    mega_vocab_size: int = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:                      # [B, S, G, D]
    """Fused over-encoding decode for NF4-quantised embedding tables (ILU).

    A single Triton kernel performs n-gram index computation and NF4
    dequantisation in one pass, eliminating the intermediate [B, S, G] index
    tensor and the second kernel launch overhead.

    Args:
        input_ids:           [B, S] int64 – input token IDs.
        oe_history_inputs:   [B, H] int64 – n-gram history (H = max_n_gram − 1).
        oe_vocab_sizes:      [G] int64 – over-encoding vocab size per gram slot.
        oe_vocab_offsets:    [G] int64 – cumulative vocab offsets per gram slot.
        n_grams:             [G] int64 – n-gram order per slot.
        LUT_qweight:         [V, D/2] int8 – NF4-packed embedding weights.
        LUT_scale:           [V, D/group_size] float32 – per-group scale.
        LUT_mean:            [V, D/group_size] float32 – per-group mean.
        group_size:          Quantisation group size.
        codebook:            16-entry float16 NF4 codebook; built if None.
        ori_vocab_size:      Original (non-OE) vocabulary size.
        mega_vocab_start_id: First valid vocab row in the embedding table.
        mega_vocab_size:     Number of valid vocab rows; defaults to LUT_qweight.size(0).
        output_dtype:        dtype of the returned tensor.

    Returns:
        Tensor of shape [B, S, G, D] with dtype output_dtype.
    """
    assert input_ids.dim() == 2, (
        f"over_encoding_decode_impl expects 2-D input_ids, got {input_ids.shape}"
    )

    B, S        = input_ids.shape
    G           = int(n_grams.size(0))
    H           = int(oe_history_inputs.size(1))
    embedding_dim = int(LUT_scale.size(1)) * group_size
    vocab_size  = LUT_qweight.size(0) if mega_vocab_size is None else mega_vocab_size

    if codebook is None:
        codebook = get_nf4_codebook(device=LUT_qweight.device, dtype=torch.float16)
    else:
        codebook = codebook.to(device=LUT_qweight.device, dtype=torch.float16)

    input_ids         = input_ids.contiguous()
    oe_history_inputs = oe_history_inputs.contiguous()
    oe_vocab_sizes    = oe_vocab_sizes.contiguous()
    oe_vocab_offsets  = oe_vocab_offsets.contiguous()
    n_grams           = n_grams.contiguous()
    LUT_qweight       = LUT_qweight.contiguous()
    LUT_scale         = LUT_scale.contiguous()
    LUT_mean          = LUT_mean.contiguous()

    N_flat = B * S * G
    # Output initialised to zeros so that out-of-range vocab IDs produce zeros.
    output = torch.zeros(N_flat, embedding_dim, dtype=output_dtype, device=input_ids.device)

    if N_flat == 0:
        return output.view(B, S, G, embedding_dim)

    block_d = triton.next_power_of_2(min(embedding_dim, 256))
    grid    = (N_flat, triton.cdiv(embedding_dim, block_d))

    _over_encoding_decode_kernel[grid](
        output,
        input_ids, oe_history_inputs,
        oe_vocab_sizes, oe_vocab_offsets, n_grams,
        LUT_qweight, LUT_scale, LUT_mean, codebook,
        B=B, S=S, G=G, H=H,
        ori_vocab_size=ori_vocab_size,
        embedding_dim=embedding_dim,
        qweight_stride_0=LUT_qweight.stride(0),
        qweight_stride_1=LUT_qweight.stride(1),
        scale_stride_0=LUT_scale.stride(0),
        scale_stride_1=LUT_scale.stride(1),
        mean_stride_0=LUT_mean.stride(0),
        mean_stride_1=LUT_mean.stride(1),
        out_stride_0=output.stride(0),
        out_stride_1=output.stride(1),
        vocab_start_id=mega_vocab_start_id,
        vocab_size=vocab_size,
        MAX_GRAM=H+1,
        GROUP_SIZE=group_size,
    )

    return output.view(B, S, G, embedding_dim)
