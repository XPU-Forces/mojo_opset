"""ILU Triton: NF4-dequantization fused Embedding lookup.

For each input token index t with vocab id = input[t]:
    local_id = vocab_id - vocab_start_id
    if 0 <= local_id < vocab_size:
        for d in range(embedding_dim):
            packed_byte = LUT_qweight[local_id, d // 2]
            nf4_idx  = packed_byte & 0x0F  if d is even
                     = (packed_byte >> 4) & 0x0F  if d is odd
            group    = d // GROUP_SIZE
            out[t,d] = codebook[nf4_idx] * LUT_scale[local_id, group]
                     + LUT_mean[local_id, group]
    else:
        out[t, :] = 0

Packing convention (matches pack_nf4_uint4_to_int8 in the test helper):
    packed_byte[j] = q_idx[2*j] | (q_idx[2*j+1] << 4)
    → even output dim 2*j  ← low  nibble (& 0x0F)
    → odd  output dim 2*j+1 ← high nibble (>> 4 & 0x0F)
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
def _embedding_nf4_dequant_kernel(
    out_ptr,           # [N, D] – typed as output_dtype
    input_ptr,         # [N]    – integer token indices
    qweight_ptr,       # [V, D/2] int8, packed NF4 (two 4-bit values per byte)
    scale_ptr,         # [V, D/GROUP_SIZE] float32
    mean_ptr,          # [V, D/GROUP_SIZE] float32
    codebook_ptr,      # [16] float16 – NF4 codebook
    N,
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
    GROUP_SIZE: tl.constexpr,  # group size for scale/mean quantisation
    BLOCK_D: tl.constexpr,     # output dims per program – set by heuristic (power-of-2)
):
    # 2-D grid: axis 0 = tokens, axis 1 = embedding-dim chunks.
    pid_n = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    # --- Vocab bounds check ---
    token_id_raw = tl.load(input_ptr + pid_n).to(tl.int64)
    local_id     = token_id_raw - tl.cast(vocab_start_id, tl.int64)
    valid        = (local_id >= tl.cast(0, tl.int64)) & (local_id < tl.cast(vocab_size, tl.int64))
    safe_id      = tl.where(valid, local_id, tl.cast(0, tl.int64))

    # --- Packed-byte range for this chunk ---
    # BLOCK_D output elements → BLOCK_D/2 packed bytes
    d_start    = pid_d * BLOCK_D
    half_start = d_start // 2

    half_offs = tl.arange(0, BLOCK_D // 2)                    # [BLOCK_D/2]
    half_idx  = half_start + half_offs                         # packed-byte index
    half_mask = valid & (half_idx < (embedding_dim // 2))

    # Load packed int8 bytes and convert to int32 for bit manipulation
    packed = tl.load(
        qweight_ptr + safe_id * qweight_stride_0 + half_idx * qweight_stride_1,
        mask=half_mask,
        other=0,
    ).to(tl.int32)

    # Unpack: even output dim ← low nibble, odd output dim ← high nibble
    low_nf4  = packed & 0x0F
    high_nf4 = (packed >> 4) & 0x0F

    # Output positions
    low_out_idx  = d_start + half_offs * 2        # d_start, d_start+2, ...
    high_out_idx = low_out_idx + 1                 # d_start+1, d_start+3, ...

    low_mask  = valid & (low_out_idx  < embedding_dim)
    high_mask = valid & (high_out_idx < embedding_dim)

    # NF4 codebook lookup → float32
    low_val  = tl.load(codebook_ptr + low_nf4,  mask=low_mask,  other=0.0).to(tl.float32)
    high_val = tl.load(codebook_ptr + high_nf4, mask=high_mask, other=0.0).to(tl.float32)

    # Group indices for scale / mean
    low_group  = low_out_idx  // GROUP_SIZE
    high_group = high_out_idx // GROUP_SIZE

    # Load scale and mean (float32)
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

    # Dequantize: val * scale + mean  (float32, auto-cast to out_ptr dtype on store)
    low_out  = low_val  * low_scale  + low_mean
    high_out = high_val * high_scale + high_mean

    base = pid_n * out_stride_0
    tl.store(out_ptr + base + low_out_idx  * out_stride_1, low_out,  mask=low_mask)
    tl.store(out_ptr + base + high_out_idx * out_stride_1, high_out, mask=high_mask)


def embedding_nf4_dequant_impl(
    input: torch.Tensor,
    LUT_qweight: torch.Tensor,
    LUT_scale: torch.Tensor,
    LUT_mean: torch.Tensor,
    *,
    group_size: int = 1,
    codebook: torch.Tensor = None,
    vocab_start_id: int = 0,
    vocab_size: int = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """NF4-dequantisation embedding lookup (ILU Triton).

    Args:
        input:          Integer token-index tensor of arbitrary shape.
        LUT_qweight:    [V, D/2] int8 – packed NF4 weights (two 4-bit values / byte).
        LUT_scale:      [V, D/group_size] float32 – per-group scale factors.
        LUT_mean:       [V, D/group_size] float32 – per-group mean offsets.
        group_size:     Number of embedding dimensions per quantisation group.
        codebook:       16-element float16 NF4 codebook.  Built automatically if None.
        vocab_start_id: First valid vocabulary ID (tokens below this map to zeros).
        vocab_size:     Number of valid vocabulary rows (tokens >= vocab_start_id +
                        vocab_size also map to zeros).  Defaults to LUT_qweight.size(0).
        output_dtype:   dtype of the returned embedding tensor.

    Returns:
        Tensor of shape (*input.shape, embedding_dim) with dtype output_dtype.
    """
    if input.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.uint16, torch.uint32, torch.uint64,
    ):
        raise ValueError(f"`input` must be an integer tensor, got {input.dtype}.")

    if LUT_qweight.ndim != 2 or LUT_scale.ndim != 2 or LUT_mean.ndim != 2:
        raise ValueError(
            "NF4 embedding tensors must all be 2D, "
            f"got qweight={tuple(LUT_qweight.shape)}, "
            f"scale={tuple(LUT_scale.shape)}, mean={tuple(LUT_mean.shape)}."
        )

    if LUT_scale.shape != LUT_mean.shape:
        raise ValueError(
            f"`LUT_scale` and `LUT_mean` must have the same shape, "
            f"got {LUT_scale.shape} and {LUT_mean.shape}."
        )

    if group_size <= 0:
        raise ValueError(f"`group_size` must be > 0, got {group_size}.")

    if len({input.device, LUT_qweight.device, LUT_scale.device, LUT_mean.device}) != 1:
        raise ValueError(
            "`input`, `LUT_qweight`, `LUT_scale`, and `LUT_mean` must be on the same device."
        )

    embedding_dim = LUT_scale.size(1) * group_size
    if LUT_qweight.size(1) * 2 != embedding_dim:
        raise ValueError(
            f"`LUT_qweight` shape {tuple(LUT_qweight.shape)} is incompatible with "
            f"`LUT_scale` shape {tuple(LUT_scale.shape)} and group_size={group_size}."
        )

    vocab_size = LUT_qweight.size(0) if vocab_size is None else vocab_size
    input_flat = input.contiguous().view(-1)
    N = int(input_flat.numel())

    if N == 0:
        return torch.empty(
            (*input.shape, embedding_dim),
            dtype=output_dtype,
            device=input.device,
        )

    if codebook is None:
        codebook = get_nf4_codebook(device=LUT_qweight.device, dtype=torch.float16)
    else:
        codebook = codebook.to(device=LUT_qweight.device, dtype=torch.float16)

    LUT_qweight = LUT_qweight.contiguous()
    LUT_scale   = LUT_scale.contiguous()
    LUT_mean    = LUT_mean.contiguous()

    # Output initialised to zeros so that out-of-range token IDs produce zeros.
    output = torch.zeros(N, embedding_dim, dtype=output_dtype, device=input.device)

    block_d = triton.next_power_of_2(min(embedding_dim, 256))
    grid    = (N, triton.cdiv(embedding_dim, block_d))

    _embedding_nf4_dequant_kernel[grid](
        output, input_flat,
        LUT_qweight, LUT_scale, LUT_mean, codebook,
        N=N,
        embedding_dim=embedding_dim,
        qweight_stride_0=LUT_qweight.stride(0),
        qweight_stride_1=LUT_qweight.stride(1),
        scale_stride_0=LUT_scale.stride(0),
        scale_stride_1=LUT_scale.stride(1),
        mean_stride_0=LUT_mean.stride(0),
        mean_stride_1=LUT_mean.stride(1),
        out_stride_0=output.stride(0),
        out_stride_1=output.stride(1),
        vocab_start_id=vocab_start_id,
        vocab_size=vocab_size,
        GROUP_SIZE=group_size,
    )
    return output.view(*input.shape, embedding_dim)
