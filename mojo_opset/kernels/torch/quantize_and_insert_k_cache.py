"""Torch reference for quantize_and_insert_k_cache (UE8M0 / MXFP8)."""

import torch

V4_BYTES_PER_TOKEN = 584
V41_BYTES_PER_TOKEN = 528
V41_QUANT_BLOCK = 32
V41_NUM_SCALES = 512 // V41_QUANT_BLOCK  # 16

V4_FP8_DIM = 448
V4_BF16_DIM = 64
V4_SCALE_DIM = 8
V4_QUANT_BLOCK = 64
V4_DATA_SIZE = V4_FP8_DIM + V4_BF16_DIM * 2  # 576

_FP8_MAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e4m3fnuz: 240.0,
}


def _ue8m0_quantize(x: torch.Tensor, quant_block: int, fp8_dtype, fp8_max: float):
    """Quantize ``[n, d]`` fp32 to fp8 with one UE8M0 scale per ``quant_block``.

    Returns ``(fp8_bytes [n, d] uint8, encoded_scales [n, d // quant_block] uint8)``.
    """
    n, d = x.shape
    tiles = x.reshape(n, d // quant_block, quant_block)
    amax = tiles.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / fp8_max))
    scale = torch.exp2(exponent)
    scaled = (tiles / scale[..., None]).clamp(-fp8_max, fp8_max)
    fp8 = scaled.to(fp8_dtype).reshape(n, d).view(torch.uint8)
    encoded = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
    return fp8, encoded


def quantize_and_insert_k_cache_fwd(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
    is_ue8m0: bool = True,
    use_fnuz: bool = False,
    bytes_per_token: int = V4_BYTES_PER_TOKEN,
) -> torch.Tensor:
    """Quantize bf16 K rows and scatter them into the paged cache in place."""
    assert is_ue8m0, "Only UE8M0 quantization is supported"
    assert k_cache.dtype == torch.uint8 and k_cache.dim() == 2
    assert k_cache.is_contiguous(), "k_cache must be contiguous"
    assert k.dim() == 2 and k.shape[1] == 512 and k.dtype == torch.bfloat16

    fp8_dtype = torch.float8_e4m3fnuz if use_fnuz else torch.float8_e4m3fn
    fp8_max = _FP8_MAX[fp8_dtype]

    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return k_cache

    block_stride = k_cache.stride(0)
    flat = k_cache.reshape(-1)

    valid = slot_mapping != -1
    slots = slot_mapping[valid]
    if slots.numel() == 0:
        return k_cache
    kv_bf16 = k[:num_tokens][valid]
    kv = kv_bf16.float()
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    pos_in_block = slots % block_size
    device = k.device

    if bytes_per_token == V41_BYTES_PER_TOKEN:
        fp8_bytes, encoded = _ue8m0_quantize(kv, V41_QUANT_BLOCK, fp8_dtype, fp8_max)
        base = block_idx * block_stride + pos_in_block * (512 + V41_NUM_SCALES)
        flat[base[:, None] + torch.arange(512, device=device)] = fp8_bytes
        flat[base[:, None] + 512 + torch.arange(V41_NUM_SCALES, device=device)] = encoded
        return k_cache

    fp8_bytes, encoded = _ue8m0_quantize(
        kv[:, :V4_FP8_DIM], V4_QUANT_BLOCK, fp8_dtype, fp8_max
    )
    tail_bytes = kv_bf16[:, V4_FP8_DIM:].contiguous().view(torch.uint8)

    base = block_idx * block_stride + pos_in_block * V4_DATA_SIZE
    flat[base[:, None] + torch.arange(V4_FP8_DIM, device=device)] = fp8_bytes
    flat[base[:, None] + V4_FP8_DIM + torch.arange(V4_BF16_DIM * 2, device=device)] = tail_bytes

    scale_base = block_idx * block_stride + block_size * V4_DATA_SIZE + pos_in_block * V4_SCALE_DIM
    flat[scale_base[:, None] + torch.arange(V4_FP8_DIM // V4_QUANT_BLOCK, device=device)] = encoded
    flat[scale_base + (V4_SCALE_DIM - 1)] = 0
    return k_cache
