"""Eager reference for latent RoPE + quantised paged-cache insert.

Writes into the same physical cells as the kernel: for ``uint8`` records the
values are addressed through the storage (row pitch 576/528 bytes) and the FP8
scales live in the page's trailing scale area, not inside the row. ``plain``
records are addressed through the tensor's own strides.
"""

import torch

FP8_MAX = 448.0
E8M0_MIN = 1e-4
FP8_DTYPE = torch.float8_e4m3fn


def _rotate_tail(latent_rows, cs_rows):
    """GPT-J half-rotation of the last 64 dims (FP32); returns (n, 64)."""
    row = latent_rows.float()
    cos, sin = cs_rows[:, :32], cs_rows[:, 32:]
    even, odd = row[:, 448::2], row[:, 449::2]
    out = torch.empty_like(row[:, 448:])
    out[:, 0::2] = even * cos - odd * sin
    out[:, 1::2] = odd * cos + even * sin
    return out


def _ue8m0_quant(x, block):
    """Group-wise UE8M0 FP8 with the kernel's operation order; returns (bytes, scale)."""
    width = x.shape[-1]
    tiles = x.reshape(*x.shape[:-1], width // block, block)
    amax = tiles.abs().amax(-1).clamp(min=E8M0_MIN)
    exponent = torch.ceil(torch.log2(amax * (1.0 / FP8_MAX)))
    scaled = tiles * torch.exp2(-exponent)[..., None]
    q = scaled.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    enc = (exponent + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    return q.view(torch.uint8).reshape(*x.shape[:-1], width), enc


def _selection(positions, slot_mapping, compress_ratio):
    valid = (slot_mapping >= 0) & ((positions + 1) % compress_ratio == 0)
    return valid.nonzero().flatten()


def _flat_storage(tensor):
    """1D view of the whole storage (element units), starting at offset 0."""
    elements = tensor.untyped_storage().size() // tensor.element_size()
    return torch.as_strided(tensor, (elements,), (1,), 0)


def rope_quant_insert_fwd(
    latent, positions, cos_sin_cache, kv_cache, slot_mapping, compress_ratio, fp8_scale, slot_contig=None
):
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return None
    index = _selection(positions, slot_mapping, compress_ratio)
    if index.numel() == 0:
        return None
    block = kv_cache.shape[1]
    slot = slot_mapping[index]
    page = (slot // block).long()
    row = (slot % block).long()
    position = positions[index]
    cs_rows = cos_sin_cache[position // compress_ratio * compress_ratio].float()
    rotated = _rotate_tail(latent[index], cs_rows)

    if kv_cache.dtype == torch.uint8:
        record = kv_cache.shape[-1]
        if record not in (584, 528):
            raise ValueError(f"unsupported paged KV record width {record}")
        flat = _flat_storage(kv_cache)
        base = kv_cache.storage_offset()
        page_base = base + page * kv_cache.stride(0)
        if record == 584:
            q, enc = _ue8m0_quant(latent[index, :448].float(), 64)
            value_offset = page_base[:, None] + row[:, None] * 576
            flat[value_offset + torch.arange(448, device=q.device)] = q
            rope_bytes = rotated.to(torch.bfloat16).view(torch.uint8)
            flat[value_offset + 448 + torch.arange(128, device=q.device)] = rope_bytes
            enc8 = torch.zeros((index.numel(), 8), dtype=torch.uint8, device=q.device)
            enc8[:, :7] = enc
            scale_offset = page_base + block * 576 + row * 8
            flat[scale_offset[:, None] + torch.arange(8, device=q.device)] = enc8
        else:
            full = torch.cat((latent[index].float()[:, :448], rotated), dim=-1)
            q, enc = _ue8m0_quant(full, 32)
            value_offset = page_base[:, None] + row[:, None] * 528
            flat[value_offset + torch.arange(512, device=q.device)] = q
            flat[value_offset + 512 + torch.arange(16, device=q.device)] = enc
        return None

    values = torch.cat((latent[index, :448], rotated.to(torch.bfloat16)), dim=-1)
    if kv_cache.dtype == torch.float8_e4m3fn:
        scaled = (values.float() * (1.0 / float(fp8_scale))).clamp(-FP8_MAX, FP8_MAX)
        values = scaled.to(FP8_DTYPE)
    # Write through uint8: the NPU backend has no fp8 element-wise index_put.
    record_bytes = kv_cache.shape[-1] * kv_cache.element_size()
    cache_bytes = kv_cache.view(torch.uint8)
    value_bytes = values.view(torch.uint8)
    cache_bytes[
        page[:, None], row[:, None], torch.arange(record_bytes, device=values.device)
    ] = value_bytes
    return None
