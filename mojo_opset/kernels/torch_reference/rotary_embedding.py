import torch


def rotary_embedding_fwd(x, inv_freq, cos, sin, cu_q_lens, total_seq_lens, position_ids, attention_scaling):
    if cu_q_lens is not None:
        if x.ndim != 2:
            raise ValueError("packed prefill x must have shape [tokens, hidden]")
        position_ids = torch.full((x.shape[0],), -1, device=x.device, dtype=torch.int32)
        offsets = cu_q_lens.cpu().tolist()
        lengths = total_seq_lens.cpu().tolist() if total_seq_lens is not None else None
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            context = 0 if lengths is None else lengths[index] - (end - start)
            position_ids[start:end] = torch.arange(context, context + end - start, device=x.device, dtype=torch.int32)
    elif position_ids is not None:
        if position_ids.shape != x.shape[:-1]:
            raise ValueError("position_ids shape must match x without its hidden dimension")
    else:
        position_ids = torch.arange(x.shape[1], device=x.device, dtype=torch.int32)
    if cos is not None and sin is not None:
        return cos[position_ids], sin[position_ids]
    freqs = position_ids[..., None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


def vision_rotary_embedding2d_fwd(inv_freq, grid_hw, rope_dim, adapooling_factor):
    device = inv_freq.device if inv_freq.device.type != "cpu" or grid_hw.device.type == "cpu" else grid_hw.device
    if grid_hw.ndim != 2 or grid_hw.shape[-1] != 2 or grid_hw.is_floating_point():
        raise ValueError("grid_hw must be an integer [batch, 2] tensor")
    grids = grid_hw.cpu().tolist()
    positions = []
    for height, width in grids:
        if height <= 0 or width <= 0 or height % adapooling_factor or width % adapooling_factor:
            raise ValueError("grid extents must be positive multiples of adapooling_factor")
        h = torch.arange(height, device=device)[:, None].expand(-1, width)
        w = torch.arange(width, device=device)[None, :].expand(height, -1)
        layout = (height // adapooling_factor, adapooling_factor, width // adapooling_factor, adapooling_factor)
        h, w = (x.reshape(layout).permute(0, 2, 1, 3).flatten() for x in (h, w))
        positions.append(torch.stack((h, w), dim=-1))
    full = torch.outer(
        torch.arange(max(max(grid) for grid in grids), device=device, dtype=inv_freq.dtype), inv_freq.to(device)
    )
    freqs = full[torch.cat(positions)].flatten(-2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()
