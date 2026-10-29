"""Original inference position-embedding semantics, without backend dispatch."""

import torch


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_infer_fwd(q, k, cos, sin, head_first, keep_cos_sin_dtype):
    cos = cos.unsqueeze(-3 if head_first else -2)
    sin = sin.unsqueeze(-3 if head_first else -2)
    dim = cos.shape[-1]
    outputs = []
    for x in (q, k):
        tail = x[..., -dim:]
        rotated = (tail * cos + _rotate_half(tail) * sin).to(x.dtype)
        outputs.append(torch.cat((x[..., :-dim], rotated), dim=-1))
    return tuple(outputs)


def vision_rope_2d_infer_fwd(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return tuple((x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype) for x in (q, k))


def pos_embed_interp_infer_fwd(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    """Eager bilinear interpolation in spatial-merge order.

    Ported from ``pos_embed_interpolate_native`` (vLLM, Apache-2.0): ``torch.linspace``
    sampling, 4-neighbour gather with pre-cast weights, then the spatial-merge permute.
    This is the accuracy reference for the Triton kernel and the NPU production fallback.
    """
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device
    out_dtype = embed_weight.dtype if dtype is None else dtype

    h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=device)

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    weights = weights.to(dtype=out_dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=out_dtype)


def pos_embed_interp_multi_infer_fwd(embed_weight, grids, num_grid_per_side, m_size, dtype):
    """Per-grid eager interpolation, concatenated along dim 0."""
    if not grids:
        raise ValueError("grids must be non-empty")
    outs = [
        pos_embed_interp_infer_fwd(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)
        for t, h, w in grids
    ]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
