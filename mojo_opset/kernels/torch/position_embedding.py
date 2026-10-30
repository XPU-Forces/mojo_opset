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


def _pos_embed_interpolate_native_one(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    """Eager reference matching upstream vLLM ``pos_embed_interpolate_native``.

    Kept deliberately close to the upstream source: FP32 linspace sampling,
    four-neighbour gather, weight cast to the output dtype, and the
    spatial-merge permute.  Changes here can hide a kernel precision bug.
    """
    if h % m_size != 0 or w % m_size != 0:
        raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

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
    weights = weights.to(dtype=dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)


def vision_pos_embed_interp_fwd(embed_weight, grid_thw, num_grid_per_side, m_size, dtype=None):
    dtype = embed_weight.dtype if dtype is None else dtype
    outputs = [
        _pos_embed_interpolate_native_one(embed_weight, int(t), int(h), int(w), num_grid_per_side, m_size, dtype)
        for t, h, w in grid_thw
    ]
    if not outputs:
        return torch.empty((0, embed_weight.shape[1]), dtype=dtype, device=embed_weight.device)
    return torch.cat(outputs, dim=0)
