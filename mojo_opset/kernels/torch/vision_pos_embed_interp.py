# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent native reference from vLLM aa31fa0 qwen3_vl.py."""

import torch


def vision_pos_embed_interp_fwd(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    hidden_dim = embed_weight.shape[1]
    h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=embed_weight.device)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=embed_weight.device)
    h_floor, w_floor = h_idxs.to(torch.long), w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)
    dh_grid, dw_grid = torch.meshgrid(h_idxs - h_floor, w_idxs - w_floor, indexing="ij")
    hf, wf = torch.meshgrid(h_floor, w_floor, indexing="ij")
    hc, wc = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01
    rows = torch.stack((hf, hf, hc, hc))
    cols = torch.stack((wf, wc, wf, wc))
    indices = (rows * num_grid_per_side + cols).reshape(4, -1)
    weights = torch.stack((w00, w01, w10, w11)).reshape(4, -1, 1).to(dtype)
    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)
    merged = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    merged = merged.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    return merged.expand(t, -1, -1).reshape(-1, hidden_dim).to(dtype)


def vision_pos_embed_interp_batch_fwd(embed_weight, grids, num_grid_per_side, m_size, dtype):
    if not grids:
        return embed_weight.new_empty((0, embed_weight.shape[1]), dtype=dtype)
    outputs = [
        vision_pos_embed_interp_fwd(embed_weight, *grids[i : i + 3], num_grid_per_side, m_size, dtype)
        for i in range(0, len(grids), 3)
    ]
    return torch.cat(outputs, dim=0)
