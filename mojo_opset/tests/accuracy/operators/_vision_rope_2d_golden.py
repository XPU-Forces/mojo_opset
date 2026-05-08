import torch


def _rotate_half_golden(x: torch.Tensor) -> torch.Tensor:
    # Frozen helper that swaps the two half-dim slices.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _vision_rotary_embedding_golden(
    seqlen: int,
    dim: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    # Frozen 1D vision rotary table builder.
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
    return torch.outer(seq, inv_freq)


def _rot_pos_emb_golden(
    grid_hw: torch.Tensor,
    rope_dim: int,
    rope_theta: float,
    device: torch.device,
    adapooling_factor: int = 1,
) -> torch.Tensor:
    # Keep the original adapooling-aware reorder exactly so the golden stays
    # anchored to the native patch packing order rather than a rewritten formula.
    pos_ids = []
    for h, w in grid_hw.to(dtype=torch.int64).tolist():
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // adapooling_factor,
            adapooling_factor,
            w // adapooling_factor,
            adapooling_factor,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // adapooling_factor,
            adapooling_factor,
            w // adapooling_factor,
            adapooling_factor,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))

    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = int(grid_hw.max().item())
    rotary_pos_emb_full = _vision_rotary_embedding_golden(
        seqlen=max_grid_size,
        dim=rope_dim // 2,
        theta=rope_theta,
        device=device,
    )
    return rotary_pos_emb_full[pos_ids].flatten(1)


def _vision_rotary_embedding_2d_golden(
    grid_hw: torch.Tensor,
    rope_theta: float,
    rope_dim: int,
    device: torch.device,
    adapooling_factor: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = _rot_pos_emb_golden(
        grid_hw,
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        device=device,
        adapooling_factor=adapooling_factor,
    )
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _apply_vision_rope_2d_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This helper intentionally stays on the native contract:
    # packed token-first q/k tensors with the full head_dim rotated.
    assert q.ndim == 3 and k.ndim == 3, "vision rope golden expects packed token-first tensors"
    assert cos.ndim == 2 and sin.ndim == 2, "vision rotary embedding golden expects [num_tokens, rope_dim]"
    assert cos.shape == sin.shape, "vision rotary embedding golden expects matched cos/sin shapes"
    assert q.shape[-1] == cos.shape[-1] and k.shape[-1] == cos.shape[-1], (
        "vision rope golden rotates the full head_dim"
    )
    q_orig_dtype = q.dtype
    k_orig_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        ((q * cos) + (_rotate_half_golden(q) * sin)).to(q_orig_dtype),
        ((k * cos) + (_rotate_half_golden(k) * sin)).to(k_orig_dtype),
    )
