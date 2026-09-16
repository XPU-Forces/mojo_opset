import torch

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _inverse_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x2, -x1), dim=-1)


def _apply_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    rope_dim = cos.shape[-1]
    nope_dim = x.shape[-1] - rope_dim
    x_nope, x_rope = torch.split(x, [nope_dim, rope_dim], dim=-1) if nope_dim else (None, x)
    x_rope_float = x_rope.float()
    if inverse:
        rotated = x_rope_float * cos.float() + _inverse_rotate_half(x_rope_float * sin.float())
    else:
        rotated = x_rope_float * cos.float() + _rotate_half(x_rope_float) * sin.float()
    rotated = rotated.to(x.dtype)
    return torch.cat((x_nope, rotated), dim=-1) if x_nope is not None else rotated


def apply_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _apply_one(q, cos, sin, inverse=False), _apply_one(k, cos, sin, inverse=False)


def apply_rope_bwd(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _apply_one(grad_q, cos, sin, inverse=True), _apply_one(grad_k, cos, sin, inverse=True)
