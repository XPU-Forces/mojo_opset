import torch
import torch.nn.functional as F


def _apply_rope(x: torch.Tensor, rope_table: torch.Tensor) -> torch.Tensor:
    half = rope_table.shape[-1] // 2
    cos = rope_table[..., :half].unsqueeze(1)
    sin = rope_table[..., half:].unsqueeze(1)
    first = x[..., :half]
    second = x[..., half : half * 2]
    return torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
            x[..., half * 2 :],
        ),
        dim=-1,
    )


def qk_norm_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.rms_norm(q, (128,), q_weight, eps)
    k = F.rms_norm(k, (128,), k_weight, eps)
    return _apply_rope(q, rope_table).contiguous(), _apply_rope(k, rope_table).contiguous()
