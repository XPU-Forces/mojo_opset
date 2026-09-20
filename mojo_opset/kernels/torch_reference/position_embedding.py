"""Original inference position-embedding semantics, without backend dispatch."""

import torch


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope_infer_fwd(q, k, cos, sin, head_first, keep_cos_sin_dtype):
    cos = cos.unsqueeze(-3 if head_first else -2)
    sin = sin.unsqueeze(-3 if head_first else -2)
    dim = cos.shape[-1]
    outputs = []
    for x in (q, k):
        tail = x[..., -dim:]
        rotated = (tail * cos + _rotate_half(tail) * sin).to(x.dtype)
        outputs.append(torch.cat((x[..., :-dim], rotated), dim=-1))
    return tuple(outputs)


def apply_vision_rope2d_infer_fwd(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return tuple((x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype) for x in (q, k))
