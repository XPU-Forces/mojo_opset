"""torch_npu rotary-mul leaves from the original inference operators."""

from typing import Optional

import torch
import torch_npu


@torch.library.custom_op("mojo_npu_torch_npu::rotary_mul", mutates_args=())
def _rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_mode: Optional[str]) -> torch.Tensor:
    if rotary_mode is None:
        return torch_npu.npu_rotary_mul(x, cos, sin)
    return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode=rotary_mode)


@_rotary_mul.register_fake
def _rotary_mul_fake(x, cos, sin, rotary_mode):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def apply_rope_infer_fwd(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_first: bool, keep_cos_sin_dtype: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos.unsqueeze(-3 if head_first else -2), sin.unsqueeze(-3 if head_first else -2)
    if cos.ndim < 4:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    dim = cos.shape[-1]
    outputs = []
    for x in (q, k):
        tail = x[..., -dim:]
        if x.ndim < 4:
            tail = tail.unsqueeze(0)
        if tail.shape[0] > 1 and cos.shape[1] == 1 and dim // 2 * x.element_size() % 32:
            raise NotImplementedError("torch_npu BNSD RoPE requires the half rotary dimension aligned to 32 bytes")
        output = _rotary_mul(tail, cos, sin, None)
        if x.ndim < 4:
            output = output.squeeze(0)
        outputs.append(torch.cat((x[..., :-dim], output), dim=-1))
    return outputs[0], outputs[1]


def apply_vision_rope2d_infer_fwd(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
    return tuple(_rotary_mul(x.unsqueeze(0), cos, sin, "half").squeeze(0).to(x.dtype) for x in (q, k))
