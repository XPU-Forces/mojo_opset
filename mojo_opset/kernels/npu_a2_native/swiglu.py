"""Queue-submitted SwiGLU with FP32 scaling and input-dtype outputs."""

from typing import Optional

import torch

from ._extension import launcher


@torch.library.custom_op("mojo_npu_native_a2::swiglu_fwd", mutates_args=())
def swiglu_fwd(a: torch.Tensor, b: torch.Tensor, scales: Optional[torch.Tensor] = None) -> torch.Tensor:
    if scales is None:
        return launcher("swiglu_forward")(a.contiguous(), b.contiguous())
    output = launcher("swiglu_forward")(a.float().contiguous(), b.float().contiguous())
    return (output * scales.float().reshape(*a.shape[:-1], 1)).to(a.dtype)


@swiglu_fwd.register_fake
def _fake_fwd(a, b, scales=None):
    return torch.empty_like(a, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_native_a2::swiglu_bwd", mutates_args=())
def _backward(
    grad: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_a = a.contiguous() if scales is None else a.float().contiguous()
    compute_b = b.contiguous() if scales is None else b.float().contiguous()
    weighted = grad if scales is None else grad.float() * scales.float().reshape(*a.shape[:-1], 1)
    da, db = launcher("swiglu_backward")(weighted.contiguous(), compute_a, compute_b)
    ds = a.new_empty((0,))
    if scales is not None:
        output = launcher("swiglu_forward")(compute_a, compute_b)
        ds = (output * grad.float()).sum(-1).reshape(scales.shape).to(scales.dtype)
    return da.to(a.dtype), db.to(b.dtype), ds


@_backward.register_fake
def _fake_bwd(grad, a, b, scales=None):
    return (
        torch.empty_like(a, memory_format=torch.contiguous_format),
        torch.empty_like(b, memory_format=torch.contiguous_format),
        a.new_empty((0,)) if scales is None else torch.empty_like(scales, memory_format=torch.contiguous_format),
    )


def swiglu_bwd(grad, a, b, scales=None):
    da, db, ds = _backward(grad, a, b, scales)
    return da, db, None if scales is None else ds
