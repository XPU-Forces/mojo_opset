"""Original torch_npu activations, with explicit non-autograd leaves."""

from typing import Optional

import torch
import torch_npu


@torch.library.custom_op("mojo_npu_torch_npu::silu_fwd", mutates_args=())
def silu_fwd(x: torch.Tensor) -> torch.Tensor:
    # torch_npu deprecates npu_silu; its aten replacement uses ACLNN on A5.
    return torch.nn.functional.silu(x.contiguous())


@silu_fwd.register_fake
def _silu_fake(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_torch_npu::silu_bwd", mutates_args=())
def silu_bwd(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu_backward(grad.contiguous(), x.contiguous())


@silu_bwd.register_fake
def _silu_bwd_fake(grad, x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_torch_npu::gelu_fwd", mutates_args=())
def gelu_fwd(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    return torch_npu.npu_gelu(x.contiguous(), approximate=approximate)


@gelu_fwd.register_fake
def _gelu_fake(x, approximate="tanh"):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_torch_npu::gelu_bwd", mutates_args=())
def gelu_bwd(grad: torch.Tensor, x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    # Match npu_gelu's autograd path; older NPU ATen backward ignores exact mode.
    return torch.ops.npu.npu_gelu_backward(grad.contiguous(), x.contiguous(), approximate=approximate)


@gelu_bwd.register_fake
def _gelu_bwd_fake(grad, x, approximate="tanh"):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_torch_npu::swiglu_fwd", mutates_args=())
def swiglu_fwd(x1: torch.Tensor, x2: torch.Tensor, scales: Optional[torch.Tensor] = None) -> torch.Tensor:
    if scales is not None:
        raise NotImplementedError("torch_npu SwiGLU does not support per-row scales")
    return torch_npu.npu_swiglu(torch.cat((x1, x2), dim=-1), dim=-1)


@swiglu_fwd.register_fake
def _swiglu_fake(x1, x2, scales=None):
    if scales is not None:
        raise NotImplementedError("torch_npu SwiGLU does not support per-row scales")
    return torch.empty_like(x1, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_torch_npu::swiglu_bwd", mutates_args=())
def _swiglu_bwd(
    grad: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, scales: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if scales is not None:
        raise NotImplementedError("torch_npu SwiGLU does not support per-row scales")
    merged = torch.cat((x1, x2), dim=-1)
    dx1, dx2 = torch_npu.npu_swiglu_backward(grad.contiguous(), merged, dim=-1).chunk(2, dim=-1)
    # Custom-op outputs must not alias one another (even disjoint slices).
    return dx1.clone(), dx2.clone()


@_swiglu_bwd.register_fake
def _swiglu_bwd_fake(grad, x1, x2, scales=None):
    return (
        torch.empty_like(x1, memory_format=torch.contiguous_format),
        torch.empty_like(x2, memory_format=torch.contiguous_format),
    )


def swiglu_bwd(grad, x1, x2, scales=None):
    dx1, dx2 = _swiglu_bwd(grad, x1, x2, scales)
    return dx1, dx2, None
