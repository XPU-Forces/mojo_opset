import torch


def gelu_fwd(x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_compute = x.to(compute_dtype)
    # Keep the semantic oracle independent of vendor GELU dispatch. Older NPU
    # ATen implementations use tanh even when approximate="none" is requested.
    if approximate == "none":
        output = 0.5 * x_compute * (1 + torch.erf(x_compute * 0.7071067811865476))
    else:
        inner = 0.7978845608028654 * (x_compute + 0.044715 * x_compute.pow(3))
        output = 0.5 * x_compute * (1 + torch.tanh(inner))
    return output.to(x.dtype)


def gelu_bwd(grad_output: torch.Tensor, x: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_compute = x.to(compute_dtype)
    if approximate == "none":
        derivative = 0.5 * (1 + torch.erf(x_compute * 0.7071067811865476))
        derivative += x_compute * torch.exp(-0.5 * x_compute.square()) * 0.3989422804014327
        return (grad_output.to(compute_dtype) * derivative).to(x.dtype)
    inner = 0.7978845608028654 * (x_compute + 0.044715 * x_compute.pow(3))
    tanh_inner = torch.tanh(inner)
    derivative = 0.5 * (1 + tanh_inner)
    derivative += (
        0.5 * x_compute * (1 - tanh_inner.square()) * 0.7978845608028654 * (1 + 3 * 0.044715 * x_compute.square())
    )
    return (grad_output.to(compute_dtype) * derivative).to(x.dtype)
