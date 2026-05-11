from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _uc_kernels():
    import uc_kernel

    return uc_kernel.load()


def _matrix_shape(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.dim() == 0:
        return 1, 1
    if tensor.dim() == 1:
        return 1, tensor.numel()
    return tensor.numel() // tensor.shape[-1], tensor.shape[-1]


def run_unary_kernel(api: str, x: torch.Tensor) -> torch.Tensor:
    original_dtype = x.dtype
    kernel_input = x.contiguous()
    if kernel_input.dtype != torch.float16:
        kernel_input = kernel_input.to(torch.float16)

    kernel_output = torch.empty_like(kernel_input)
    rows, cols = _matrix_shape(kernel_input)
    _uc_kernels()[api](kernel_input, kernel_output, rows, cols)

    output = kernel_output.reshape(x.shape)
    if output.dtype != original_dtype:
        output = output.to(original_dtype)
    return output


def run_binary_kernel(api: str, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.shape != rhs.shape:
        raise ValueError(f"UC backend {api} expects matching input shapes, got {lhs.shape} and {rhs.shape}.")

    original_dtype = lhs.dtype
    kernel_lhs = lhs.contiguous()
    kernel_rhs = rhs.contiguous()
    if kernel_lhs.dtype != torch.float16:
        kernel_lhs = kernel_lhs.to(torch.float16)
    if kernel_rhs.dtype != torch.float16:
        kernel_rhs = kernel_rhs.to(torch.float16)

    kernel_output = torch.empty_like(kernel_lhs)
    rows, cols = _matrix_shape(kernel_lhs)
    _uc_kernels()[api](kernel_lhs, kernel_rhs, kernel_output, rows, cols)

    output = kernel_output.reshape(lhs.shape)
    if output.dtype != original_dtype:
        output = output.to(original_dtype)
    return output
