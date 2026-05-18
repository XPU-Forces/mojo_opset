from functools import lru_cache

import torch


_DTYPE_API_SUFFIX = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


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


def _typed_api(api: str, dtype: torch.dtype) -> str:
    suffix = _DTYPE_API_SUFFIX.get(dtype)
    if suffix is None:
        raise NotImplementedError(f"UC backend {api} does not support dtype {dtype}.")

    kernels = _uc_kernels()
    typed_api = f"{api}_{suffix}"
    if typed_api in kernels.keys():
        return typed_api
    if dtype == torch.float16 and api in kernels.keys():
        return api
    raise NotImplementedError(f"UC backend {api} does not provide a {suffix} kernel artifact.")


def run_unary_kernel(api: str, x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty_like(x)

    kernel_input = x.contiguous()
    kernel_output = torch.empty_like(kernel_input)
    rows, cols = _matrix_shape(kernel_input)
    _uc_kernels()[_typed_api(api, kernel_input.dtype)](kernel_input, kernel_output, rows, cols)
    return kernel_output.reshape(x.shape)


def run_binary_kernel(api: str, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.shape != rhs.shape:
        raise ValueError(f"UC backend {api} expects matching input shapes, got {lhs.shape} and {rhs.shape}.")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"UC backend {api} expects matching input dtypes, got {lhs.dtype} and {rhs.dtype}.")
    if lhs.numel() == 0:
        return torch.empty_like(lhs)

    kernel_lhs = lhs.contiguous()
    kernel_rhs = rhs.contiguous()
    kernel_output = torch.empty_like(kernel_lhs)
    rows, cols = _matrix_shape(kernel_lhs)
    _uc_kernels()[_typed_api(api, kernel_lhs.dtype)](kernel_lhs, kernel_rhs, kernel_output, rows, cols)
    return kernel_output.reshape(lhs.shape)


def run_kernel(api: str, dtype: torch.dtype, *args) -> None:
    _uc_kernels()[_typed_api(api, dtype)](*args)
