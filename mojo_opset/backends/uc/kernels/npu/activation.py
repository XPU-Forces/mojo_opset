import torch

from .loader import load_uc_pybind_module


_GELU_MODULE_BY_DTYPE = {
    torch.float16: "uc_gelu_kernel",
    torch.bfloat16: "uc_gelu_bf16_kernel",
}
_SILU_MODULE_BY_DTYPE = {
    torch.float16: "uc_silu_kernel",
    torch.bfloat16: "uc_silu_bf16_kernel",
}
_SWIGLU_MODULE_BY_DTYPE = {
    torch.float16: "uc_swiglu_kernel",
    torch.bfloat16: "uc_swiglu_bf16_kernel",
}
_XBLOCK_SUB = 8
_BLOCK_SIZE = 128


def _current_npu_stream_ptr() -> int:
    import torch_npu

    return int(torch_npu.npu.current_stream().npu_stream)


def _align_up(value: int, block: int) -> int:
    return ((value + block - 1) // block) * block


def _reshape_to_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 0:
        return x.reshape(1, 1)
    if x.numel() == 0:
        return x.reshape(0, 0)
    return x.reshape(-1, x.shape[-1])


def _check_npu_tensor(x: torch.Tensor, op_name: str) -> None:
    if x.device.type != "npu":
        raise TypeError(f"UC {op_name} expects an NPU tensor, got device={x.device}")


def _check_supported_dtype(x: torch.Tensor, module_by_dtype: dict[torch.dtype, str], op_name: str) -> None:
    if x.dtype not in module_by_dtype:
        supported = ", ".join(str(dtype) for dtype in module_by_dtype)
        raise NotImplementedError(f"UC {op_name} supports only dtypes {{{supported}}}; got dtype={x.dtype}.")


def _padded_2d(x_2d: torch.Tensor, padded_rows: int, padded_cols: int) -> torch.Tensor:
    rows, cols = x_2d.shape
    if padded_rows == rows and padded_cols == cols:
        return x_2d

    x_kernel = torch.empty((padded_rows, padded_cols), device=x_2d.device, dtype=x_2d.dtype)
    x_kernel[:rows, :cols].copy_(x_2d)
    return x_kernel


def _run_unary_activation(x: torch.Tensor, module_by_dtype: dict[torch.dtype, str], op_name: str) -> torch.Tensor:
    _check_npu_tensor(x, op_name)
    _check_supported_dtype(x, module_by_dtype, op_name)
    if x.numel() == 0:
        return torch.empty_like(x)

    x_contiguous = x.contiguous()
    x_2d = _reshape_to_2d(x_contiguous)
    rows, cols = x_2d.shape
    padded_rows = _align_up(rows, _XBLOCK_SUB)
    padded_cols = _align_up(cols, _BLOCK_SIZE)
    x_kernel = _padded_2d(x_2d, padded_rows, padded_cols)
    y_kernel = torch.empty_like(x_kernel)

    kernel = load_uc_pybind_module(module_by_dtype[x.dtype])
    kernel.run_kernel(
        int(x_kernel.data_ptr()),
        int(y_kernel.data_ptr()),
        int(padded_rows),
        int(padded_cols),
        _current_npu_stream_ptr(),
    )

    return y_kernel[:rows, :cols].contiguous().reshape(x.shape)


def _run_binary_activation(
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    module_by_dtype: dict[torch.dtype, str],
    op_name: str,
) -> torch.Tensor:
    _check_npu_tensor(gate_out, op_name)
    _check_npu_tensor(up_out, op_name)
    _check_supported_dtype(gate_out, module_by_dtype, op_name)
    if up_out.dtype != gate_out.dtype:
        raise TypeError(f"UC {op_name} expects matching dtypes, got {gate_out.dtype} and {up_out.dtype}.")
    if up_out.device != gate_out.device:
        raise TypeError(f"UC {op_name} expects tensors on the same device, got {gate_out.device} and {up_out.device}.")
    if up_out.shape != gate_out.shape:
        raise ValueError(f"UC {op_name} expects matching shapes, got {tuple(gate_out.shape)} and {tuple(up_out.shape)}.")
    if gate_out.numel() == 0:
        return torch.empty_like(gate_out)

    gate_contiguous = gate_out.contiguous()
    up_contiguous = up_out.contiguous()
    gate_2d = _reshape_to_2d(gate_contiguous)
    up_2d = _reshape_to_2d(up_contiguous)
    rows, cols = gate_2d.shape
    padded_rows = _align_up(rows, _XBLOCK_SUB)
    padded_cols = _align_up(cols, _BLOCK_SIZE)
    gate_kernel = _padded_2d(gate_2d, padded_rows, padded_cols)
    up_kernel = _padded_2d(up_2d, padded_rows, padded_cols)
    y_kernel = torch.empty_like(gate_kernel)

    kernel = load_uc_pybind_module(module_by_dtype[gate_out.dtype])
    kernel.run_kernel(
        int(gate_kernel.data_ptr()),
        int(up_kernel.data_ptr()),
        int(y_kernel.data_ptr()),
        int(padded_rows),
        int(padded_cols),
        _current_npu_stream_ptr(),
    )

    return y_kernel[:rows, :cols].contiguous().reshape(gate_out.shape)


def gelu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    return _run_unary_activation(x, _GELU_MODULE_BY_DTYPE, "Gelu")


def silu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    return _run_unary_activation(x, _SILU_MODULE_BY_DTYPE, "Silu")


def swiglu_fwd_impl(gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
    return _run_binary_activation(gate_out, up_out, _SWIGLU_MODULE_BY_DTYPE, "SwiGLU")
