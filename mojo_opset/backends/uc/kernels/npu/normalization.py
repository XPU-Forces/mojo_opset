import torch

from .loader import load_uc_pybind_module


_EPS = 1e-5
_XBLOCK_SUB = 8
_BLOCK_SIZE = 128
_NORM_MODULES = {
    torch.float16: {
        "rmsnorm": "uc_rmsnorm_kernel",
        "layernorm": "uc_layernorm_kernel",
        "residual_add_rmsnorm": "uc_residual_add_rmsnorm_kernel",
        "residual_add_layernorm": "uc_residual_add_layernorm_kernel",
    },
    torch.bfloat16: {
        "rmsnorm": "uc_rmsnorm_bf16_kernel",
        "layernorm": "uc_layernorm_bf16_kernel",
        "residual_add_rmsnorm": "uc_residual_add_rmsnorm_bf16_kernel",
        "residual_add_layernorm": "uc_residual_add_layernorm_bf16_kernel",
    },
}


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


def _module_name(op_name: str, dtype: torch.dtype) -> str:
    if dtype not in _NORM_MODULES:
        supported = ", ".join(str(dtype) for dtype in _NORM_MODULES)
        raise NotImplementedError(f"UC {op_name} supports only dtypes {{{supported}}}; got dtype={dtype}.")
    return _NORM_MODULES[dtype][op_name]


def _check_npu_tensor(x: torch.Tensor, op_name: str) -> None:
    if x.device.type != "npu":
        raise TypeError(f"UC {op_name} expects an NPU tensor, got device={x.device}")


def _check_vector_param(param: torch.Tensor, x: torch.Tensor, cols: int, name: str, op_name: str) -> None:
    _check_npu_tensor(param, op_name)
    if param.dtype != x.dtype:
        raise NotImplementedError(f"UC {op_name} expects {name}.dtype={x.dtype}, got {param.dtype}.")
    if param.device != x.device:
        raise TypeError(f"UC {op_name} expects {name} on {x.device}, got {param.device}.")
    if param.dim() != 1 or param.numel() != cols:
        raise ValueError(f"UC {op_name} expects {name} shape ({cols},), got {tuple(param.shape)}.")


def _check_same_tensor_shape(lhs: torch.Tensor, rhs: torch.Tensor, lhs_name: str, rhs_name: str, op_name: str) -> None:
    _check_npu_tensor(rhs, op_name)
    if rhs.dtype != lhs.dtype:
        raise TypeError(f"UC {op_name} expects {rhs_name}.dtype={lhs.dtype}, got {rhs.dtype}.")
    if rhs.device != lhs.device:
        raise TypeError(f"UC {op_name} expects {rhs_name} on {lhs.device}, got {rhs.device}.")
    if rhs.shape != lhs.shape:
        raise ValueError(f"UC {op_name} expects matching {lhs_name}/{rhs_name} shapes, got {tuple(lhs.shape)} and {tuple(rhs.shape)}.")


def _prepare_input_2d(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int]:
    x_2d = _reshape_to_2d(x.contiguous())
    rows, cols = x_2d.shape
    padded_rows = _align_up(rows, _XBLOCK_SUB)
    padded_cols = _align_up(cols, _BLOCK_SIZE)
    if padded_rows == rows and padded_cols == cols:
        return x_2d, rows, cols, padded_rows, padded_cols

    x_kernel = torch.zeros((padded_rows, padded_cols), device=x.device, dtype=x.dtype)
    x_kernel[:rows, :cols].copy_(x_2d)
    return x_kernel, rows, cols, padded_rows, padded_cols


def _prepare_like_2d(x: torch.Tensor, padded_rows: int, padded_cols: int) -> torch.Tensor:
    x_2d = _reshape_to_2d(x.contiguous())
    rows, cols = x_2d.shape
    if padded_rows == rows and padded_cols == cols:
        return x_2d

    x_kernel = torch.zeros((padded_rows, padded_cols), device=x.device, dtype=x.dtype)
    x_kernel[:rows, :cols].copy_(x_2d)
    return x_kernel


def _prepare_vector(param: torch.Tensor, padded_cols: int) -> torch.Tensor:
    param = param.contiguous()
    cols = param.numel()
    if padded_cols == cols:
        return param

    param_kernel = torch.zeros((padded_cols,), device=param.device, dtype=param.dtype)
    param_kernel[:cols].copy_(param)
    return param_kernel


def _check_eps(eps: float, op_name: str) -> None:
    if float(eps) != _EPS:
        raise NotImplementedError(f"UC {op_name} currently supports eps={_EPS}, got eps={eps}.")


def rmsnorm_fwd_impl(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    op_name = "RMSNorm"
    _check_eps(eps, op_name)
    _check_npu_tensor(x, op_name)
    module_name = _module_name("rmsnorm", x.dtype)
    if x.numel() == 0:
        return torch.empty_like(x)

    x_kernel, rows, cols, padded_rows, padded_cols = _prepare_input_2d(x)
    _check_vector_param(weight, x, cols, "weight", op_name)
    weight_kernel = _prepare_vector(weight, padded_cols)
    y_kernel = torch.empty_like(x_kernel)

    kernel = load_uc_pybind_module(module_name)
    kernel.run_kernel(
        int(x_kernel.data_ptr()),
        int(weight_kernel.data_ptr()),
        int(y_kernel.data_ptr()),
        int(padded_rows),
        int(padded_cols),
        int(cols),
        _current_npu_stream_ptr(),
    )
    return y_kernel[:rows, :cols].contiguous().reshape(x.shape)


def layernorm_fwd_impl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    op_name = "LayerNorm"
    _check_eps(eps, op_name)
    _check_npu_tensor(x, op_name)
    module_name = _module_name("layernorm", x.dtype)
    if x.numel() == 0:
        return torch.empty_like(x)

    x_kernel, rows, cols, padded_rows, padded_cols = _prepare_input_2d(x)
    _check_vector_param(weight, x, cols, "weight", op_name)
    _check_vector_param(bias, x, cols, "bias", op_name)
    weight_kernel = _prepare_vector(weight, padded_cols)
    bias_kernel = _prepare_vector(bias, padded_cols)
    y_kernel = torch.empty_like(x_kernel)

    kernel = load_uc_pybind_module(module_name)
    kernel.run_kernel(
        int(x_kernel.data_ptr()),
        int(weight_kernel.data_ptr()),
        int(bias_kernel.data_ptr()),
        int(y_kernel.data_ptr()),
        int(padded_rows),
        int(padded_cols),
        int(cols),
        _current_npu_stream_ptr(),
    )
    return y_kernel[:rows, :cols].contiguous().reshape(x.shape)


def residual_add_rmsnorm_fwd_impl(
    hidden_state: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_pos: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    op_name = "ResidualAddRMSNorm"
    _check_eps(eps, op_name)
    _check_npu_tensor(hidden_state, op_name)
    _check_same_tensor_shape(hidden_state, residual, "hidden_state", "residual", op_name)
    module_name = _module_name("residual_add_rmsnorm", hidden_state.dtype)
    if hidden_state.numel() == 0:
        empty = torch.empty_like(hidden_state)
        return empty, empty if norm_pos == "post" else torch.empty_like(residual)

    hidden_kernel, rows, cols, padded_rows, padded_cols = _prepare_input_2d(hidden_state)
    residual_kernel = _prepare_like_2d(residual, padded_rows, padded_cols)
    _check_vector_param(weight, hidden_state, cols, "weight", op_name)
    weight_kernel = _prepare_vector(weight, padded_cols)
    y_kernel = torch.empty_like(hidden_kernel)
    residual_out_kernel = torch.empty_like(hidden_kernel)

    kernel = load_uc_pybind_module(module_name)
    kernel.run_kernel(
        int(hidden_kernel.data_ptr()),
        int(residual_kernel.data_ptr()),
        int(weight_kernel.data_ptr()),
        int(y_kernel.data_ptr()),
        int(residual_out_kernel.data_ptr()),
        int(padded_rows),
        int(padded_cols),
        int(cols),
        _current_npu_stream_ptr(),
    )
    y = y_kernel[:rows, :cols].contiguous().reshape(hidden_state.shape)
    if norm_pos == "post":
        return y, y
    residual_out = residual_out_kernel[:rows, :cols].contiguous().reshape(hidden_state.shape)
    return y, residual_out


def residual_add_layernorm_fwd_impl(
    hidden_state: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    norm_pos: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    op_name = "ResidualAddLayerNorm"
    _check_eps(eps, op_name)
    _check_npu_tensor(hidden_state, op_name)
    _check_same_tensor_shape(hidden_state, residual, "hidden_state", "residual", op_name)
    module_name = _module_name("residual_add_layernorm", hidden_state.dtype)
    if hidden_state.numel() == 0:
        empty = torch.empty_like(hidden_state)
        return empty, empty if norm_pos == "post" else torch.empty_like(residual)

    hidden_kernel, rows, cols, padded_rows, padded_cols = _prepare_input_2d(hidden_state)
    residual_kernel = _prepare_like_2d(residual, padded_rows, padded_cols)
    _check_vector_param(weight, hidden_state, cols, "weight", op_name)
    _check_vector_param(bias, hidden_state, cols, "bias", op_name)
    weight_kernel = _prepare_vector(weight, padded_cols)
    bias_kernel = _prepare_vector(bias, padded_cols)
    y_kernel = torch.empty_like(hidden_kernel)
    residual_out_kernel = torch.empty_like(hidden_kernel)

    kernel = load_uc_pybind_module(module_name)
    kernel.run_kernel(
        int(hidden_kernel.data_ptr()),
        int(residual_kernel.data_ptr()),
        int(weight_kernel.data_ptr()),
        int(bias_kernel.data_ptr()),
        int(y_kernel.data_ptr()),
        int(residual_out_kernel.data_ptr()),
        int(padded_rows),
        int(padded_cols),
        int(cols),
        _current_npu_stream_ptr(),
    )
    y = y_kernel[:rows, :cols].contiguous().reshape(hidden_state.shape)
    if norm_pos == "post":
        return y, y
    residual_out = residual_out_kernel[:rows, :cols].contiguous().reshape(hidden_state.shape)
    return y, residual_out
