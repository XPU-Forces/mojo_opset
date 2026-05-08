import torch

from .loader import load_uc_pybind_module


_GELU_MODULE_NAME = "uc_gelu_kernel"
_GELU_SHAPE = (4096, 128)
_GELU_DTYPE = torch.float16


def _current_npu_stream_ptr() -> int:
    import torch_npu

    return int(torch_npu.npu.current_stream().npu_stream)


def gelu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise TypeError(f"UC Gelu expects an NPU tensor, got device={x.device}")
    if tuple(x.shape) != _GELU_SHAPE or x.dtype != _GELU_DTYPE:
        raise NotImplementedError(
            f"UC Gelu currently supports only shape={_GELU_SHAPE}, dtype={_GELU_DTYPE}; "
            f"got shape={tuple(x.shape)}, dtype={x.dtype}."
        )

    x_contiguous = x.contiguous()
    y = torch.empty_like(x_contiguous)

    kernel = load_uc_pybind_module(_GELU_MODULE_NAME)
    kernel.run_kernel(int(x_contiguous.data_ptr()), int(y.data_ptr()), _current_npu_stream_ptr())
    return y.reshape(x.shape)
