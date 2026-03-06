from .layernorm import layernorm_infer_impl
from .layernorm import layernorm_bwd_impl
from .layernorm import layernorm_fwd_impl

__all__ = [
    "layernorm_infer_impl",
    "layernorm_bwd_impl",
    "layernorm_fwd_impl",
]

from mojo_opset.backends.ttx.kernels.utils import tensor_device_guard_for_triton_kernel

# NOTE(liuyuan): Automatically add guard to torch tensor for triton kernels.
tensor_device_guard_for_triton_kernel(__path__, __name__, "mlu")
