from .group_rmsnorm import group_rmsnorm_impl
from .kv_cache import store_paged_kv_impl
from .layernorm import layernorm_bwd_impl
from .layernorm import layernorm_fwd_impl
from .layernorm import layernorm_infer_impl
from .swa import swa_paged_decode_impl
from .swa import swa_paged_prefill_impl

__all__ = [
    "group_rmsnorm_impl",
    "kv_cache",
    "layernorm_bwd_impl",
    "layernorm_fwd_impl",
    "layernorm_infer_impl",
    "store_paged_kv_impl",
    "swa_paged_decode_impl",
    "swa_paged_prefill_impl",
]

from mojo_opset.backends.ttx.kernels.utils import tensor_device_guard_for_triton_kernel

# NOTE(liuyuan): Automatically add guard to torch tensor for triton kernels.
tensor_device_guard_for_triton_kernel(__path__, __name__, "mlu")
