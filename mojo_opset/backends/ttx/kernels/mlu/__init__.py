from .layernorm import layernorm_infer_impl
from .layernorm import layernorm_bwd_impl
from .layernorm import layernorm_fwd_impl
from .swa import swa_paged_decode_impl
from .swa import swa_paged_prefill_impl
from .group_rmsnorm import group_rmsnorm_impl
from .fa_paged_decode import paged_attention_decode_impl
from .fa_paged_prefill import paged_attention_prefill_impl

__all__ = [
    "layernorm_infer_impl",
    "layernorm_bwd_impl",
    "layernorm_fwd_impl",
    "swa_paged_prefill_impl",
    "swa_paged_decode_impl",
    "group_rmsnorm_impl",
    "paged_attention_decode_impl",
    "paged_attention_prefill_impl",
]

from mojo_opset.backends.ttx.kernels.utils import tensor_device_guard_for_triton_kernel

# NOTE(liuyuan): Automatically add guard to torch tensor for triton kernels.
tensor_device_guard_for_triton_kernel(__path__, __name__, "mlu")
