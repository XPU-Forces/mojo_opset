from .convolution import causal_conv1d_update_bdt_fwd
from .diffution_attention import diffusion_attention_bwd_impl
from .diffution_attention import diffusion_attention_fwd_impl
from .flash_attention import paged_attention_decode_impl
from .flash_attention import paged_attention_prefill_impl
from .fused_linear_cross_entropy import fused_linear_cross_entropy_1d_bwd_impl
from .fused_linear_cross_entropy import fused_linear_cross_entropy_1d_fwd_impl
from .fused_linear_cross_entropy import fused_linear_cross_entropy_bwd_impl
from .fused_linear_cross_entropy import fused_linear_cross_entropy_fwd_impl
from .gelu import gelu_bwd_impl
from .gelu import gelu_fwd_impl
from .gemm import matmul_impl
from .group_gemm import k_grouped_matmul_impl
from .group_gemm import m_grouped_matmul_impl
from .kv_cache import store_paged_kv_impl
from .quant import quant_int8_infer_impl
from .rmsnorm import rmsnorm_bwd_impl
from .rmsnorm import rmsnorm_fwd_impl
from .rmsnorm import rmsnorm_infer_impl
from .rope import rope_bwd_impl
from .rope import rope_fwd_impl
from .sdpa import sdpa_bwd_impl
from .sdpa import sdpa_fwd_impl
from .sdpa import sdpa_infer_impl
from .silu import silu_bwd_impl
from .silu import silu_fwd_impl
from .store_lowrank import store_label_cache_fwd_impl
from .swiglu import swiglu_bwd_impl
from .swiglu import swiglu_fwd_impl

__all__ = [
    "causal_conv1d_update_bdt_fwd",
    "paged_attention_decode_impl",
    "paged_attention_prefill_impl",
    "fused_linear_cross_entropy_bwd_impl",
    "fused_linear_cross_entropy_fwd_impl",
    "fused_linear_cross_entropy_1d_bwd_impl",
    "fused_linear_cross_entropy_1d_fwd_impl",
    "gelu_bwd_impl",
    "gelu_fwd_impl",
    "quant_int8_impl",
    "rmsnorm_bwd_impl",
    "rmsnorm_fwd_impl",
    "rmsnorm_infer_impl",
    "rope_bwd_impl",
    "rope_fwd_impl",
    "silu_bwd_impl",
    "silu_fwd_impl",
    "swiglu_bwd_impl",
    "swiglu_fwd_impl",
    "sdpa_infer_impl",
    "sdpa_fwd_impl",
    "sdpa_bwd_impl",
    "diffusion_attention_fwd_impl",
    "diffusion_attention_bwd_impl",
    "m_grouped_matmul_impl",
    "k_grouped_matmul_impl",
    "store_paged_kv_impl",
    "store_label_cache_fwd_impl",
]

from mojo_opset.backends.ttx.kernels.utils import tensor_device_guard_for_triton_kernel
# NOTE(liuyuan): Automatically add guard to torch tensor for triton kernels.
tensor_device_guard_for_triton_kernel(__path__, __name__)
