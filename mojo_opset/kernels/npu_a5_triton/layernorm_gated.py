"""A5 (Ascend 950) provider: reuses the A2 Triton kernel.

The gated norm kernel uses no Cube unit, no L0/L1 buffers and no architecture specific
intrinsic, so the A2 implementation is valid on A5 as well. Keep this module as a thin
re-export so the provider table stays uniform across architectures.

The FP8 quant epilogue is the one capability the task statement scopes to 950 only; it is
expressed with ``Q.dtype.element_ty`` rather than a hardcoded float8e4nv, so the same source
compiles for whichever FP8 type the caller's output tensor carries.
"""

from mojo_opset.kernels.npu_a2_triton.layernorm_gated import layernorm_gated_infer_impl
from mojo_opset.kernels.npu_a2_triton.layernorm_gated import layernorm_gated_infer_fwd
from mojo_opset.kernels.npu_a2_triton.layernorm_gated import layer_norm_gated_fwd  # noqa: F401

__all__ = ["layernorm_gated_infer_impl", "layernorm_gated_infer_fwd"]
