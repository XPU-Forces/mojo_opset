"""A5 binding for the Ascend Triton bilinear pos-embed interpolation kernel.

The kernel is pure data movement plus bilinear weights: it has no cube/ND2NZ path and
no architecture-specific intrinsic, so A2 and A5 share one implementation. Only the
persistent-grid width follows the device core count, which is queried at runtime.
"""

from mojo_opset.kernels.npu_a2_triton.pos_embed_interp import pos_embed_interp_infer_fwd  # noqa: F401
from mojo_opset.kernels.npu_a2_triton.pos_embed_interp import pos_embed_interp_multi_infer_fwd  # noqa: F401
