"""A5 (Ascend 950) provider: reuses the A2 Triton kernel.

The QK RMSNorm + RoPE kernel uses no Cube unit, no L0/L1 buffers and no architecture specific
intrinsic, so the A2 implementation is valid on A5 as well. Keep this module as a thin
re-export so the provider table stays uniform across architectures.
"""

from mojo_opset.kernels.npu_a2_triton.qk_norm_rope import qk_norm_rope_infer_fwd

__all__ = ["qk_norm_rope_infer_fwd"]
