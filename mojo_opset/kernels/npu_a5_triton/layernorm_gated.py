"""Gated LayerNorm / RMSNorm forward (``layer_norm_gated``) for Ascend 950.

950 exposes a larger UB (256 KiB) and more vector cores; both are detected at
runtime by the shared implementation, so this binding is intentionally a thin
re-export of the A2 module's kernel.
"""

from mojo_opset.kernels._layernorm_gated_impl import layer_norm_gated_fwd

__all__ = ["layer_norm_gated_fwd"]
