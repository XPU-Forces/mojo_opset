"""Gated LayerNorm / RMSNorm forward (``layer_norm_gated``) for Ascend A2/A3.

Tiling constants are derived from the detected platform (UB capacity and vector
core count), so this binding shares exactly one implementation with the A5 one;
the two packages cannot drift apart.
"""

from mojo_opset.kernels._layernorm_gated_impl import layer_norm_gated_fwd

__all__ = ["layer_norm_gated_fwd"]
