"""FlexAttention v2 shares the mathematical reference with the original API."""

from .flex_attention import flex_attention_bwd as flex_attention_v2_bwd
from .flex_attention import flex_attention_fwd as flex_attention_v2_fwd

__all__ = ["flex_attention_v2_fwd", "flex_attention_v2_bwd"]
