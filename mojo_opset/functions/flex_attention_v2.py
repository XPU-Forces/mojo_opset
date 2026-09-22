"""FlexAttention v2 interface with explicit grouped-query attention support."""

from .flex_attention import FlexAttentionFunction
from .flex_attention import _validate


def flex_attention_v2(
    q,
    k,
    v,
    score_mod=None,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    return_lse=False,
    *,
    implementation=None,
    **kwargs,
):
    """Torch-compatible call shape with explicit enable_gqa; score_mod/LSE are unsupported."""
    if score_mod is not None or return_lse:
        raise NotImplementedError("FlexAttention v2 does not support score_mod or return_lse")
    _validate(q, k, v, block_mask)
    if not enable_gqa and q.shape[1] != k.shape[1]:
        raise ValueError("enable_gqa=False requires equal Q/K/V head counts")
    return FlexAttentionFunction.apply(q, k, v, block_mask, scale, implementation, "flex_attention_v2")
