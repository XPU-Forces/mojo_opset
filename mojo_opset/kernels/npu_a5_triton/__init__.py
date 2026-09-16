"""A5 shared bindings and explicit SKU selections; no runtime fallback."""

from mojo_opset.utils._exports import extend_bindings as _extend_bindings

OPS = {
    "swa_infer": "swa",
    "moe_dynamic_quant": "quant",
    "flex_attention": "flex_attention",
    "rotary_embedding": "rope",
    "vision_rotary_embedding2d": "vision_rope",
    "mrope": "mrope",
    "sdpa_infer": "sdpa",
    "diffusion_attention": "diffution_attention",
    "n_gram_prefill": "over_encoding.n_gram",
    "n_gram_decode": "over_encoding.n_gram",
    "embedding_nf4_dequant": "over_encoding.embedding",
    "over_encoding_decode": "over_encoding.fused_over_encoding",
    "lightning_indexer": "lightning_indexer",
    "store_lowrank": "store_lowrank",
    "store_paged_kv_cache": "kv_cache",
    "apply_rope_infer": "rope",
    "apply_vision_rope2d_infer": "vision_rope",
    "paged_decode_gqa_infer": "flash_attention",
    "paged_prefill_gqa_infer": "flash_attention",
    "paged_prefill_swa_infer": "swa",
    "paged_decode_swa_infer": "swa",
    "top_k_sampling": "sample",
    "top_p_sampling": "sample",
    "top_p_filter": "sample",
    "reject_sampling": "sample",
    "join_prob_reject_sampling": "sample",
    "apply_penalties_temperature": "sample",
    "group_gemm": "group_gemm",
    "quant_gemm": "int8_gemm",
    "static_quant": "quant",
    "dequant": "quant",
    "dynamic_quant": "quant",
    "linear_cross_entropy": "fused_linear_cross_entropy",
    "causal_conv1d_update_state_infer": "convolution",
    "layer_norm_infer": "layernorm",
    "rms_norm_infer": "rmsnorm",
    "group_rms_norm_infer": "group_rmsnorm",
    "residual_add_layer_norm_infer": "fused_add_layernorm",
    "residual_add_rms_norm_infer": "fused_add_rmsnorm",
    "causal_conv1d": "convolution",
    "flash_attention": "swa",
    "flash_attention_infer": "swa",
    "silu": "silu",
    "gelu": "gelu",
    "swiglu": "swiglu",
    "rms_norm": "rmsnorm",
    "swa": "swa",
    "apply_rope": "rope",
}

# Add only differences once specialized wrappers exist, for example:
# OVERRIDES = {"npu.a5.950pr": {"flash_attention": "sku_950pr.attention"}}


_extend_bindings(__name__, OPS)
del _extend_bindings
