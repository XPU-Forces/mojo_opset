"""Triton kernels for Iluvatar targets; concrete files are loaded lazily."""

OPS = {
    "layer_norm_infer": "layernorm",
    "rms_norm_infer": "rmsnorm",
    "group_rms_norm_infer": "group_rmsnorm",
    "residual_add_layer_norm_infer": "fused_add_layernorm",
    "residual_add_rms_norm_infer": "fused_add_rmsnorm",
}
