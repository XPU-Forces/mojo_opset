"""Triton kernels for Cambricon targets; concrete files are loaded lazily."""

OPS = {
    "layer_norm_infer": "layernorm",
    "group_rms_norm_infer": "group_rmsnorm",
}
