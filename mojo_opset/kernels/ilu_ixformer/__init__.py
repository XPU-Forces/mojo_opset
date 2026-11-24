"""ixFormer kernel provider for Iluvatar targets."""

OPS = {
    "layer_norm_infer": "normalization",
    "rms_norm_infer": "normalization",
    "group_rms_norm_infer": "normalization",
    "residual_add_layer_norm_infer": "normalization",
    "residual_add_rms_norm_infer": "normalization",
}
