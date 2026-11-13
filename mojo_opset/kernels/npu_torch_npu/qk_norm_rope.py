"""torch_npu QK RMSNorm + RoPE composition (upstream _npu_qk_norm_rope, half-split only)."""

import torch

from mojo_opset.kernels.npu_torch_npu.normalization import rms_norm_infer_fwd
from mojo_opset.kernels.npu_torch_npu.position_embedding import _rotary_mul


def qk_norm_rope_fwd(q, k, q_weight, k_weight, rope_table, eps, interleaved):
    if interleaved:
        raise NotImplementedError("torch_npu qk_norm_rope supports the half-split layout only")
    rotary_dim = rope_table.shape[-1]
    half = rotary_dim // 2
    outputs = []
    for x, weight in ((q, q_weight), (k, k_weight)):
        x_norm = rms_norm_infer_fwd(x, weight, eps)
        x_rot, x_pass = x_norm[..., :rotary_dim], x_norm[..., rotary_dim:]
        cos = rope_table[..., :half].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        sin = rope_table[..., half:].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        x_rot = _rotary_mul(x_rot.unsqueeze(0), cos, sin, None).squeeze(0)
        outputs.append(torch.cat((x_rot, x_pass), dim=-1))
    return tuple(outputs)
