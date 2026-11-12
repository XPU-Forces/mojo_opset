"""Eager FP32 reference for the fused Q/KV RMSNorm and MXFP8 Q quantisation.

The arithmetic order mirrors the vllm kernels (``common/ops/fused_qk_rmsnorm.py``
and ``deepseek_v41/common/ops/query_quant.py``): one FP32 reduction per row, one
cast back to the activation dtype, then - for the quantising entry - the MXFP8
group amax / UE8M0 exponent chain and the F8_128x4 swizzle.
"""

import torch

MXFP8_MAX = 448.0
GROUP = 32


def _rms_norm_f32(x, weight, eps):
    xf = x.to(torch.float32)
    variance = xf.pow(2).sum(dim=-1, keepdim=True) / x.shape[-1]
    return xf * torch.rsqrt(variance + eps) * weight.to(torch.float32)


def _rms_norm(x, weight, eps):
    return _rms_norm_f32(x, weight, eps).to(x.dtype)


def _mxfp8_values(y, dtype):
    """Return (FP32 quantised values, UE8M0 scale bytes) for one activation row block."""
    y = y.to(dtype).to(torch.float32)  # keep the norm's activation-dtype rounding
    tokens, width = y.shape
    grouped = y.view(tokens, width // GROUP, GROUP)
    amax = grouped.abs().amax(dim=-1)
    normalized = amax * (1.0 / MXFP8_MAX)
    # Exponent field plus mantissa carry, matching the kernel's uint32 bit chain.
    # int64 carries the value because the NPU backend has no uint32 comparison.
    bits = normalized.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    scale = torch.minimum(exponent + bump.to(torch.int64), torch.tensor(254, device=y.device))
    scale = torch.where(normalized <= 0, torch.zeros_like(scale), scale)
    # inv = 2^(127 - scale), with the zero-scale case mapped to 0.
    inv_bits = torch.where(scale == 0, torch.zeros_like(scale), (254 - scale) << 23)
    inv = inv_bits.to(torch.int32).view(torch.float32)
    return (grouped * inv.unsqueeze(-1)).reshape(tokens, width), scale


def _scale_bytes(scale, tokens, width):
    """Scatter UE8M0 bytes into the F8_128x4 layout (rows >= tokens stay zero)."""
    device = scale.device
    groups = width // GROUP
    groups4 = (groups + 3) // 4
    total = ((tokens + 127) // 128) * 128 * groups4 * 4
    out = torch.zeros(total, dtype=torch.uint8, device=device)
    if tokens == 0:
        return out
    rows = torch.arange(tokens, device=device)[:, None]
    group = torch.arange(groups, device=device)[None, :]
    offset = (
        (rows // 128 * groups4 + group // 4) * 512
        + rows % 32 * 16
        + rows % 128 // 32 * 4
        + group % 4
    )
    out[offset.reshape(-1)] = scale.to(torch.uint8).reshape(-1)
    return out


def fused_q_kv_rmsnorm_fwd(qr, kv, q_weight, kv_weight, eps):
    return _rms_norm(qr, q_weight, eps), _rms_norm(kv, kv_weight, eps)


def fused_q_kv_rmsnorm_quant_fwd(qr, kv, q_weight, kv_weight, eps):
    # The Q norm stays FP32 until the MXFP8 chain re-rounds it through the
    # activation dtype (matching the kernel's store/load boundary).
    normed = _rms_norm_f32(qr, q_weight, eps)
    values, scale = _mxfp8_values(normed, qr.dtype)
    return (
        values.to(torch.float8_e4m3fn),
        _scale_bytes(scale, qr.shape[0], qr.shape[1]),
        _rms_norm(kv, kv_weight, eps),
    )
