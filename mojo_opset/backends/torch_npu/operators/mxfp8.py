"""
Copyright (c) 2026 Bytedance. All Rights Reserved.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_MX_BLOCK_SIZE = 32
_MX_E8M0_DTYPE = None


def _mx_e8m0_dtype():
    global _MX_E8M0_DTYPE
    if _MX_E8M0_DTYPE is None:
        import torch_npu

        _MX_E8M0_DTYPE = torch_npu.float8_e8m0fnu
    return _MX_E8M0_DTYPE


def _pad_mx_scale_cols_for_gmm(scale: torch.Tensor, *, in_features: int) -> torch.Tensor:
    """Pad checkpoint [..., ceil(in/32)] scales to ceil(in/64)*2 pair layout."""
    nb32 = (in_features + _MX_BLOCK_SIZE - 1) // _MX_BLOCK_SIZE
    nb64 = (in_features + 63) // 64
    need = nb64 * 2
    scale = scale[..., :nb32]
    if scale.shape[-1] < need:
        scale = F.pad(scale, (0, need - scale.shape[-1]))
    return scale.contiguous()


def prepare_mx_weight_scale_for_quant_matmul(
    per_group_scales: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Checkpoint scale [out, ceil(in/32)] -> [ceil(in/64), out, 2]."""
    nb64 = (in_features + 63) // 64
    scale = per_group_scales
    if scale.dim() == 3 and scale.shape[-2] == 1:
        scale = scale.squeeze(-2)
    if scale.shape[0] != out_features:
        raise ValueError(f"scale out dim {scale.shape[0]} != out_features {out_features}")
    scale = _pad_mx_scale_cols_for_gmm(scale, in_features=in_features)
    return scale.reshape(out_features, nb64, 2).transpose(0, 1).contiguous()


def prepare_mx_expert_scale_for_grouped_matmul(
    per_group_scales: torch.Tensor,
    *,
    in_features: int,
) -> torch.Tensor:
    """Expert scale [E, n, ceil(k/32)] -> [E, ceil(k/64), n, 2]."""
    nb64 = (in_features + 63) // 64
    scale = per_group_scales
    if scale.dim() == 4:
        if scale.shape[-2] == 1:
            scale = scale.squeeze(-2)
        elif scale.shape[-1] == 2:
            scale = scale.reshape(scale.shape[0], scale.shape[1], -1)
    e, n = scale.shape[0], scale.shape[1]
    scale = _pad_mx_scale_cols_for_gmm(scale, in_features=in_features)
    return scale.reshape(e, n, nb64, 2).transpose(1, 2)


def decode_e8m0_scale_u8(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode uint8 E8M0 (biased exp, value=2**(raw-127)) to float32."""
    s = scale_u8.to(dtype=torch.int32)
    return torch.where(
        s == 0,
        torch.zeros((), device=scale_u8.device, dtype=torch.float32),
        torch.pow(
            torch.tensor(2.0, device=scale_u8.device, dtype=torch.float32),
            (s - 127).to(torch.float32),
        ),
    )


def mx_dequant_lastdim(
    fp8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    block_size: int = _MX_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an MXFP8 tensor along the last dimension."""
    mx = scale_u8
    if mx.dim() == fp8.dim() + 1:
        if mx.shape[-2] == 1:
            mx = mx.squeeze(-2)
        elif mx.shape[-1] == 2:
            mx = mx.reshape(*mx.shape[:-2], -1)
    elif mx.dim() == fp8.dim() and mx.shape[-2] == 1:
        mx = mx.squeeze(-2)

    k = fp8.shape[-1]
    nb = (k + block_size - 1) // block_size
    mx = mx[..., :nb].contiguous()
    scales = decode_e8m0_scale_u8(mx)
    scales = scales.repeat_interleave(block_size, dim=-1)[..., :k]
    return (fp8.float() * scales).to(dtype=out_dtype)


def mx_dequant_weight(
    weight_fp8: torch.Tensor,
    per_group_scales: torch.Tensor,
    *,
    block_size: int = _MX_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP8 weight [..., in_features] using per-row block scales."""
    return mx_dequant_lastdim(
        weight_fp8,
        per_group_scales,
        block_size=block_size,
        out_dtype=out_dtype,
    )


def mx_dynamic_quant_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic MX quantize activation to FP8 E4M3 plus uint8 E8M0 scale."""
    import torch_npu

    return torch_npu.npu_dynamic_mx_quant(
        x,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        block_size=_MX_BLOCK_SIZE,
        scale_alg=0,
        dst_type_max=0.0,
    )


def mxfp8_quant_matmul_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    per_group_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """MXFP8 linear via fused npu_quant_matmul."""
    import torch_npu

    if weight_fp8.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight_fp8.shape)}")
    out_features, in_features = weight_fp8.shape
    x_fp8, x_scale = mx_dynamic_quant_activation(x)
    w_kn = weight_fp8.t().contiguous()
    scale_kno = prepare_mx_weight_scale_for_quant_matmul(
        per_group_scales,
        out_features=out_features,
        in_features=in_features,
    )
    e8 = _mx_e8m0_dtype()
    kwargs = {
        "pertoken_scale": x_scale.contiguous(),
        "pertoken_scale_dtype": e8,
        "output_dtype": x.dtype,
        "group_sizes": [1, 1, _MX_BLOCK_SIZE],
        "x1_dtype": torch.float8_e4m3fn,
        "x2_dtype": torch.float8_e4m3fn,
        "scale_dtype": e8,
    }
    if bias is not None:
        kwargs["bias"] = bias
    return torch_npu.npu_quant_matmul(x_fp8, w_kn, scale_kno, **kwargs)


def mxfp8_linear_dequant_fallback(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    per_group_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fallback path: MX quant activation, dequant activation/weight, then BF16 linear."""
    x_fp8, x_scale = mx_dynamic_quant_activation(x)
    x_q = mx_dequant_lastdim(x_fp8, x_scale, out_dtype=x.dtype)
    w_q = mx_dequant_weight(weight_fp8, per_group_scales, out_dtype=x.dtype)
    if x_q.device.type == "npu":
        try:
            import torch_npu

            return torch_npu.npu_linear(x_q, w_q, bias)
        except Exception:
            pass
    return F.linear(x_q, w_q, bias)


def mxfp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    per_group_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefer fused MXFP8 npu_quant_matmul, fallback to explicit dequant + BF16 linear."""
    if x.device.type != "npu":
        return mxfp8_linear_dequant_fallback(x, weight_fp8, per_group_scales, bias)
    try:
        return mxfp8_quant_matmul_linear(x, weight_fp8, per_group_scales, bias)
    except Exception as exc:
        logger.warning("mxfp8_quant_matmul_linear failed (%s); fallback to dequant path", exc)
        return mxfp8_linear_dequant_fallback(x, weight_fp8, per_group_scales, bias)
