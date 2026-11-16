"""Gated LayerNorm/RMSNorm reference migrated from SGLang.

Copyright (c) 2024, Tri Dao. Adapt from SGLang
python/sglang/kernels/ops/attention/fla/layernorm_gated.py (task attachment
test_layernorm_gated.py reference: rms_norm_ref / layer_norm_ref /
_per_token_fp8_quant_ref), which adapts
https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
"""

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP8_MIN_AMAX = 1e-10


def _gate(z, activation):
    if activation in ("swish", "silu"):
        return F.silu(z)
    if activation == "sigmoid":
        return torch.sigmoid(z)
    raise NotImplementedError(f"Unsupported activation: {activation}")


def layer_norm_gated_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias,
    z,
    eps: float,
    out,
    group_size,
    norm_before_gate: bool,
    is_rms_norm: bool,
    activation: str,
    quant_heads: int,
):
    """FP32-upcast gated normalization; returns ((q, s) | out), mean, rstd.

    mean is None for RMSNorm; mean/rstd use the kernel's group-major layout
    [ngroups * M] (index group * M + row). A 3D gate [T, H, D] is addressed by
    the kernel row mapping row -> (row // H, row % H). The FP8 epilogue mirrors
    the kernel pipeline: round the normalized output to the storage dtype
    first, then per-token amax scale and clamp.
    """
    M, N = x.shape
    group_size = N if group_size is None else group_size
    ngroups = N // group_size
    gate = None if z is None else (z.reshape(M, N) if z.ndim == 3 else z)

    values = x.float()
    gate_values = gate.float() if gate is not None else None
    if gate_values is not None and not norm_before_gate:
        values = values * _gate(gate_values, activation)

    grouped = values.reshape(M, ngroups, group_size)
    if is_rms_norm:
        mean = None
        rstd = torch.rsqrt(grouped.square().mean(dim=-1) + eps)
        normalized = grouped * rstd.unsqueeze(-1)
    else:
        mean = grouped.mean(dim=-1)
        rstd = torch.rsqrt((grouped - mean.unsqueeze(-1)).square().mean(dim=-1) + eps)
        normalized = (grouped - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)

    result = normalized.reshape(M, N) * weight.float()
    if bias is not None:
        result = result + bias.float()
    if gate_values is not None and norm_before_gate:
        result = result * _gate(gate_values, activation)
    result = result.to(x.dtype)
    if out is not None:
        out.copy_(result)
        result = out

    mean_out = None if mean is None else mean.transpose(0, 1).reshape(-1).contiguous()
    rstd_out = rstd.transpose(0, 1).reshape(-1).contiguous()
    if not quant_heads:
        return result, mean_out, rstd_out

    y = result.reshape(M // quant_heads, quant_heads * N).float()
    scale = torch.clamp(y.abs().amax(dim=-1, keepdim=True), min=FP8_MIN_AMAX) / FP8_MAX
    q = (y * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q, scale), mean_out, rstd_out
