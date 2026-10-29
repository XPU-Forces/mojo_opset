"""FP32 reference and executable torch path for gated normalization."""

import torch
import torch.nn.functional as F


def _activation(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation in ("swish", "silu"):
        return F.silu(z)
    if activation == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(f"unsupported activation: {activation!r}")


def _expanded_gate(z: torch.Tensor, M: int, N: int) -> torch.Tensor:
    if z.ndim == 3:
        assert z.shape[0] * z.shape[1] == M and z.shape[2] == N
        return z.reshape(M, N)
    assert z.shape == (M, N)
    return z


def _quantize(y: torch.Tensor, quant_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, width = y.shape
    grouped = y.reshape(tokens // quant_heads, quant_heads * width)
    amax = torch.amax(torch.abs(grouped), dim=-1, keepdim=True)
    scale = torch.maximum(amax, torch.full_like(amax, 1e-10)) / 448.0
    quantized = torch.clamp(grouped * (1.0 / scale), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def layernorm_gated_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
    quant_heads: int = 0,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    if N % group_size:
        raise ValueError("N must be divisible by group_size")
    ngroups = N // group_size

    x_f = x.float()
    z_f = None if z is None else _expanded_gate(z, M, N).float()
    if z_f is not None and not norm_before_gate:
        x_f = x_f * _activation(z_f, activation)

    grouped = x_f.reshape(M, ngroups, group_size)
    if is_rms_norm:
        mean = None
        rstd = torch.rsqrt(grouped.square().mean(dim=-1) + eps)
    else:
        mean = grouped.mean(dim=-1)
        centered = grouped - mean.unsqueeze(-1)
        rstd = torch.rsqrt(centered.square().mean(dim=-1) + eps)

    normalized = grouped * rstd.unsqueeze(-1)
    if not is_rms_norm:
        normalized = (grouped - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
    normalized = normalized.reshape(M, N)
    result = normalized * weight.float().unsqueeze(0)
    if bias is not None:
        result = result + bias.float().unsqueeze(0)
    if z_f is not None and norm_before_gate:
        result = result * _activation(z_f, activation)

    if out is None:
        out = torch.empty_like(x)
    out.copy_(result.to(out.dtype))

    mean_out = None if mean is None else mean.transpose(0, 1).contiguous().reshape(-1)
    rstd_out = rstd.transpose(0, 1).contiguous().reshape(-1)

    if quant_heads:
        if ngroups != 1 or M % quant_heads:
            raise ValueError("quant_heads requires ngroups=1 and M divisible by quant_heads")
        rounded = out.float()
        q, s = _quantize(rounded, quant_heads)
        return (q, s), mean_out, rstd_out
    return out, mean_out, rstd_out
