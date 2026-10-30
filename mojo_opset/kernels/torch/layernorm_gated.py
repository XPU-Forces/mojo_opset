"""FP32 reference for fused gated normalization, including the FP8 epilogue.

The FP8 quantisation mirrors ``per_token_group_quant_fp8`` from torch_npu, which
the task requires the fused epilogue to match bit for bit.  The reference
follows the public implementation of the same community task,
``brandfucker/mojo_opset#516`` (Apache-2.0), for the normalisation and gate
semantics.
"""

import torch


def _gate(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(z)
    return z * torch.sigmoid(z)


def _quant_per_token(y: torch.Tensor, quant_heads: int):
    """Match the unfused ``per_token_group_quant_fp8`` on the rounded output."""

    grouped = y.reshape(y.shape[0] // quant_heads, quant_heads * y.shape[1]).float()
    amax = torch.amax(torch.abs(grouped), dim=-1, keepdim=True)
    scale = torch.maximum(amax, torch.full_like(amax, 1e-10)) / 448.0
    quantized = torch.clamp(grouped * (1.0 / scale), -448.0, 448.0).to(torch.float8_e4m3fn)
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
    ngroups = N // group_size
    dtype = x.dtype

    xg = x.float().reshape(M, ngroups, group_size)
    wg = weight.float().reshape(ngroups, group_size)
    bg = bias.float().reshape(ngroups, group_size) if bias is not None else None
    zg = None
    if z is not None:
        zg = _gate(z.float(), activation).reshape(M, ngroups, group_size)
        if not norm_before_gate:
            xg = xg * zg

    if is_rms_norm:
        mean = None
        rstd = 1 / torch.sqrt(xg.square().mean(dim=-1) + eps)
        xhat = xg * rstd.unsqueeze(-1)
    else:
        mean = xg.mean(dim=-1)
        var = (xg - mean.unsqueeze(-1)).square().mean(dim=-1)
        rstd = 1 / torch.sqrt(var + eps)
        xhat = (xg - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)

    y = xhat * wg
    if bg is not None:
        y = y + bg
    if zg is not None and norm_before_gate:
        y = y * zg
    y = y.reshape(M, N).to(dtype)

    mean = mean.transpose(0, 1).contiguous().reshape(-1) if mean is not None else None
    rstd = rstd.transpose(0, 1).contiguous().reshape(-1)

    if quant_heads:
        return _quant_per_token(y, quant_heads), mean, rstd
    if out is None:
        out = torch.empty_like(x)
    out.copy_(y)
    return out, mean, rstd
