"""FP32 oracle for gated normalization and its per-token FP8 epilogue.

The semantics are those of SGLang v0.5.22
``python/sglang/kernels/ops/attention/fla/layernorm_gated.py``: the whole
computation runs in FP32 over ``[M, ngroups, group_size]`` views, the result is
rounded to the output dtype once at the end, and ``quant_heads`` quantises that
rounded value the way ``per_token_group_quant_fp8`` does.  Keeping the arithmetic
in this order is what makes the module usable as a bit-exact oracle for the
Triton kernel, so the expressions below are deliberately spelled out rather than
folded into equivalent-looking ones.
"""

import torch

FP8_MAX = 448.0
AMAX_FLOOR = 1e-10


def _activate(gate: torch.Tensor, name: str) -> torch.Tensor:
    """SGLang's gate: plain sigmoid, and ``z * sigmoid(z)`` for swish/silu."""

    if name == "sigmoid":
        return torch.sigmoid(gate)
    return gate * torch.sigmoid(gate)


def _as_groups(tensor: torch.Tensor, rows: int, ngroups: int, width: int) -> torch.Tensor:
    """FP32 ``[rows, ngroups, width]`` view of a gate or input of any rank.

    A gate may arrive as ``[M, N]`` or as a strided ``[T, H, D]`` view; both tile
    the same ``rows * N`` values, so both flatten the same way.
    """

    return tensor.float().reshape(rows, ngroups, width)


def _stats(values: torch.Tensor, eps, is_rms_norm: bool):
    """Mean (or ``None``) and reciprocal standard deviation along the group axis."""

    if is_rms_norm:
        return None, 1 / torch.sqrt(values.square().mean(dim=-1) + eps)
    mean = values.mean(dim=-1)
    centered = values - mean.unsqueeze(-1)
    return mean, 1 / torch.sqrt(centered.square().mean(dim=-1) + eps)


def _fp8_per_token(y: torch.Tensor, quant_heads: int):
    """One e4m3 scale per ``quant_heads``-row block, as the unfused epilogue does."""

    rows, width = y.shape
    blocks = y.reshape(rows // quant_heads, quant_heads * width).float()
    amax = torch.amax(torch.abs(blocks), dim=-1, keepdim=True)
    scale = torch.maximum(amax, torch.full_like(amax, AMAX_FLOOR)) / FP8_MAX
    quantized = torch.clamp(blocks * (1.0 / scale), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
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
    rows, cols = x.shape
    width = cols if group_size is None else group_size
    ngroups = cols // width

    values = _as_groups(x, rows, ngroups, width)
    gate = None if z is None else _activate(_as_groups(z, rows, ngroups, width), activation)
    if gate is not None and not norm_before_gate:
        values = values * gate

    mean, rstd = _stats(values, eps, is_rms_norm)
    normed = values * rstd.unsqueeze(-1) if is_rms_norm else (values - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)
    result = normed * weight.float().reshape(ngroups, width)
    if bias is not None:
        result = result + bias.float().reshape(ngroups, width)
    if gate is not None and norm_before_gate:
        result = result * gate
    rounded = result.reshape(rows, cols).to(x.dtype)

    if mean is not None:
        mean = mean.transpose(0, 1).contiguous().reshape(-1)
    rstd = rstd.transpose(0, 1).contiguous().reshape(-1)

    if quant_heads:
        return _fp8_per_token(rounded, quant_heads), mean, rstd
    if out is None:
        out = torch.empty_like(x)
    out.copy_(rounded)
    return out, mean, rstd
