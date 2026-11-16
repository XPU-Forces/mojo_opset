"""Device-independent reference semantics for fused LayerNorm/RMSNorm + gate."""

import torch


def _activation(g, activation):
    if activation in ("swish", "silu"):
        return g * torch.sigmoid(g)
    if activation == "sigmoid":
        return torch.sigmoid(g)
    raise ValueError(f"Unsupported activation: {activation}")


def _activation_grad(g, activation):
    sigmoid = torch.sigmoid(g)
    if activation in ("swish", "silu"):
        return sigmoid * (1 + g * (1 - sigmoid))
    if activation == "sigmoid":
        return sigmoid * (1 - sigmoid)
    raise ValueError(f"Unsupported activation: {activation}")


def _compute_dtype(x):
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


def _norm_gated_fwd(
    x,
    g,
    weight,
    bias,
    activation,
    eps,
    residual,
    out_dtype,
    residual_dtype,
    is_rms_norm,
):
    compute = _compute_dtype(x)
    values = x.to(compute)
    if residual is not None:
        values = values + residual.to(compute)
        residual_out = values.to(residual_dtype if residual_dtype is not None else x.dtype)
    elif residual_dtype is not None and residual_dtype != x.dtype:
        residual_out = values.to(residual_dtype)
    else:
        residual_out = None

    if is_rms_norm:
        mean = None
        centered = values
        variance = values.pow(2).mean(dim=-1, keepdim=True)
    else:
        mean = values.mean(dim=-1, keepdim=True)
        centered = values - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normalized = centered * rstd

    y = normalized * weight.to(compute) if weight is not None else normalized
    if bias is not None:
        y = y + bias.to(compute)
    y = y * _activation(g.to(compute), activation)
    y = y.to(out_dtype if out_dtype is not None else x.dtype)
    return (
        y,
        None if mean is None else mean.squeeze(-1),
        rstd.squeeze(-1),
        residual_out if residual_out is not None else x,
    )


def _norm_gated_bwd(
    dy,
    x,
    g,
    weight,
    bias,
    activation,
    eps,
    mean,
    rstd,
    dresidual,
    has_residual,
    is_rms_norm,
    x_dtype,
    recompute_output,
):
    del eps
    compute = _compute_dtype(x)
    values = x.to(compute)
    grad = dy.to(compute)
    gate_input = g.to(compute)
    rstd_f = rstd.to(compute).unsqueeze(-1)
    if is_rms_norm:
        normalized = values * rstd_f
    else:
        normalized = (values - mean.to(compute).unsqueeze(-1)) * rstd_f

    gate = _activation(gate_input, activation)
    dgate = _activation_grad(gate_input, activation)

    pre_gate = normalized * weight.to(compute) if weight is not None else normalized
    if bias is not None:
        pre_gate = pre_gate + bias.to(compute)

    dg = grad * pre_gate * dgate
    grad_pre = grad * gate

    if weight is not None:
        dw = (grad_pre * normalized).sum(dim=0).to(weight.dtype)
        weighted = grad_pre * weight.to(compute)
    else:
        dw = None
        weighted = grad_pre
    db = grad_pre.sum(dim=0).to(bias.dtype) if bias is not None else None

    if is_rms_norm:
        c1 = (normalized * weighted).mean(dim=-1, keepdim=True)
        dx = (weighted - normalized * c1) * rstd_f
    else:
        c1 = (normalized * weighted).mean(dim=-1, keepdim=True)
        c2 = weighted.mean(dim=-1, keepdim=True)
        dx = (weighted - normalized * c1 - c2) * rstd_f

    if dresidual is not None:
        dx = dx + dresidual.to(compute)

    dtype = x_dtype if x_dtype is not None else x.dtype
    dx = dx.to(dtype)
    dg = dg.to(dtype)
    dresidual_in = dx if has_residual else None
    if recompute_output:
        return dx, dg, dw, db, dresidual_in, (pre_gate * gate).to(dy.dtype)
    return dx, dg, dw, db, dresidual_in


def layer_norm_gated_fwd(
    x,
    g,
    weight,
    bias,
    activation="swish",
    eps=1e-5,
    residual=None,
    out_dtype=None,
    residual_dtype=None,
    is_rms_norm=False,
):
    return _norm_gated_fwd(
        x, g, weight, bias, activation, eps, residual, out_dtype, residual_dtype, is_rms_norm
    )


def layer_norm_gated_bwd(
    dy,
    x,
    g,
    weight,
    bias,
    activation="swish",
    eps=1e-5,
    mean=None,
    rstd=None,
    dresidual=None,
    has_residual=False,
    is_rms_norm=False,
    x_dtype=None,
    recompute_output=False,
):
    return _norm_gated_bwd(
        dy, x, g, weight, bias, activation, eps, mean, rstd, dresidual,
        has_residual, is_rms_norm, x_dtype, recompute_output,
    )


def rms_norm_gated_fwd(
    x,
    g,
    weight,
    bias,
    activation="swish",
    eps=1e-5,
    residual=None,
    out_dtype=None,
    residual_dtype=None,
):
    return _norm_gated_fwd(
        x, g, weight, bias, activation, eps, residual, out_dtype, residual_dtype, True
    )


def rms_norm_gated_bwd(
    dy,
    x,
    g,
    weight,
    bias,
    activation="swish",
    eps=1e-5,
    mean=None,
    rstd=None,
    dresidual=None,
    has_residual=False,
    x_dtype=None,
    recompute_output=False,
):
    return _norm_gated_bwd(
        dy, x, g, weight, bias, activation, eps, mean, rstd, dresidual,
        has_residual, True, x_dtype, recompute_output,
    )
