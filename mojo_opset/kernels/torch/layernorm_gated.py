"""Device-independent gated normalization semantics (reference path)."""

import torch


def _gate(z, activation):
    if activation in ("swish", "silu"):
        return z * torch.sigmoid(z)
    if activation == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(f"Unsupported activation: {activation}")


def layer_norm_gated_fwd(
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
    # Two call conventions coexist upstream: this package's functions layer passes
    # (x, w, b, eps, z=...), the reference layer proposed in mojo_opset#525 passes
    # (x, w, b, z, eps, ...) positionally. Normalise so either dispatcher works.
    if not isinstance(eps, (int, float)) or isinstance(z, (int, float)):
        eps, z = z, eps
    M, N = x.shape
    group_size = N if group_size is None else group_size
    assert N % group_size == 0
    ngroups = N // group_size

    z2d = None
    if z is not None:
        z2d = z.reshape(M, N) if z.ndim == 3 else z
        assert z2d.shape == (M, N)

    values = x.float()
    if z2d is not None and not norm_before_gate:
        values = values * _gate(z2d.float(), activation)

    grouped = values.reshape(M, ngroups, group_size)
    if is_rms_norm:
        mean = None
        var = grouped.square().mean(dim=-1)
    else:
        mean = grouped.mean(dim=-1)
        var = (grouped - mean.unsqueeze(-1)).square().mean(dim=-1)
    rstd = torch.rsqrt(var + eps)

    # Stats are written group-major: element ``group * M + row``.
    mean_flat = None if mean is None else mean.transpose(0, 1).reshape(-1)
    rstd_flat = rstd.transpose(0, 1).reshape(-1)

    if is_rms_norm:
        x_hat = grouped * rstd.unsqueeze(-1)
    else:
        x_hat = (grouped - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)

    w = weight.float().reshape(ngroups, group_size)
    y = x_hat * w.unsqueeze(0)
    if bias is not None:
        y = y + bias.float().reshape(ngroups, group_size).unsqueeze(0)

    if z2d is not None and norm_before_gate:
        y = y * _gate(z2d.float(), activation).reshape(M, ngroups, group_size)

    y = y.reshape(M, N)

    if quant_heads:
        assert ngroups == 1 and M % quant_heads == 0
        # Round to the activation dtype first; the amax then matches the fused path.
        rounded = y.to(x.dtype).float().reshape(M // quant_heads, quant_heads * N)
        amax = rounded.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(amax, min=1e-10) / 448.0
        q = torch.clamp(rounded * (1.0 / scale), -448.0, 448.0).to(torch.float8_e4m3fn)
        return (q, scale), mean_flat, rstd_flat

    y = y.to(x.dtype)
    if out is not None:
        out.copy_(y)
        y = out
    return y, mean_flat, rstd_flat
