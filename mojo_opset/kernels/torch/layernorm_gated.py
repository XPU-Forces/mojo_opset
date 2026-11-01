"""Reference gated LayerNorm / RMSNorm, written as the eager operator chain.

This is the accuracy yardstick: it composes ``torch.mean`` / ``torch.var`` style reductions
with ``torch.nn.functional.silu`` and a plain affine, all in FP32, rather than restating the
Triton kernel's algebra. The two therefore share no formula -- only the contract they are both
supposed to satisfy -- so a mistake in either one shows up as a mismatch instead of cancelling.

Statistics are accumulated in FP32 and the result is rounded once at the end, which is what the
kernel does too. ``mean``/``rstd`` come back in FP32 with the ``[ngroups * M]`` layout the
kernel writes.
"""

import os
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

_QUANT_MAX = 448.0


def _act(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation in ("swish", "silu"):
        return z * torch.sigmoid(z)
    if activation == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(f"unsupported activation {activation!r}")


def layernorm_gated_infer_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    batch_invariant: bool = False,
) -> Tuple[object, Optional[torch.Tensor], torch.Tensor]:
    m, n = x.shape
    group = n if group_size is None else group_size
    ngroups = n // group

    # Everything is promoted explicitly: torch would otherwise keep some of these reductions in
    # the input dtype, which is already a different rounding path from the kernel's FP32 ones.
    xf = x.to(torch.float32)
    gate = None
    if z is not None:
        gate = z.reshape(m, n).to(torch.float32)

    if gate is not None and not norm_before_gate:
        xf = xf * _act(gate, activation)

    xg = xf.reshape(m, ngroups, group)
    if is_rms_norm:
        mean_per_group = None
        var = xg.pow(2).mean(dim=-1, keepdim=True)
    else:
        mean_per_group = xg.mean(dim=-1, keepdim=True)
        var = (xg - mean_per_group).pow(2).mean(dim=-1, keepdim=True)
    rstd_per_group = torch.rsqrt(var + eps)

    if is_rms_norm:
        x_hat = xg * rstd_per_group
    else:
        x_hat = (xg - mean_per_group) * rstd_per_group
    y = x_hat.reshape(m, n)

    y = y * weight.to(torch.float32)
    if bias is not None:
        y = y + bias.to(torch.float32)
    if gate is not None and norm_before_gate:
        y = y * _act(gate, activation)

    # Layout contract: [ngroups * M], group-major, matching the kernel's `group * M + row`.
    if is_rms_norm:
        mean_out = None
    else:
        mean_out = mean_per_group.squeeze(-1).transpose(0, 1).reshape(-1).contiguous()
    rstd_out = rstd_per_group.squeeze(-1).transpose(0, 1).reshape(-1).contiguous()

    if quant_heads:
        if ngroups != 1:
            raise ValueError("quant_heads requires a single group")
        # Round to the output dtype before the amax, so this matches the unfused
        # per_token_group_quant_fp8 rather than measuring a value that will never be stored.
        yq = y.reshape(m, n).to(x.dtype).to(torch.float32)
        yq = yq.reshape(m // quant_heads, quant_heads * n)
        amax = yq.abs().amax(dim=-1)
        scale = torch.clamp(amax, min=1e-10) / _QUANT_MAX
        q = torch.clamp(yq / scale[:, None], -_QUANT_MAX, _QUANT_MAX).to(torch.float8_e4m3fn)
        return (q, scale.reshape(-1, 1)), mean_out, rstd_out

    return y.to(x.dtype), mean_out, rstd_out


def _batch_invariant_flag() -> bool:
    """``LN_GATED_BATCH_INVARIANT=1`` pins the row tile so results do not depend on M."""
    return os.environ.get("LN_GATED_BATCH_INVARIANT", "").strip() not in ("", "0", "false", "False")


def layer_norm_gated_fwd(
    x,
    weight,
    bias,
    z,
    eps,
    out,
    group_size,
    norm_before_gate,
    is_rms_norm,
    activation,
    quant_heads,
):
    """Public op id binding; keeps the upstream positional order and forwards to the reference."""
    return layernorm_gated_infer_fwd(
        x,
        weight,
        bias,
        float(eps),
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=int(quant_heads),
        batch_invariant=_batch_invariant_flag(),
    )
