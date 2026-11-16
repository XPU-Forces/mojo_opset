"""Device-independent inference normalization semantics."""

import torch
import torch.nn.functional as F


def indexed_gate_rms_norm_scale_shift_infer_fwd(residual, gate, branch, weight, shift, scale, indices, eps):
    updated = (residual.float() + gate.index_select(0, indices).float() * branch.float()).to(torch.bfloat16)
    values = updated.float()
    variance = (values * values).mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps) * weight.float()
    modulated = normalized * (1.0 + scale.index_select(0, indices).float()) + shift.index_select(0, indices).float()
    return updated.contiguous(), modulated.to(torch.bfloat16).contiguous()


def layer_norm_infer_fwd(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps).to(x.dtype).contiguous()


def rms_norm_infer_fwd(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight, eps).to(x.dtype).contiguous()


def group_rms_norm_infer_fwd(input_groups, weight, eps):
    return [
        F.rms_norm(x, (x.shape[-1],), None if weight is None else weight[i], eps).to(x.dtype).contiguous()
        for i, x in enumerate(input_groups)
    ]


def residual_add_layer_norm_infer_fwd(x, residual, weight, bias, eps):
    # Preserve the original core operator: add in input dtype, then normalize.
    # Do not silently replace this with norm(x.float() + residual.float()).
    summed = x + residual
    return layer_norm_infer_fwd(summed, weight, bias, eps), summed


def residual_add_rms_norm_infer_fwd(x, residual, weight, eps):
    # Match MojoResidualAddRMSNorm's rounding boundary before normalization.
    summed = x + residual
    return rms_norm_infer_fwd(summed, weight, eps), summed
