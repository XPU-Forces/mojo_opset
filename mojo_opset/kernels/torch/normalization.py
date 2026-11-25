"""Device-independent inference normalization semantics."""

import torch.nn.functional as F


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
