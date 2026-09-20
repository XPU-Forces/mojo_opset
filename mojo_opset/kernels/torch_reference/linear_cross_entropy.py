"""Original mojo master reference, preserving its existing option behavior.

Source: 7defc828, mojo_opset/core/functions/loss_function.py.
Unlike the original TTX provider, this reference ignores softcap/accum_dtype,
adds a mean z-loss only with return_z_loss, and folds grad_zloss into grad_loss.
These pre-existing differences are deliberately not consolidated here.
"""

import torch
import torch.nn.functional as F


def _loss(inputs, weight, target, bias, ce_weight, ignore_index, lse_square_scale,
          label_smoothing, reduction, return_z_loss):
    logits = F.linear(inputs, weight, bias).float()
    loss = F.cross_entropy(logits, target, weight=ce_weight, ignore_index=ignore_index,
                           reduction=reduction, label_smoothing=label_smoothing)
    zloss = None
    if return_z_loss:
        valid_logits = logits[target != ignore_index]
        if valid_logits.numel() > 0:
            lse = torch.logsumexp(valid_logits, dim=-1)
            zloss = lse_square_scale * (lse * lse).sum() / (target != ignore_index).sum()
            loss = loss + zloss
        else:
            zloss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    return loss, zloss


def linear_cross_entropy_fwd(inputs, weight, target, bias, ce_weight, ignore_index,
                             lse_square_scale, label_smoothing, reduction, softcap,
                             return_z_loss, accum_dtype, weight_requires_grad):
    loss, zloss = _loss(inputs, weight, target, bias, ce_weight, ignore_index,
                        lse_square_scale, label_smoothing, reduction, return_z_loss)
    return loss, zloss, None, None, None


def linear_cross_entropy_bwd(grad_loss, grad_zloss, inputs, weight, target, bias, ce_weight,
                             ignore_index, lse_square_scale, label_smoothing, reduction,
                             softcap, return_z_loss, accum_dtype, dx, dw, db):
    x = inputs.detach().requires_grad_(True)
    w = weight.detach().requires_grad_(True)
    b = None if bias is None else bias.detach().requires_grad_(True)
    with torch.enable_grad():
        loss, _ = _loss(x, w, target, b, ce_weight, ignore_index, lse_square_scale,
                        label_smoothing, reduction, return_z_loss)
    upstream = grad_loss + grad_zloss if return_z_loss and grad_zloss is not None else grad_loss
    gradients = torch.autograd.grad(loss, (x, w) if b is None else (x, w, b), upstream)
    return (*gradients, None) if b is None else gradients
