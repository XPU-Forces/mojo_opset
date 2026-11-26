"""PyTorch reference for GLM-5.3-Flash mHC coefficients."""

import torch
import torch.nn.functional as F


def mhc_coefficients_fwd(x, phi, alpha, bias):
    batch, seq, streams, hidden = x.shape
    flat = x.reshape(batch, seq, streams * hidden).float()
    normed = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + 1e-5)
    pre_logits, post_logits, residual_logits = F.linear(normed, phi).split((4, 4, 16), dim=-1)
    pre_bias, post_bias, residual_bias = bias.split((4, 4, 16))
    pre_scale, post_scale, residual_scale = alpha.unbind()

    eps = 1e-6
    pre = torch.sigmoid(pre_logits * pre_scale + pre_bias) + eps
    post = 2 * torch.sigmoid(post_logits * post_scale + post_bias)
    residual_logits = (residual_logits * residual_scale + residual_bias).reshape(batch, seq, 4, 4)
    residual = torch.softmax(residual_logits, dim=-1) + eps
    residual = residual / (residual.sum(dim=-2, keepdim=True) + eps)
    for _ in range(19):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + eps)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + eps)

    collapsed = (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
    return collapsed, post, residual.flatten(-2)
