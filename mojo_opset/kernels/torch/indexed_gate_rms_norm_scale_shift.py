# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference for indexed gated RMSNorm."""

import torch


def indexed_gate_rms_norm_scale_shift_infer_fwd(residual, gate, branch, weight, shift, scale, indices, eps):
    updated = (residual.float() + gate.index_select(0, indices).float() * branch.float()).to(torch.bfloat16)
    values = updated.float()
    variance = (values * values).mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps) * weight.float()
    modulated = normalized * (1.0 + scale.index_select(0, indices).float()) + shift.index_select(0, indices).float()
    return updated.contiguous(), modulated.to(torch.bfloat16).contiguous()
