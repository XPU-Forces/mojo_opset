from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoRMSNormDynamicQuant(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        *,
        smooth_scale: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        epsilon: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reference implementation for fused RMSNorm + dynamic quant.

        Mirrors ``torch.ops.custom.npu_rms_norm_dynamic_quant`` signature:
            (Tensor x, Tensor gamma, *, Tensor? smooth_scale=None, Tensor? beta=None, float epsilon=1e-6)
                -> (Tensor y_out_int8, Tensor scale_out_fp32)

        Notes:
        - RMSNorm is computed in float32 for numerical stability.
        - ``smooth_scale`` (if provided) multiplies the RMSNorm output before quantization.
        - Per-row scale is ``row_max(abs(input)) / 127`` (float32), where "row" is the last-dim reduction.
        """
        if x.dim() < 1:
            raise ValueError("MojoRMSNormDynamicQuant: x must have at least 1 dimension.")
        if gamma.dim() != 1:
            raise ValueError(f"MojoRMSNormDynamicQuant: gamma must be 1D, got shape {tuple(gamma.shape)}.")
        if x.size(-1) != gamma.numel():
            raise ValueError(
                "MojoRMSNormDynamicQuant: gamma length must match x.size(-1), "
                f"got gamma={gamma.numel()} vs x.size(-1)={x.size(-1)}."
            )
        if beta is not None:
            if beta.dim() != 1:
                raise ValueError(f"MojoRMSNormDynamicQuant: beta must be 1D, got shape {tuple(beta.shape)}.")
            if beta.numel() != gamma.numel():
                raise ValueError(
                    "MojoRMSNormDynamicQuant: beta length must match gamma length, "
                    f"got beta={beta.numel()} vs gamma={gamma.numel()}."
                )
        if smooth_scale is not None:
            if smooth_scale.dim() != 1:
                raise ValueError(
                    f"MojoRMSNormDynamicQuant: smooth_scale must be 1D, got shape {tuple(smooth_scale.shape)}."
                )
            if smooth_scale.numel() != gamma.numel():
                raise ValueError(
                    "MojoRMSNormDynamicQuant: smooth_scale length must match gamma length, "
                    f"got smooth_scale={smooth_scale.numel()} vs gamma={gamma.numel()}."
                )

        # RMSNorm in float32.
        y = F.rms_norm(
            x.float(),
            (x.size(-1),),
            weight=gamma.float(),
            bias=None if beta is None else beta.float(),
            eps=float(epsilon),
        )

        if smooth_scale is not None:
            y = y * smooth_scale.float()

        # Per-row scale: (...,) in float32.
        scale = y.abs().amax(dim=-1) / 127.0
        # Match common dynamic-quant behavior: avoid division by zero.
        scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale).to(dtype=torch.float32)

        # Quantize.
        q = torch.round(y / scale.unsqueeze(-1))
        q = torch.clamp(q, -128, 127).to(dtype=torch.int8)
        return q.contiguous(), scale.contiguous()

