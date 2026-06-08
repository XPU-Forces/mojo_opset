"""UC backend for ``MojoFusedSwiGLUMoEScaleDynamicQuantize``.

This op fuses, per row of a flattened MoE routed batch:

    left, right = chunk(input, dim=-1)
    output      = (silu(left * beta) / beta) * right    # SwiGLU
    output      = output * expanded_smooth_scale        # optional per-expert scale
    scale       = output.abs().amax(-1).clamp(1e-12) / 127
    quantized   = clamp(round(output / scale), -128, 127).to(int8)

Kernel contract (see ``uc-kernel/kernels/mojo_fused_swiglu_moe_scale_dynamic_quantize_bf16.py``):

  * ``x``            : ``(M, 2H)`` bf16  — gate half then up half on last dim
  * ``smooth_scale`` : ``(M, H)``  bf16  — host pre-expanded (ones when absent)
  * ``y``            : ``(M, H)``  int8
  * ``scale``        : ``(M,)``    fp32
  * trailing scalars : ``(M, N=2H, H, BETA)`` — first-appearance rule

Per project rule "wheel 没实现的就直接给报错" (2026-06-08): every guard
that previously fell back to ``super().forward(...)`` (torch reference)
now raises ``NotImplementedError`` — UC wrappers must never silently
fall back to torch.
"""

from typing import Optional

import torch

from mojo_opset.experimental.operators.moe import MojoFusedSwiGLUMoEScaleDynamicQuantize

from ._utils import _uc_kernels


_UC_KERNEL_API = "mojo_fused_swiglu_moe_scale_dynamic_quantize_bf16"


class UCFusedSwiGLUMoEScaleDynamicQuantize(MojoFusedSwiGLUMoEScaleDynamicQuantize):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        smooth_scale: Optional[torch.Tensor],
        token_count: torch.Tensor,
        beta: float = 1.0,
        quant_mode: int = 0,
    ):
        # ----- shape / dtype / mode guards -----
        if input.dim() != 3:
            raise NotImplementedError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize expects 3D input, got {input.dim()}D."
            )
        if input.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize only supports bf16, got {input.dtype}."
            )
        if self.quant_dtype != torch.int8:
            raise NotImplementedError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize only supports int8 quant, got {self.quant_dtype}."
            )
        if quant_mode not in (0, 1):
            raise NotImplementedError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize supports quant_mode 0|1, got {quant_mode}."
            )
        if beta == 0:
            raise NotImplementedError("UC FusedSwiGLUMoEScaleDynamicQuantize requires beta != 0.")

        two_h = input.shape[-1]
        if two_h % 2 != 0:
            raise NotImplementedError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize requires even last dim, got {two_h}."
            )
        hidden = two_h // 2
        rows = input.shape[0] * input.shape[1]

        if token_count.dim() != 1:
            raise NotImplementedError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize expects 1D token_count, got {token_count.dim()}D."
            )
        if token_count.dtype not in (torch.int32, torch.int64):
            raise NotImplementedError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize expects int32/int64 token_count, got {token_count.dtype}."
            )
        if int(token_count.sum().item()) != rows:
            raise ValueError(
                f"UC FusedSwiGLUMoEScaleDynamicQuantize: token_count sum ({int(token_count.sum().item())}) "
                f"does not equal flattened input rows ({rows})."
            )

        if smooth_scale is not None:
            if smooth_scale.dim() != 2:
                raise NotImplementedError(
                    f"UC FusedSwiGLUMoEScaleDynamicQuantize expects 2D smooth_scale, got {smooth_scale.dim()}D."
                )
            if smooth_scale.size(0) != token_count.numel():
                raise ValueError(
                    f"UC FusedSwiGLUMoEScaleDynamicQuantize: smooth_scale rows {smooth_scale.size(0)} "
                    f"must equal token_count length {token_count.numel()}."
                )
            if smooth_scale.size(1) != hidden:
                raise ValueError(
                    f"UC FusedSwiGLUMoEScaleDynamicQuantize: smooth_scale cols {smooth_scale.size(1)} "
                    f"must equal hidden dim {hidden}."
                )

        kernels = _uc_kernels()
        if _UC_KERNEL_API not in kernels:
            raise NotImplementedError(
                f"UC kernel {_UC_KERNEL_API!r} is not in the loaded uc-kernel wheel. "
                "See docs/project-ops/uc-kernel-fail-todo-2026-06-08.md."
            )

        # Zero-element fast path
        if input.numel() == 0:
            quantized_shape = (*input.shape[:-1], hidden)
            scale_shape = input.shape[:-1]
            return (
                torch.empty(quantized_shape, dtype=self.quant_dtype, device=input.device),
                torch.empty(scale_shape, dtype=torch.float32, device=input.device),
            )

        # ----- host-side prep -----
        kernel_x = input.contiguous().reshape(rows, two_h)

        if smooth_scale is None:
            smooth_exp = torch.ones((rows, hidden), dtype=torch.bfloat16, device=input.device)
        else:
            token_count_dev = token_count.to(device=input.device, dtype=torch.int64)
            # repeat_interleave(per-expert smooth, dim=0) -> (rows, H)
            smooth_exp = smooth_scale.to(
                device=input.device,
                dtype=torch.bfloat16,
            ).repeat_interleave(token_count_dev, dim=0).contiguous()

        kernel_y = torch.empty((rows, hidden), dtype=torch.int8, device=input.device)
        kernel_scale = torch.empty((rows,), dtype=torch.float32, device=input.device)

        # ----- launch kernel -----
        # Trailing scalars follow prim_func first-appearance order: (M, N=2H, H, BETA)
        kernels[_UC_KERNEL_API](
            kernel_x,
            smooth_exp,
            kernel_y,
            kernel_scale,
            rows,
            two_h,
            hidden,
            float(beta),
        )

        # ----- reshape outputs back to operator convention -----
        out_shape = (*input.shape[:-1], hidden)
        scale_shape = input.shape[:-1]
        return (
            kernel_y.reshape(out_shape).to(self.quant_dtype),
            kernel_scale.reshape(scale_shape),
        )
