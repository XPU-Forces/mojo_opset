from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
import torch_npu

from mojo_opset.core import MojoLayerNormQuant
from mojo_opset.core import MojoResidualAddLayerNormQuant
from mojo_opset.core import MojoResidualAddRMSNormQuant
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm
from mojo_opset.core import MojoRMSNormQuant


def _cast_smooth_scale(
    smooth_scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if smooth_scale is None:
        return None
    return smooth_scale.to(dtype=dtype)


def _dynamic_quant(
    hidden_state: torch.Tensor,
    quant_dtype: torch.dtype,
    smooth_scale: Optional[torch.Tensor] = None,
):
    quantized, scale = torch_npu.npu_dynamic_quant(
        hidden_state,
        smooth_scales=_cast_smooth_scale(smooth_scale, hidden_state.dtype),
        dst_type=quant_dtype,
    )
    return quantized, scale.unsqueeze(-1)


def _cast_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return weight.to(dtype=dtype)


class TorchNpuRMSNorm(MojoRMSNorm, default_priority=0):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        **kwargs,
    ):
        super().__init__(norm_size, eps, **kwargs)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden_state, self.weight, epsilon=self.variance_epsilon)[0]


class TorchNpuResidualAddRMSNorm(MojoResidualAddRMSNorm, default_priority=0):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        norm_pos: str = "post",
        **kwargs,
    ):
        super().__init__(norm_size, eps, norm_pos, **kwargs)

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state_out, _, residual_before_norm = torch_npu.npu_add_rms_norm(
            hidden_state, residual, self.weight, self.variance_epsilon
        )

        if self.norm_pos == "pre":
            return hidden_state_out, residual_before_norm
        else:
            return hidden_state_out, hidden_state_out


class TorchNpuRMSNormQuant(MojoRMSNormQuant, default_priority=0):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        hidden_state: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        normed = torch_npu.npu_rms_norm(
            hidden_state,
            _cast_weight(self.weight, hidden_state.dtype),
            epsilon=self.variance_epsilon,
        )[0]
        return _dynamic_quant(normed, self.quant_dtype, smooth_scale)


class TorchNpuLayerNormQuant(MojoLayerNormQuant, default_priority=0):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        hidden_state: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        normed = F.layer_norm(
            hidden_state.float(),
            [hidden_state.shape[-1]],
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        ).to(hidden_state.dtype)
        return _dynamic_quant(normed, self.quant_dtype, smooth_scale)


class TorchNpuResidualAddRMSNormQuant(MojoResidualAddRMSNormQuant, default_priority=0):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        normed, _, residual_before_norm = torch_npu.npu_add_rms_norm(
            hidden_state,
            residual,
            _cast_weight(self.weight, hidden_state.dtype),
            self.variance_epsilon,
        )
        quantized, scale = _dynamic_quant(normed, self.quant_dtype, smooth_scale)
        return quantized, residual_before_norm, scale


class TorchNpuResidualAddLayerNormQuant(MojoResidualAddLayerNormQuant, default_priority=0):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
    ):
        residual_before_norm = hidden_state + residual
        normed = F.layer_norm(
            residual_before_norm.float(),
            [residual_before_norm.shape[-1]],
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        ).to(hidden_state.dtype)
        quantized, scale = _dynamic_quant(normed, self.quant_dtype, smooth_scale)
        return quantized, residual_before_norm, scale
