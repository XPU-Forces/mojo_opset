import torch
import torch_npu

from mojo_opset.core import MojoDequantSwiGLUClampQuant
from mojo_opset.core import MojoDequantSwiGLUQuant
from mojo_opset.core import MojoDynamicQuant
from mojo_opset.core import MojoMoEDynamicQuant


class TorchNpuDynamicQuant(MojoDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
    ):
        kwargs = {"dst_type": self.quant_dtype}
        if self.inv_smooth_scale is not None:
            kwargs["smooth_scales"] = self.inv_smooth_scale.to(dtype=input.dtype)
        output, scale = torch_npu.npu_dynamic_quant(input, **kwargs)
        return output, scale.unsqueeze(-1)


class TorchNpuMoEDynamicQuant(MojoMoEDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        token_count: torch.Tensor,
    ):
        kwargs = {
            "dst_type": self.quant_dtype,
            "group_index": torch.cumsum(
                token_count.to(dtype=torch.int32, device=input.device),
                dim=0,
                dtype=torch.int32,
            ),
        }
        kwargs["smooth_scales"] = self.inv_smooth_scale.to(dtype=input.dtype)
        output, scale = torch_npu.npu_dynamic_quant(input, **kwargs)
        return output, scale.unsqueeze(-1)


class TorchNpuDequantSwiGLUQuant(MojoDequantSwiGLUQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        activation_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        quant_offset: torch.Tensor = None,
        token_count: torch.Tensor = None,
    ):
        output, scale = torch_npu.npu_dequant_swiglu_quant(
            x,
            weight_scale=self.weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=self.quant_scale,
            quant_offset=quant_offset,
            group_index=token_count,
            activate_left=self.activate_left,
            quant_mode=self.quant_mode,
        )
        return output, scale.unsqueeze(-1)


class TorchNpuDequantSwiGLUClampQuant(MojoDequantSwiGLUClampQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        activation_scale: torch.Tensor,
        token_count: torch.Tensor,
    ):
        # Match the core contract: left-gated clamped SiLU, dynamic INT8, no extra alpha/bias.
        output, scale = torch_npu.npu_dequant_swiglu_clamp_quant(
            input,
            weight_scale=self.weight_scale,
            quant_scale=self.quant_scale,
            activation_scale=activation_scale.reshape(-1),
            group_index=token_count,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=self.clamp_limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )
        return output, scale.unsqueeze(-1)
