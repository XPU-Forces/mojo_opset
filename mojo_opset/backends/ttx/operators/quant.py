import torch

from mojo_opset.backends.ttx.kernels import dynamic_quant
from mojo_opset.core import MojoDynamicQuant
from mojo_opset.core import MojoMoEDynamicQuant
from mojo_opset.core import MojoStaticQuant


class TTXStaticQuant(MojoStaticQuant):
    pass


class TTXDynamicQuant(MojoDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
    ):
        return dynamic_quant(input, self.inv_smooth_scale)


class TTXMoEDynamicQuant(MojoMoEDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        token_count: torch.Tensor,
    ):
        input_fp = input.float() * self.inv_smooth_scale.float().repeat_interleave(token_count, dim=0)
        scale_tensor = torch.ones(
            input_fp.shape[-1],
            device=input_fp.device,
            dtype=torch.float32,
        )
        return dynamic_quant(input_fp, scale_tensor)
