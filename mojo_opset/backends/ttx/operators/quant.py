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
        if self.smooth_scale is not None:
            scale_tensor = self.smooth_scale.float()
        else:
            scale_tensor = torch.ones(
                input.shape[-1],
                device=input.device,
                dtype=torch.float32,
            )
        return dynamic_quant(input, scale_tensor)


class TTXMoEDynamicQuant(MojoMoEDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        token_count: torch.Tensor,
    ):
        input_fp = input.float() * self.smooth_scale.float().repeat_interleave(token_count, dim=0)
        scale_tensor = torch.ones(
            input_fp.shape[-1],
            device=input_fp.device,
            dtype=torch.float32,
        )
        return dynamic_quant(input_fp, scale_tensor)
