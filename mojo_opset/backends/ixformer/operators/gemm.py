from typing import Optional

import torch

from ixformer import functions as ixf_f
from mojo_opset.core.operators.gemm import MojoGemmDequant
from mojo_opset.core.operators.gemm import MojoGroupGemm

class IxformerGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["ilu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        if group_list.device.type != "cpu":
            group_list = group_list.to("cpu")

        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"

        return ixf_f.moe_w16a16_group_gemm(input, self.weight, input.dtype, group_list, format="TN" if self.trans_weight else "NN")


class IxformerGemmDequant(MojoGemmDequant):
    supported_platforms_list = ["ilu"]
    def __init__(
        self,
        weight_scale_size: int,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = False,
        **kwargs,
    ):
        super().__init__(weight_scale_size, output_dtype, trans_weight, **kwargs)
        if self.output_dtype == torch.float32:
            raise NotImplementedError("IxformerGemmDequant does not support float32 output dtype")
        setattr(self.weight_scale, "force_dtype", torch.float32)

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return ixf_f.w8a8(input, weight, input_scale, self.weight_scale, bias, format="TN" if self.trans_weight else "NN", out_dtype=self.output_dtype)