from typing import Union

import torch

from ixformer import functions as ixf_f
from mojo_opset.core.operators.gemm import MojoQuantGemm
from mojo_opset.core.operators.gemm import MojoGroupGemm

class IxformerGroupGemm(MojoGroupGemm):
    supported_platforms_list = ["ilu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        if group_list.device.type != "cpu":
            group_list = group_list.to("cpu")

        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if input.dtype not in [torch.float16, torch.bfloat16]:
            raise NotImplementedError("IxformerGroupGemm does not support input dtype other than float16 and bfloat16")

        if self.trans_weight:
            num_groups_w, n, bk = self.weight.shape
            if bk % 32 != 0 or n % 2 != 0:
                raise NotImplementedError("K of input should be divisible by 32 and N of weight should be divisible by 2 when trans_weight is True")
        else:
            num_groups_w, bk, n = self.weight.shape
            if bk % 32 != 0 or n % 32 != 0:
                raise NotImplementedError("K of input should be divisible by 32 and N of weight should be divisible by 32 when trans_weight is False")

        return ixf_f.moe_w16a16_group_gemm(input, self.weight, input.dtype, group_list, format="TN" if self.trans_weight else "NN")


class IxformerQuantGemm(MojoQuantGemm):
    supported_platforms_list = ["ilu"]

    @staticmethod
    def _cast_weight_scale_post_hook(module, incompatible_keys):
        module.weight_scale = torch.nn.Parameter(
            module.weight_scale.detach().to(torch.float32),
            requires_grad=module.weight_scale.requires_grad,
        )

    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = False,
        quant_dtype: torch.dtype = torch.int8,
        weight_dtype: Union[str, torch.dtype] = torch.int8,
        **kwargs,
    ):
        super().__init__(in_features, out_features, output_dtype, trans_weight, quant_dtype, weight_dtype, **kwargs)
        
        if self.output_dtype == torch.float32:
            raise NotImplementedError("IxformerQuantGemm does not support float32 output dtype")
        
        self.register_load_state_dict_post_hook(self._cast_weight_scale_post_hook)

    def forward(
        self,
        input: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:

        return ixf_f.w8a8(input, self.weight, input_scale, self.weight_scale, format="TN" if self.trans_weight else "NN", out_dtype=self.output_dtype)