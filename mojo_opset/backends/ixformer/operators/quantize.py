import torch

from ixformer import functions as ixf_f
from mojo_opset.core import MojoStaticQuant
from mojo_opset.core import MojoDynamicQuant


class IxformerStaticQuant(MojoStaticQuant):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        input: torch.Tensor,
    ):
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"IxformerStaticQuant only supports fp16/bf16 input, got {input.dtype}."
            )

        if self.scale.dtype not in (torch.float, torch.bfloat16):
            raise NotImplementedError(
                f"IxformerStaticQuant only supports fp32/bf16 scale, got {self.scale.dtype}."
            )

        if input.dim() < len(self.input_size):
            raise ValueError(
                f"input must have at least {len(self.input_size)} dims for scale shape "
                f"{self.input_size}, got {tuple(input.shape)}."
            )
        if tuple(input.shape[-len(self.input_size):]) != self.input_size:
            raise ValueError(
                f"input trailing dims {tuple(input.shape[-len(self.input_size):])} must "
                f"match scale shape {self.input_size}."
            )

        output = torch.empty_like(input, dtype=self.quant_dtype, device=input.device)

        ixf_f.static_quant(output, input, self.scale, self.quant_dtype)

        return output, self.scale

class IxformerDynamicQuant(MojoDynamicQuant):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        input: torch.Tensor,
    ):
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"IxformerDynamicQuant only supports fp16/bf16 input, got {input.dtype}."
            )

        if self.inv_smooth_scale is not None and self.inv_smooth_scale.dtype != input.dtype and self.inv_smooth_scale.dtype != torch.float:
            raise NotImplementedError(
                f"IxformerDynamicQuant only supports fp32 inv_smooth_scale or the same dtype with input, got {self.inv_smooth_scale.dtype}."
            )

        output, scale = ixf_f.dynamic_quant(input, self.inv_smooth_scale)

        return output, scale
