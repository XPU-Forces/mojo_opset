import torch

from mojo_opset.backends.ttx.kernels import quant_int8_infer
from mojo_opset.core import MojoQuantInt8


class TTXQuantInt8(MojoQuantInt8):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input_tensor: torch.Tensor,
        scale_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        3D dynamic quantization function

        Args:
            input_tensor: Input tensor with shape [Batch, Sequence, Depth]
            scale_tensor: Scale tensor with shape [Depth]

        Returns:
            output_tensor: Quantized tensor with shape [Batch, Sequence, Depth] (int8)
            quant_scale_tensor: Quantization scale tensor with shape [Batch, Sequence]
        """
        assert (
            input_tensor.dim() == 3
        ), f"Input tensor must be 3D, got {input_tensor.dim()}D"
        assert (
            scale_tensor.dim() == 1
        ), f"Scale tensor must be 1D, got {scale_tensor.dim()}D"
        assert (
            input_tensor.shape[2] == scale_tensor.shape[0]
        ), f"Input Depth {input_tensor.shape[2]} must match scale length {scale_tensor.shape[0]}"

        output_tensor, quant_scale_tensor = quant_int8_infer(
            input_tensor=input_tensor,
            scale_tensor=scale_tensor,
        )

        return output_tensor, quant_scale_tensor.to(input_tensor.dtype)
