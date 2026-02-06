import torch

from mojo_opset.backends.ttx.kernels import quant_int8_infer
from mojo_opset.core import MojoQuantInt8, MojoQuant


class TTXQuant(MojoQuant):
    pass

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
            input_tensor: Input tensor with shape BSD, BSND, BNSD
            scale_tensor: Scale tensor with shape [D]

        Returns:
            output_tensor: Quantized tensor with shape [Batch, Sequence, Depth] (int8)
            quant_scale_tensor: Quantization scale tensor with shape [Batch, Sequence]
        """
        if scale_tensor is None:
            head_dim = input_tensor.shape[-1]
            scale_tensor = torch.ones(head_dim, dtype=input_tensor.dtype, device=input_tensor.device)

        assert (
            input_tensor.dim() in [3, 4]
        ), f"Input tensor must be 3D or 4D, got {input_tensor.dim()}D"
        assert (
            scale_tensor.dim() == 1
        ), f"Scale tensor must be 1D, got {scale_tensor.dim()}D"
        assert (
            input_tensor.shape[-1] == scale_tensor.shape[0]
        ), f"Input Depth {input_tensor.shape[-1]} must match scale length {scale_tensor.shape[0]}"

        output_tensor, quant_scale_tensor = quant_int8_infer(
            input_tensor=input_tensor,
            scale_tensor=scale_tensor,
        )

        return output_tensor, quant_scale_tensor.to(input_tensor.dtype)
