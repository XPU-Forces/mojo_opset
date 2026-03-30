from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import quant_infer
from mojo_opset.core import MojoQuant, MojoQuantIndexer


class TTXQuant(MojoQuant):
    pass


class TTXQuantIndexer(MojoQuantIndexer):
    supported_platforms_list = ["npu"]

    def forward(self, input_tensor: torch.Tensor, scale_tensor: Optional[torch.Tensor] = None):
        if scale_tensor is None:
            scale_tensor = torch.ones(
                input_tensor.shape[-1],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
        return quant_infer(input_tensor, scale_tensor)

