import torch

from mojo_opset.backends.uc.kernels import gelu_fwd
from mojo_opset.core import MojoGelu


class UCGelu(MojoGelu):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return gelu_fwd(hidden_state)
