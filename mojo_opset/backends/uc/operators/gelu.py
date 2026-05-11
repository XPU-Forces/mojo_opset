import torch

from mojo_opset.core import MojoGelu


class UCGelu(MojoGelu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("UC backend stub: mojo_opset.backends.uc.operators.gelu.UCGelu.forward")
        raise NotImplementedError("UC backend MojoGelu is waiting for the uc-kernel pybind implementation.")
