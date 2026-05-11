import torch

from mojo_opset.core import MojoSilu


class UCSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("UC backend stub: mojo_opset.backends.uc.operators.silu.UCSilu.forward")
        raise NotImplementedError("UC backend MojoSilu is waiting for the uc-kernel pybind implementation.")
