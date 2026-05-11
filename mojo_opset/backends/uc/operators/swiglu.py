import torch

from mojo_opset.core import MojoSwiGLU


class UCSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu"]

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        print("UC backend stub: mojo_opset.backends.uc.operators.swiglu.UCSwiGLU.forward")
        raise NotImplementedError("UC backend MojoSwiGLU is waiting for the uc-kernel pybind implementation.")
