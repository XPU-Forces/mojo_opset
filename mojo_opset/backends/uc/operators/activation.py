import torch

from mojo_opset.backends.uc.kernels import gelu_fwd
from mojo_opset.backends.uc.kernels import silu_fwd
from mojo_opset.backends.uc.kernels import swiglu_fwd
from mojo_opset.core import MojoGelu
from mojo_opset.core import MojoSilu
from mojo_opset.core import MojoSwiGLU


class UCGelu(MojoGelu):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return gelu_fwd(hidden_state)


class UCSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return silu_fwd(hidden_state)


class UCSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu"]

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit > 0:
            return super().forward(gate_out, up_out)
        return swiglu_fwd(gate_out, up_out)
