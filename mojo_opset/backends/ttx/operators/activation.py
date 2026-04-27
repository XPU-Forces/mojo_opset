import torch

from mojo_opset.backends.ttx.kernels import gelu_fwd
from mojo_opset.backends.ttx.kernels import silu_fwd
from mojo_opset.backends.ttx.kernels import swiglu_fwd

from mojo_opset.core import MojoGelu
from mojo_opset.core import MojoSilu
from mojo_opset.core import MojoSwiGLU


class TTXGelu(MojoGelu):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, hidden_state: torch.Tensor):
        return gelu_fwd(hidden_state)


class TTXSilu(MojoSilu):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, hidden_state: torch.Tensor):
        return silu_fwd(hidden_state)


class TTXSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor):
        if self.swiglu_limit > 0:
            return super().forward(gate_out, up_out)
        return swiglu_fwd(gate_out, up_out)
