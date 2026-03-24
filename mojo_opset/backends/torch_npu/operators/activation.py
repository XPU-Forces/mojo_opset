import torch
import torch_npu

from mojo_opset.core import MojoGelu
from mojo_opset.core import MojoSilu
from mojo_opset.core import MojoSwiGLU


class TorchNpuGelu(MojoGelu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        """
        Forward pass with GELU activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            approximate (str, optional): Approximation method for GELU. Defaults to 'none'.

        Returns:
            torch.Tensor: Same shape as input with element-wise GELU applied.
        """
        if approximate not in ["none", "tanh"]:
            raise ValueError(f"Unsupported approximate method: {approximate}\". Only 'none' and 'tanh' are supported.")
        return torch_npu.npu_gelu(x, approximate=approximate)


class TorchNpuSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor):
        return torch_npu.npu_silu(hidden_state)


class TorchNpuSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu"]

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor):
        merged = torch.cat([gate_out, up_out], dim=-1)
        return torch_npu.npu_swiglu(merged, dim=-1)
