import torch
import torch_npu

from ..operator import MojoOperator


class MojoGelu(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GELU activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Same shape as input with element-wise GELU applied.
        """
        return torch.nn.functional.gelu(x)
    
class TorchNpuGelu(MojoGelu):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
    
    def forward(
        self, x: torch.Tensor, 
        approximate: str = 'none'
    ) -> torch.Tensor:
        """
        Forward pass with GELU activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            approximate (str, optional): Approximation method for GELU. Defaults to 'none'.

        Returns:
            torch.Tensor: Same shape as input with element-wise GELU applied.
        """
        if approximate not in ['none', 'tanh']:
            raise ValueError(f"Unsupported approximate method: {approximate}\". "
                             "Only 'none' and 'tanh' are supported.")
        return torch_npu.npu_gelu(x, approximate=approximate)



class MojoGeluQuant(MojoOperator):
    pass


class MojoSilu(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SiLU (Swish) activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Same shape as input with element-wise SiLU applied.

        Notes:
            Uses torch.nn.functional.silu; preserves dtype and device.
            SiLU is defined as x * sigmoid(x).
        """
        return torch.nn.functional.silu(x)


class MojoSiluQuant(MojoOperator):
    pass


class MojoSwiGLU(MojoOperator):
    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            gate_out (torch.Tensor): Gate output tensor of any shape.
            up_out (torch.Tensor): Up output tensor of any shape.

        Returns:
            torch.Tensor: Same shape as gate_out with element-wise SwiGLU applied.

        Notes:
            SwiGLU is defined as SiLU(gate_out) * up_out.
        """
        return torch.nn.functional.silu(gate_out) * up_out
