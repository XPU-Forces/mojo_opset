import torch
import torch_npu

from mojo_opset.core import MojoMoeTopkGatingDispatchDynamicQuant


class TorchNpuMoeTopkGatingDispatchDynamicQuant(MojoMoeTopkGatingDispatchDynamicQuant):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        x: torch.Tensor,
        finished: torch.Tensor = None,
        k: int = 1,
        ) -> torch.Tensor:
        """
        Forward pass with MOE top-k gating dispatch dynamic quantization.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            finished (torch.Tensor): Finished tensor of shape (batch_size,).
            k (int): Top-k value.

        Returns:
            torch.Tensor: Same shape as input with element-wise MOE top-k gating dispatch dynamic quantization applied.
        """

        return torch_npu.npu_moe_gating_top_k_softmax(x, finished, k)
