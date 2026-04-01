import torch
import torch_npu

from mojo_opset.core import MojoMoeTopkGatingDispatchDynamicQuant
from mojo_opset.core import MojoMoeGatingTopk


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


class TorchNpuMoeGatingTopk(MojoMoeGatingTopk):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        x, 
        k, 
        bias=None, 
        k_group=1, 
        group_count=1, 
        group_select_mode=0, 
        renorm=0, 
        norm_type=0, 
        out_flag=False, 
        routed_scaling_factor=1.0, 
        eps: float = 1e-20
    ) -> torch.Tensor:
        """
        Forward pass with MOE top-k gating dispatch dynamic quantization (version 2).

        Args:
            x (torch.Tensor): Input tensor of any shape.
            finished (torch.Tensor): Finished tensor of shape (batch_size,).
            k (int): Top-k value.

        Returns:
            torch.Tensor: Same shape as input with element-wise MOE top-k gating dispatch dynamic quantization applied.
        """

        y_npu, expert_idx_npu, out_npu  = torch_npu.npu_moe_gating_top_k(
            x, 
            k, 
            bias=bias, 
            k_group=k_group, 
            group_count=group_count, 
            group_select_mode=group_select_mode, 
            renorm=renorm, 
            norm_type=norm_type, 
            out_flag=out_flag, 
            routed_scaling_factor=routed_scaling_factor, 
            eps=eps
        )
        return y_npu, expert_idx_npu, out_npu