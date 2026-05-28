import torch

from mojo_opset.core import MojoHcPost


class AscendcHcPost(MojoHcPost):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return torch.ops.custom.npu_hc_post(x, residual, post, comb)