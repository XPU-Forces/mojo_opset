import torch

from mojo_opset.backends.ttx.kernels import hc_post_impl
from mojo_opset.core import MojoHcPost


class TTXHcPost(MojoHcPost):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return hc_post_impl(x, residual, post, comb)
