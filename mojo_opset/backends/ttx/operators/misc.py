import torch
from mojo_opset.backends.ttx.kernels import quest
from mojo_opset.core import MojoQuest


class TTXQuest(MojoQuest):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        mins: torch.Tensor,
        maxs: torch.Tensor,
        top_k_page: int,
    ):
        return quest(query, mins, maxs, top_k_page)
