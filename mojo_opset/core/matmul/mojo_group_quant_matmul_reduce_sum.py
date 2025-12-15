import torch
from typing import Tuple

from ..mojo_operator import MojoOperator


class MojoGroupQuantMatmulReduceSum(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, x1: torch.Tensor, x2: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_ref(
        self, x1: torch.Tensor, x2: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor
    ) -> torch.Tensor:
        out = torch.bmm(x1.float(), x2.float()).to(torch.float32)
        out = x2_scale[None, None, :] * out
        out = x1_scale[:, :, None] * out

        b, m, k = x1.shape
        b, k, n = x2.shape
        out_1 = torch.zeros(m, n, dtype=torch.bfloat16, device=out.device)
        for i in range(b):
            out_1 += out[i, ...].to(torch.bfloat16)

        return out_1

    def forward_analysis(
        self, x1: torch.Tensor, x2: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor
    ) -> Tuple[int, int, int]:
        pass
