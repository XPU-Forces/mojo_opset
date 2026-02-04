import torch
import math
from ..operator import MojoOperator


class MojoQuant(MojoOperator):
    pass


class MojoQuantInt8(MojoOperator):

    def forward(self, input_tensor: torch.Tensor, scale_tensor: torch.Tensor = None):
        # * golden need fp64 or fp32 to make sure correct.
        if scale_tensor is None:
            scale_tensor = input_tensor.to(torch.float64)
        else:
            scale_tensor = input_tensor.to(torch.float64) * scale_tensor.to(torch.float64)

        max_abs = scale_tensor.abs().amax(dim=-1)
        quant_vals = 127 * (scale_tensor / max_abs.unsqueeze(-1))

        q = torch.trunc(quant_vals + 0.5 * torch.sign(quant_vals))
        return q.to(torch.int8), (max_abs / 127.0).to(input_tensor.dtype)


class MojoDequant(MojoOperator):
    pass


class MojoEmbedding(MojoOperator):
    pass


class MojoParallelEmbedding(MojoOperator):
    pass


def hadamard(n: int, dtype, device):
    """
    Torch version hadamard matrix generation
    refer to https://pytorch.org/blog/hadacore/
    """
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))

    if 2**lg2 != n:
        raise ValueError(f"n must be a power of 2, but got {n}")

    H = torch.tensor([1], dtype=dtype, device=device)
    for _ in range(0, lg2):
        H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))
    return H
