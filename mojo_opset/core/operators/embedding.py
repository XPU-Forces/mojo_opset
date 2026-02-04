import torch

from ..operator import MojoOperator


class MojoQuant(MojoOperator):
    pass


class MojoQuantInt8(MojoOperator):

    def __init__(self):
        pass

    def forward(self, input_tensor:torch.Tensor, scale_tensor:torch.Tensor=None):
        if scale_tensor is None:
            scaled_fp32 = scaled_fp32.to(torch.float64)
        else:
            scaled_fp32 = input_tensor.to(torch.float64) * scale_tensor.to(torch.float64)

        max_abs = scaled_fp32.abs().amax(dim=-1)
        quant_vals = 127* (scaled_fp32 / max_abs.unsqueeze(-1))

        q = torch.trunc(quant_vals + 0.5 * torch.sign(quant_vals))
        return q.to(torch.int8), (max_abs / 127.0).to(input_tensor.dtype)


class MojoDequant(MojoOperator):
    pass


class MojoEmbedding(MojoOperator):
    pass


class MojoParallelEmbedding(MojoOperator):
    pass
