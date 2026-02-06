import torch

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
