import torch

from mojo_opset.backends.ttx.kernels import hc_pre
from mojo_opset.core import MojoHcPre


class TTXHcPre(MojoHcPre):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ):
        return hc_pre(
            x, hc_fn, hc_scale, hc_base,
            hc_mult=hc_mult,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
        )
