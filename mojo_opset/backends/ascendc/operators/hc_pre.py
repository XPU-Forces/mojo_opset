import torch

from mojo_opset.core import MojoHcPre


class AscendcHcPre(MojoHcPre):
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
        return torch.ops.custom.npu_hc_pre(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )
