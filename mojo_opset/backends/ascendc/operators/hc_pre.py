import torch

from mojo_opset.core import MojoHcPre
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


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
        try:
            import torch_npu

            if hasattr(torch_npu, "npu_hc_pre"):
                # print("aaaaaaaaaaaa")
                return torch_npu.npu_hc_pre(
                    x,
                    hc_fn,
                    hc_scale,
                    hc_base,
                    hc_mult=hc_mult,
                    hc_sinkhorn_iters=hc_sinkhorn_iters,
                    norm_eps=norm_eps,
                    hc_eps=hc_eps,
                )
        except Exception:
            pass

        try:
            try:
                import custom_ops  # noqa: F401
            except Exception:
                pass
            # print("bbbbbbbbbbb")
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
        except Exception:
            logger.warning("AscendC HcPre kernel not available, falling back to reference implementation.")
            return super().forward(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult=hc_mult,
                hc_sinkhorn_iters=hc_sinkhorn_iters,
                norm_eps=norm_eps,
                hc_eps=hc_eps,
            )
