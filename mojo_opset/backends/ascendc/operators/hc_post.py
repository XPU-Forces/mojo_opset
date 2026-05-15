import torch

from mojo_opset.core import MojoHcPost
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class AscendcHcPost(MojoHcPost):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        try:
            import torch_npu

            if hasattr(torch_npu, "npu_hc_post"):
                return torch_npu.npu_hc_post(x, residual, post, comb)
        except Exception:
            pass

        try:
            import custom_ops
            return torch.ops.custom.npu_hc_post(x, residual, post, comb)
        except Exception:
            logger.warning("AscendC HcPost kernel not available, falling back to reference implementation.")
            return super().forward(x, residual, post, comb)