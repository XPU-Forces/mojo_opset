from __future__ import annotations

from typing import Optional, Tuple

import torch

from mojo_opset.core import MojoMoEGatingTopK
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class AscendcMoEGatingTopK(MojoMoEGatingTopK):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        k: int,
        *,
        bias: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        tid2eid: Optional[torch.Tensor] = None,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        out_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            import torch_npu

            if (
                hasattr(torch_npu, "npu_moe_gating_top_k")
                and input_ids is None
                and tid2eid is None
            ):
                return torch_npu.npu_moe_gating_top_k(
                    x,
                    k,
                    bias=bias,
                    k_group=k_group,
                    group_count=group_count,
                    routed_scaling_factor=routed_scaling_factor,
                    eps=eps,
                    group_select_mode=group_select_mode,
                    renorm=renorm,
                    norm_type=norm_type,
                    out_flag=out_flag,
                )
        except Exception:
            pass

        try:
            try:
                import custom_ops  # noqa: F401
            except Exception:
                pass

            return torch.ops.custom.npu_moe_gating_top_k(
                x,
                k,
                bias=bias,
                input_ids=input_ids,
                tid2eid=tid2eid,
                k_group=k_group,
                group_count=group_count,
                routed_scaling_factor=routed_scaling_factor,
                eps=eps,
                group_select_mode=group_select_mode,
                renorm=renorm,
                norm_type=norm_type,
                out_flag=out_flag,
            )
        except Exception:
            logger.warning(
                "AscendC MoEGatingTopK kernel not available, falling back to reference implementation."
            )
            return super().forward(
                x,
                k,
                bias=bias,
                input_ids=input_ids,
                tid2eid=tid2eid,
                k_group=k_group,
                group_count=group_count,
                group_select_mode=group_select_mode,
                renorm=renorm,
                norm_type=norm_type,
                out_flag=out_flag,
                routed_scaling_factor=routed_scaling_factor,
                eps=eps,
            )

