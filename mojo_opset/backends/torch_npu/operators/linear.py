from typing import List, Optional, Union

import torch
import torch_npu

from mojo_opset.core import MojoDequantGroupLinear
from mojo_opset.core import MojoGroupLinear
from mojo_opset.core import MojoQuantGroupLinear
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class TorchNpuGroupLinear(MojoGroupLinear):
    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        weight_list = [weight[g].contiguous() for g in range(num_groups)]
        group_list_values = [int(x) for x in group_list.cumsum(0).tolist()]
        outputs = torch_npu.npu_grouped_matmul([input], weight_list, group_type=0, group_list=group_list_values)
        return torch.cat(outputs, dim=0)


class TorchNpuQuantGroupLinear(MojoQuantGroupLinear):
    def forward(self, input: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 3, "input must be 3D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        if x2_scale.dtype != torch.bfloat16:
            x2_scale = x2_scale.to(torch.bfloat16)

        fmt = torch_npu.get_npu_format(weight)
        if fmt != 29:
            x2_nz = torch_npu.npu_format_cast(weight, 29)
            logger.info(f"Not support weight format {fmt}, cast to NZ format")
        else:
            x2_nz = weight

        return torch_npu.npu_quant_matmul_reduce_sum(input, x2_nz, x1_scale, x2_scale)


class TorchNpuDequantGroupLinear(MojoDequantGroupLinear):
    def forward(
        self,
        input: torch.Tensor,
        group_list: torch.Tensor,
        antiquant_scale: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        antiquant_offset: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        weight_list = [weight[g].contiguous() for g in range(num_groups)]
        group_list_values = [int(x) for x in group_list.cumsum(0).tolist()]

        scale_list = None
        if antiquant_scale is not None:
            if isinstance(antiquant_scale, list):
                scale_list = antiquant_scale
            elif antiquant_scale.dim() > 0 and antiquant_scale.size(0) == num_groups:
                scale_list = [antiquant_scale[g] for g in range(num_groups)]
            else:
                scale_list = [antiquant_scale]

        if scale_list is not None:
            scale_list = [s.to(input.device) if isinstance(s, torch.Tensor) else s for s in scale_list]

        offset_list = None
        if antiquant_offset is not None:
            if isinstance(antiquant_offset, list):
                offset_list = antiquant_offset
            elif antiquant_offset.dim() > 0 and antiquant_offset.size(0) == num_groups:
                offset_list = [antiquant_offset[g] for g in range(num_groups)]
            else:
                offset_list = [antiquant_offset]

        if offset_list is not None:
            offset_list = [o.to(input.device) if isinstance(o, torch.Tensor) else o for o in offset_list]

        outputs = torch_npu.npu_grouped_matmul(
            [input],
            weight_list,
            group_type=0,
            group_list=group_list_values,
            antiquant_scale=scale_list,
            antiquant_offset=offset_list,
        )
        return torch.cat(outputs, dim=0)
