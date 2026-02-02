import torch
import torch_npu

from mojo_opset.core import MojoGroupLinear


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