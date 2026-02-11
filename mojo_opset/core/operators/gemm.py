import torch


from torch.nn import functional as F
from torch.distributed.tensor import DTensor

from ..operator import MojoOperator


class MojoGroupGemm(MojoOperator):
    def __init__(
        self,
        group_num,
        in_feature,
        out_feature,
        trans_weight=False,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(group_num, out_feature, in_feature, **self.tensor_factory_kwargs))
        self.trans_weight = trans_weight

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """
        Grouped linear forward over variable-length segments.

        Splits the 2D input into contiguous groups defined by `group_list`,
        applies a per-group weight, and concatenates outputs.

        Args:
            input (torch.Tensor): 2D tensor of shape (N, Din); rows are grouped
                contiguously. Sum(group_list) must equal N.
            group_list (torch.Tensor): 1D tensor of length G with row counts per group.

        Returns:
            torch.Tensor: 2D tensor of shape (N, Dout), concatenated per-group outputs.

        Notes:
            - Expects `self.weight` of shape (G, Din, Dout). If `trans_weight` is True,
            weights are transposed from (G, Dout, Din) to (G, Din, Dout).
            - Each group's output is computed as `input_g @ weight_g`.
        """

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        if isinstance(input, DTensor):
            input = input.to_local()

        assert input.dim() == 2, "input must be 2D"
        assert weight.dim() == 3, "weight must be 3D"

        num_groups = group_list.numel()
        assert weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            weight = weight.transpose(1, 2).contiguous()

        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)

        out_list = []

        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            a_g = input[start:end, :]
            b_g = weight[g, :, :]
            out_g = F.linear(a_g, b_g)
            out_list.append(out_g)

        return torch.cat(out_list, dim=0)


class MojoGemmAllReduce(MojoOperator):
    pass


class MojoAllGatherGemm(MojoOperator):
    pass


class MojoGemmAll2All(MojoOperator):
    pass


class MojoGemmReduceScatter(MojoOperator):
    pass
