import torch

from mojo_opset.core import MojoGroupLinear
from mojo_opset.core import MojoLinear
from mojo_opset.utils.mode import LAST_PRIORITY


class RefLinear(MojoLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_dim = self.weight.shape[0]
        if input.shape[-1] != in_dim:
            raise ValueError(f"input should have last dim {in_dim}, but got {input.shape[-1]}")
        if self.is_varlen:
            if input.ndim not in (2, 3):
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight.t(), self.bias)
        else:
            if input.ndim not in (3, 4):
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight.t(), self.bias)


class RefGroupLinear(MojoGroupLinear):
    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor of shape [sum(group_list), M]
            group_list: 1D tensor, num_tokens per expert

        Returns:
            Tensor of shape [sum(group_list), N]
        """
        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"
        num_groups = group_list.numel()
        assert self.weight.size(0) == num_groups, "self.weight must have same group count as group_list"

        if self.trans_weight:
            self.weight = self.weight.transpose(1, 2).contiguous()

        group_start = group_list.cumsum(0) - group_list
        group_end = group_list.cumsum(0)

        out_list = []
        for g, (start, end) in enumerate(zip(group_start.tolist(), group_end.tolist())):
            a_g = input[start:end, :]
            b_g = self.weight[g, :, :]
            out_g = a_g @ b_g
            out_list.append(out_g)

        return torch.cat(out_list, dim=0)
