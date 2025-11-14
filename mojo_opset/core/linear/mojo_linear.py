import os
import torch
from typing import Any, Tuple, Optional, Union

from ..mojo_operator import MojoOperator
from ...mojo_utils import get_mojo_exec_mode

class MojoLinear(MojoOperator):
    def __init__(
        self,
        input_layout: str = "NZ",
        weight: torch.Tensor = None,
        bias: Optional[torch.Tensor] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Common parameter definitions for Linear operator.

        Init parameters:
        - input_layout (str): Input layout enumeration, values {"KN","NZ"}, default "NZ".
        - weight (torch.Tensor): Weight tensor, shape [in_dim, out_dim].
        - bias (Optional[torch.Tensor]): Bias tensor, shape aligned with output dimension; optional.
        - is_varlen (bool): When True, prioritize TND (per token) computation; when False, use BSND; default True.
        - op_name (str): Operator name placeholder.
        """
        super().__init__(op_name)

        if input_layout not in {"KN", "NZ"}:
            raise ValueError('input_layout 需为 {"KN","NZ"}')
        self.input_layout = input_layout

        if weight is None or not isinstance(weight, torch.Tensor):
            raise TypeError("weight 必须为 torch.Tensor 且不可为 None")
        if weight.ndim not in (2, ):
            raise ValueError(f"weight 需为 2-D，实际为 {tuple(weight.shape)}")
        self.weight = weight

        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("bias 必须为 torch.Tensor 或 None")
            
            if weight.ndim == 2:
                out_dim = weight.shape[1]
                if bias.ndim != 1 or bias.shape[0] != out_dim:
                    raise ValueError("bias 形状需为 [out_dim]，并与 weight 的输出维一致")
        self.bias = bias
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")
        self.is_varlen = is_varlen

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - input：输入张量。

        输出：
        - output: 输出张量，形状需遵循矩阵乘法规则。

        """
        
        raise NotImplementedError("MojoGroupLinear forward 仅做通用参数校验，不包含具体计算")


    def forward_ref(self, input: torch.Tensor) -> torch.Tensor:
        """
        参考实现（golden）：标准线性变换，严格区分 TND/BNSD 输入。
        输入布局契约：
        - 当 is_varlen=True（TND）：仅接受 [T, in_dim] 或 [T, G, in_dim]
        - 当 is_varlen=False（BNSD）：仅接受 [B, S, in_dim] 或 [B, S, G, in_dim]
        - 否则报错（Expected TND/BNSD ...）。
        公式：Y = X · W + b，其中 W=[in_dim,out_dim]，b=[out_dim]
        返回：形状遵循矩阵乘法规则，最后一维为 out_dim，dtype 与输入一致。
        """
        in_dim = self.weight.shape[0]
        if input.shape[-1] != in_dim:
            raise ValueError("input 的最后一维需与 weight 的 in_dim 对齐")
        if self.is_varlen:
            # 仅接受 TND
            if not (input.ndim in (2, 3)):
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight.t(), self.bias)
        else:
            # 仅接受 BNSD
            if not (input.ndim in (3, 4)):
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(input.shape)}")
            return torch.nn.functional.linear(input, self.weight.t(), self.bias)


class MojoBatchLinear(MojoOperator):
    pass


class MojoGroupLinear(MojoOperator):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        trans_weight = False,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        
        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

        mode_str = get_mojo_exec_mode(MojoGroupLinear.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)


    def forward_ref(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
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


    def forward_std(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


    def forward_analysis(self, input: torch.Tensor, group_list: torch.Tensor) -> Tuple[int, int, int]:
        pass