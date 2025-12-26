from typing import Optional
from typing import Tuple

import torch

from ...utils.mode import get_mojo_exec_mode
from ..mojo_operator import MojoOperator


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
            raise ValueError('input_layout should be {"KN","NZ"}')
        self.input_layout = input_layout

        if weight is None or not isinstance(weight, torch.Tensor):
            raise TypeError("weight should be torch.Tensor and not None")
        if weight.ndim not in (2,):
            raise ValueError(f"weight should be 2-D, but got {tuple(weight.shape)}")
        self.weight = weight

        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("bias should be torch.Tensor or None")
            if weight.ndim == 2:
                out_dim = weight.shape[1]
                if bias.ndim != 1 or bias.shape[0] != out_dim:
                    raise ValueError(f"bias should be 1-D with shape [out_dim={out_dim}], but got {tuple(bias.shape)}")
        self.bias = bias
        self.is_varlen = is_varlen
        mode_str = get_mojo_exec_mode(MojoLinear.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)


class MojoBatchLinear(MojoOperator):
    pass


class MojoGroupLinear(MojoOperator):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        trans_weight=False,
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

    def forward_std(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MojoLinearAllReduce(MojoOperator):
    pass


class MojoAllGatherLinear(MojoOperator):
    pass


class MojoLinearAll2All(MojoOperator):
    pass


class MojoLinearReduceScatter(MojoOperator):
    pass
