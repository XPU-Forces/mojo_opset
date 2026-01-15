from typing import Optional

import torch

from ..operator import MojoOperator


class MojoMoEGate(MojoOperator):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        top_k: int,
        select_method: str = "TOPKSoftmax",
        is_varlen: bool = True,
    ):
        """
        Common parameter definitions for MoE Gating operator.

        Init parameters:
        - gate_weight (torch.Tensor): Gating weight, common shape [hidden_dim, num_experts].
        - top_k (int): Number of experts to select, positive integer.
        - select_method (str): Selection method enumeration, {"TOPKSoftmax", "AuxTC"}; default "TOPKSoftmax".
        - is_varlen (bool): When True, prioritize TND (per token) computation; when False, use BSND; default True.
        - op_name (str): Operator name placeholder.

        Scope: Only covers common parameters, does not involve backend specialization or quantization implementation.
        """
        super().__init__()
        self.gate_weight = gate_weight

        self.top_k = top_k

        self.select_method = select_method
        self.is_varlen = is_varlen


class MojoMoECombine(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Common parameter definitions for MoE Combine operator.

        Init parameters:
        - ep_group: Expert parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - tp_group: Tensor parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - is_varlen (bool): When True, prioritize TND (per token) aggregation; when False, use BSND; default True.
        - op_name: Operator name placeholder.

        Scope: Only covers common semantics, does not involve backend communication or core partitioning details.
        """
        super().__init__(op_name)
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.is_varlen = is_varlen


class MojoBigEPCombine(MojoOperator):
    pass


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
        is_varlen: bool = True,
    ):
        """
        Common parameter definitions for MoE Dispatch operator.

        Init parameters:
        - ep_group: Expert parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - tp_group: Tensor parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - is_varlen (bool): When True, prioritize TND (per token) routing; when False, use BSND; default True.
        - op_name: Operator name placeholder.

        Scope: Only covers common semantics, does not involve backend communication implementation or core partitioning details.
        """
        super().__init__()
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.is_varlen = is_varlen


class MojoBigEPDispatch(MojoOperator):
    pass
