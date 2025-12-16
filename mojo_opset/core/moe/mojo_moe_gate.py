import torch

from ..mojo_operator import MojoOperator


class MojoMoEGate(MojoOperator):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        top_k: int,
        select_method: str = "TOPKSoftmax",
        is_varlen: bool = True,
        op_name: str = "",
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
        super().__init__(op_name)
        self.gate_weight = gate_weight

        self.top_k = top_k

        self.select_method = select_method
        self.is_varlen = is_varlen
