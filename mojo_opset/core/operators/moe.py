from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoMoEGating(MojoOperator):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Gating operator.

        Init parameters:
        - gate_weight (torch.Tensor): Gating weight, common shape [hidden_dim, num_experts].
        - top_k (int): Number of experts to select, positive integer.

        Scope: Only covers common parameters, does not involve backend specialization or quantization implementation.
        """
        super().__init__()
        self.gate_weight = torch.nn.Parameter(torch.empty(hidden_size, num_experts, **self.tensor_factory_kwargs))
        self.top_k = top_k
        
    def forward(self, hidden_states: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for MoE Gating operator.

        Input:
        - hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Output:
        - torch.Tensor: Output tensor of shape [batch_size, seq_len, num_experts].
        """
        
        raise NotImplementedError("MojoMoEGating forward 仅进行通用参数校验,不包含具体 gating 逻辑")


    def forward_ref(self, hidden_states: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for MoE Gating operator.

        Input:
        - hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Output:
        - torch.Tensor: Output tensor of shape [batch_size, seq_len, num_experts].
        """
        gate_logits = torch.matmul(hidden_states, self.gate_weight)
        gate_logits = torch.softmax(gate_logits, dim=-1)
        top_k_logits, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_weights = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True)
        return indices, gate_weights


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
    ):
        """
        Common parameter definitions for MoE Dispatch operator.

        Init parameters:
        - ep_group: Expert parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - tp_group: Tensor parallel process group (torch.distributed.ProcessGroup placeholder), optional.

        Scope: Only covers common semantics, does not involve backend communication implementation or core partitioning details.
        """
        super().__init__()
        self.ep_group = ep_group
        self.tp_group = tp_group


class MojoMoECombine(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
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
        super().__init__()
        self.ep_group = ep_group
        self.tp_group = tp_group

