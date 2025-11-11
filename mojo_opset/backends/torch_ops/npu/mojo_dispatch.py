import torch
import torch_npu

from typing import Optional, Tuple
from mojo_opset.core.moe.mojo_moe_dispatch import MojoMoEDispatch, MojoMoEBigEPDispatch

class TorchMoEDispatch(MojoMoEDispatch, default_priority=2):
    def __init__(
        self,
        num_experts: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        is_varlen: bool = True,
        op_name: str = "TorchMoEDispatch",
        layer_idx: int = 0,
    ):
        """
        Parameters:
        - num_experts (int): number of experts.
        - ep_size (int): expert distributed size, default is 1 (no ep).
        - ep_rank (int): expert rank, default is 0.
        The above 3 parameters are used to identify the valid expert idx in this rank
        - is_varlen (bool): if its' True, dispatch by TND; if its' False, dispatch by BSND. Default is True.
        - op_name(str): operator name.
        - layer_idx(int): layer index. Default is 0.
        """

        super().__init__(num_experts, ep_size, ep_rank, is_varlen, op_name, layer_idx)


    def forward_std(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        
        # API Doc link https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_init_routing_v2.md
        dispatched_hidden_states, dispatched_idx, expert_token_num, _ = torch_npu.npu_moe_init_routing_v2(
                hidden_states, 
                expert_ids, 
                expert_num = self.num_experts,
                expert_tokens_num_flag = True, # Now only support True but default is False :(
                quant_mode = -1, # No quantization, default is staitc quantization -1 :(
                active_expert_range = (self.start_expert_id_in_rank, self.end_expert_id_in_rank),
        )
        
        return dispatched_hidden_states, dispatched_idx, expert_token_num


class TorchMoEBigEPDispatch(MojoMoEBigEPDispatch, default_priority=2):
    def __init__(
        self,
        num_experts: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        is_varlen: bool = True,
        op_name: str = "TorchMoEBigEPDispatch",
        layer_idx: int = 0,
    ):
        """
        Parameters:
        - num_experts (int): number of experts.
        - ep_size (int): expert distributed size, default is 1 (no ep).
        - ep_rank (int): expert rank, default is 0.
        The above 3 parameters are used to identify the valid expert idx in this rank
        - is_varlen (bool): if its' True, dispatch by TND; if its' False, dispatch by BSND. Default is True.
        - op_name(str): operator name.
        - layer_idx(int): layer index. Default is 0.
        """
        super().__init__(num_experts, ep_size, ep_rank, is_varlen, op_name, layer_idx)
        self.group_ep_name = self.ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank)
        if self.tp_group is not None:
            self.group_tp_name = self.tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank)
        else:
            self.group_tp_name = None
    
    def forward_std(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        dp_mask: torch.Tensor = None,  
    ) -> Tuple[torch.Tensor]:
        
        # API Doc link https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_distribute_dispatch_v2.md
        dispatched_hidden_states, _, combine_token_counts, expert_token_num, \
        ep_recv_counts, tp_recv_counts, _ = torch_npu.npu_moe_distribute_dispatch_v2(
                hidden_states, 
                expert_ids, 
                self.group_ep_name, 
                self.ep_size,
                self.ep_rank,
                self.num_experts,
                scales=None,
                tp_group_name=self.group_tp_name,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                x_active_mask = dp_mask,
                quant_mode = 0
        )
        
        return dispatched_hidden_states, expert_token_num, combine_token_counts, ep_recv_counts, tp_recv_counts
