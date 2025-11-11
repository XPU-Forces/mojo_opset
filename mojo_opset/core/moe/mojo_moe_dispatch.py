from abc import abstractmethod
from typing import Any, Tuple, Optional, Union

import torch
import torch.distributed as dist
from typing import Optional

from ..mojo_operator import MojoOperator


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        is_varlen: bool = True,
        op_name: str = "MojoMoEDispatch",
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

        super().__init__(op_name)
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen Must be type bool")
        if not is_varlen:
            raise NotImplementedError("Currently only support is_varlen=True")
        
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        if (num_experts % ep_size != 0):
            raise ValueError(f"num_experts({num_experts}) must be divisible by ep_size({ep_size})")
        
        num_experts_per_rank = num_experts // ep_size
        self.start_expert_id_in_rank = ep_rank * num_experts_per_rank
        self.end_expert_id_in_rank = (ep_rank+1) * num_experts_per_rank - 1
        self.is_varlen = is_varlen

    @abstractmethod
    def forward_std(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input Params
        - hidden_states: shape[num_tokens, hidden_dim] , dtype float16/bfloat16/float32.
        - expert_ids: shape[num_tokens, TOPK], dtype=int32.

        Return:
        - dispatched_hidden_states: shape[num_tokens * TOPK, hidden_dim] , dtype float16/bfloat16/float32.
        - dispatched_idx: correspondence between dispatched_hidden_states and hidden_states, shape[num_tokens * TOPK], dtype=int32.
        - expert_token_num: shape [B], dtype=int32. count of tokens dispatched to each expert.
        """
        raise NotImplementedError
        
    def forward_ref(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = hidden_states
        if self.is_varlen:
            # 仅接受 TND
            if x.ndim != 2:
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(x.shape)}")
            T, D = x.shape
            if expert_ids.ndim == 2:
                K = expert_ids.shape[-1]
            elif expert_ids.ndim == 1:
                K = 1
            else:
                raise ValueError("expert_ids 需为 [T] 或 [T,K]")
            y = x.unsqueeze(1).expand(T, K, D).contiguous()  # [T,K,D]
            expert_ids = expert_ids.unsqueeze(2).expand(T, K, 1).contiguous()  # [T,K,1]
            mask = (expert_ids >= self.start_expert_id_in_rank) & (expert_ids <= self.end_expert_id_in_rank)
            y = torch.where(mask, y, torch.zeros_like(y)) # [T,K,D]
            y = y.reshape(T*K, D)
            expert_ids = expert_ids.reshape(T*K, )
            expert_token_num = torch.bincount(expert_ids, minlength=self.num_experts) # [num_experts]

            return y, expert_ids, expert_token_num
        else:
            # 仅接受 BNSD
            if x.ndim != 3:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(x.shape)}")
            B, S, D = x.shape
            if expert_ids.ndim == 3:
                K = expert_ids.shape[-1]
            elif expert_ids.ndim == 2:
                K = 1
            else:
                raise ValueError("expert_ids 需为 [B,S] 或 [B,S,K]")
            y = x.unsqueeze(2).expand(B, S, K, D).contiguous()  # [B,S,K,D]
            expert_ids = expert_ids.unsqueeze(3).expand(B, S, K, 1).contiguous()  # [B,S,K,1]
            mask = (expert_ids >= self.start_expert_id_in_rank) & (expert_ids <= self.end_expert_id_in_rank)
            y = torch.where(mask, y, torch.zeros_like(y)) # [B,S,K,D]
            y = y.reshape(B*S*K, D)
            expert_ids = expert_ids.reshape(B*S*K, )
            expert_token_num = torch.bincount(expert_ids, minlength=self.num_experts) # [num_experts]

            return y, expert_ids, expert_token_num


class MojoMoEBigEPDispatch(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        ep_group,
        tp_group = None,
        is_varlen: bool = True,
        op_name: str = "MojoMoEBigEPDispatch",
        layer_idx: int = 0,
    ):
        """
        - ep_group: torch.distributed.ProcessGroup, The group used for expert parallelism. 
        - tp_group: torch.distributed.ProcessGroup, The group used for tensor parallelism. Optional.
        - is_varlen (bool): Whether to use varlen. Default is True.
        - op_name: str, operator name. 
        - layer_idx: int, layer index. Default is 0.

        When tp_group and ep_group are not None at same time, we will do TP in tp_group first, then do EP in ep_group.
        """
        super().__init__(op_name)
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.num_experts = num_experts

        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        if self.tp_group is not None:
            self.tp_size = dist.get_world_size(tp_group)
            self.tp_rank = dist.get_rank(tp_group)
        else:
            self.tp_size = 1
            self.tp_rank = 0

        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen must be type bool")
        self.is_varlen = is_varlen

    @abstractmethod
    def forward_std(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        dp_mask: Optional[torch.Tensor] = None,  # currently follow huawei implementation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input Params
        - hidden_states: shape[num_tokens, hidden_dim] , dtype float16/bfloat16/float32.
        - expert_ids: shape[num_tokens, TOPK], dtype=int32.
        - dp_mask: shape[num_tokens], dtype=bool. Optional. It indicates that which tokens are really valid in this rank \
                       to avoid the padding tokens are sent to the communication. Default is None.

        Return:
        - dispatched_hidden_states: shape[num_tokens * TOPK, hidden_dim] , dtype float16/bfloat16/float32.
        - dispatched_idx: correspondence between dispatched_hidden_states and hidden_states, shape[num_tokens * TOPK], dtype=int32.
        - expert_token_num: shape [B], dtype=int32. count of tokens dispatched to each expert.
        - ep_recv_counts: shape [num_experts], dtype=int32. count of tokens received from each expert.
        - tp_recv_counts: shape [tp_size], dtype=int32. count of tokens received from each tensor parallel rank.
        - expert_send_counts: shape [num_experts], dtype=int32. count of tokens sent to each expert.
        """
        raise NotImplementedError

    # TODO: a little complicated. finish it later
    def forward_ref(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        dp_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #pass

        x = hidden_states
        ep_send_list = []
        ep_recv_list = []

        if self.is_varlen:
            # TND only
            if x.ndim != 2:
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(x.shape)}")
            T, D = x.shape
            if expert_ids.ndim == 2:
                K = expert_ids.shape[-1]
            elif expert_ids.ndim == 1:
                K = 1
            else:
                raise ValueError("expert_ids should be [T] or [T,K]")
            
            # transpose original hidden states to list([T*K, D]) for send.
            y = x.unsqueeze(1).expand(T, K, D).contiguous()  # [T,K,D]
            expert_ids = expert_ids.unsqueeze(2).expand(T, K, 1).contiguous()  # [T,K,1]

            for rank in range(self.ep_size):
                start_expert_id_in_rank = rank * self.num_experts // self.ep_size
                end_expert_id_in_rank = (rank + 1) * self.num_experts // self.ep_size - 1
                mask = (expert_ids >= start_expert_id_in_rank) & (expert_ids <= end_expert_id_in_rank)
                y_rank = torch.where(mask, y, torch.zeros_like(y)) # [T,K,D]
                y_rank = y_rank.reshape(T*K, D)  # [T*K, D]
                expert_ids_rank = torch.where(mask, expert_ids, torch.zeros_like(expert_ids)) # [T,K,1]
                expert_ids_rank = expert_ids_rank.reshape(T*K, ) # [T*K, ]
                ep_send_list.append(y_rank) 
                ep_recv_list.append(torch.empty_like(y_rank))   

            y = torch.where(mask, y, torch.zeros_like(y)) # [T,K,D]
            y = y.reshape(T*K, D)
            expert_ids = expert_ids.reshape(T*K, )
            expert_token_num = torch.bincount(expert_ids, minlength=self.num_experts) # [num_experts]

            return y, expert_ids, expert_token_num
        else:
            # 仅接受 BNSD
            if x.ndim != 3:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(x.shape)}")
            B, S, D = x.shape
            if expert_ids.ndim == 3:
                K = expert_ids.shape[-1]
            elif expert_ids.ndim == 2:
                K = 1
            else:
                raise ValueError("expert_ids 需为 [B,S] 或 [B,S,K]")
            y = x.unsqueeze(2).expand(B, S, K, D).contiguous()  # [B,S,K,D]
            expert_ids = expert_ids.unsqueeze(3).expand(B, S, K, 1).contiguous()  # [B,S,K,1]
            mask = (expert_ids >= self.start_expert_id_in_rank) & (expert_ids <= self.end_expert_id_in_rank)
            y = torch.where(mask, y, torch.zeros_like(y)) # [B,S,K,D]
            y = y.reshape(B*S*K, D)
            expert_ids = expert_ids.reshape(B*S*K, )
            expert_token_num = torch.bincount(expert_ids, minlength=self.num_experts) # [num_experts]

            return y, expert_ids, expert_token_num
