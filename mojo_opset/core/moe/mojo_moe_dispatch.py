import torch
from typing import Optional

from ..mojo_operator import MojoOperator


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        MoEDispatch 的通用参数定义。

        Init 参数：
        - ep_group：专家并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - tp_group：张量并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - is_varlen (bool)：为 True 时优先按 TND（逐 token）路由；为 False 时按 BSND；默认 True。
        - op_name：算子名称占位。

        范围：仅覆盖通用语义，不涉及后端通信实现与分核细节。
        """
        super().__init__(op_name)
        self.ep_group = ep_group
        self.tp_group = tp_group
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")
        self.is_varlen = is_varlen

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward 参数（通用层面）：
        - hidden_states：形状 [B, S, hidden_dim]，dtype 浮点（float16/bfloat16/float32）。
        - expert_ids：专家 id 列表，形状 [B, S] 或 [B, S, K]，dtype=int32。
        - active_mask：可选，形状 [B] 或 [B, S]，dtype=bool；用于 DP 空跑场景处理。
        """

        raise NotImplementedError("MojoMoEDispatch forward 仅进行通用参数校验，不包含具体分发逻辑")

    def forward_ref(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参考实现（golden）：MoE Dispatch，严格区分 TND/BNSD 输入。
        语义：按照 expert_ids 路由，将每个位置的隐藏状态复制到所选专家通道。
        输入布局契约：
        - 当 is_varlen=True（TND）：hidden_states=[T,D]；expert_ids=[T] 或 [T,K]
        - 当 is_varlen=False（BNSD）：hidden_states=[B,S,D]；expert_ids=[B,S] 或 [B,S,K]
        active_mask：若提供，TND 下为 [T]，BNSD 下为 [B] 或 [B,S]；inactive 位置置零。
        返回形状：TND → [T,K,D]；BNSD → [B,S,K,D]。
        """
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
            if active_mask is not None:
                if active_mask.ndim != 1 or active_mask.shape[0] != T:
                    raise ValueError("TND 下 active_mask 需为 [T]")
                mask = active_mask[:, None, None]
                y = torch.where(mask, y, torch.zeros_like(y))
            return y
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
            y = x.unsqueeze(2).expand(B, S, K, D).contiguous()
            if active_mask is not None:
                if active_mask.ndim == 1:
                    mask = active_mask[:, None, None, None]
                elif active_mask.ndim == 2:
                    mask = active_mask[:, :, None, None]
                else:
                    raise ValueError("active_mask 需为 [B] 或 [B,S]")
                y = torch.where(mask, y, torch.zeros_like(y))
            return y


class MojoBigEPDispatch(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        MoEDispatch 的通用参数定义。

        Init 参数：
        - ep_group：专家并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - tp_group：张量并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - is_varlen (bool)：为 True 时优先按 TND（逐 token）路由；为 False 时按 BSND；默认 True。
        - op_name：算子名称占位。

        范围：仅覆盖通用语义，不涉及后端通信实现与分核细节。
        """
        super().__init__(op_name)
        self.ep_group = ep_group
        self.tp_group = tp_group
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")
        self.is_varlen = is_varlen

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward 参数（通用层面）：
        - hidden_states：形状 [B, S, hidden_dim]，dtype 浮点（float16/bfloat16/float32）。
        - expert_ids：专家 id 列表，形状 [B, S] 或 [B, S, K]，dtype=int32。
        - active_mask：可选，形状 [B] 或 [B, S]，dtype=bool；用于 DP 空跑场景处理。
        """

        raise NotImplementedError("MojoMoEDispatch forward 仅进行通用参数校验，不包含具体分发逻辑")