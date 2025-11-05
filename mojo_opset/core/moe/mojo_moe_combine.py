import torch
from typing import Optional

from ..mojo_operator import MojoOperator


class MojoMoECombine(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        MoECombine 的通用参数定义。

        Init 参数：
        - ep_group：专家并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - tp_group：张量并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - is_varlen (bool)：为 True 时优先按 TND（逐 token）聚合；为 False 时按 BSND；默认 True。
        - op_name：算子名称占位。

        范围：仅覆盖通用语义，不涉及后端通信与分核细节。
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
        expert_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward 参数（通用层面）：
        - hidden_states：形状 [B, S, D_out] 或专家输出聚合的张量；dtype 浮点。
        - expert_weights：形状 [B, S, K]，dtype 浮点；用于加权合并。
        - expert_ids：形状 [B, S, K]，dtype=int32；与 expert_weights 对齐。
        - active_mask：可选，形状 [B] 或 [B, S]，dtype=bool。
        """

        raise NotImplementedError("MojoMoECombine forward 仅进行通用参数校验，不包含具体合并逻辑")

    def forward_ref(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参考实现（golden）：MoE Combine，严格区分 TND/BNSD 输入。
        语义：将专家输出按权重聚合。
        输入布局契约：
        - 当 is_varlen=True（TND）：
          · hidden_states=[T,K,D] 与 expert_weights=[T,K]；或 hidden_states=[T,D]（视作已合并）
        - 当 is_varlen=False（BNSD）：
          · hidden_states=[B,S,K,D] 与 expert_weights=[B,S,K]；或 hidden_states=[B,S,D]
        active_mask：若提供，TND 下为 [T]，BNSD 下为 [B] 或 [B,S]；inactive 位置置零。
        返回：TND → [T,D]；BNSD → [B,S,D]。
        """
        hs = hidden_states
        if self.is_varlen:
            # 仅接受 TND
            if hs.ndim == 3:
                T, K, D = hs.shape
                if expert_weights.ndim != 2 or expert_weights.shape != (T, K):
                    raise ValueError("TND 下 expert_weights 需为 [T,K]")
                w = expert_weights.unsqueeze(-1)  # [T,K,1]
                y = (hs * w).sum(dim=1)  # [T,D]
            elif hs.ndim == 2:
                y = hs
            else:
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(hs.shape)}")
            if active_mask is not None:
                if active_mask.ndim != 1 or active_mask.shape[0] != y.shape[0]:
                    raise ValueError("TND 下 active_mask 需为 [T]")
                y = torch.where(active_mask[:, None], y, torch.zeros_like(y))
            return y
        else:
            # 仅接受 BNSD
            if hs.ndim == 4:
                B, S, K, D = hs.shape
                if expert_weights.shape[:2] != (B, S) or expert_weights.shape[-1] != K:
                    raise ValueError("expert_weights 需为 [B,S,K]")
                w = expert_weights.unsqueeze(-1)  # [B,S,K,1]
                y = (hs * w).sum(dim=2)  # [B,S,D]
            elif hs.ndim == 3:
                y = hs
            else:
                raise ValueError("hidden_states 需为 [B,S,D] 或 [B,S,K,D]")
            if active_mask is not None:
                if active_mask.ndim == 1:
                    mask = active_mask[:, None, None]
                elif active_mask.ndim == 2:
                    mask = active_mask[:, :, None]
                else:
                    raise ValueError("active_mask 需为 [B] 或 [B,S]")
                y = torch.where(mask, y, torch.zeros_like(y))
            return y


class MojoBigEPCombine(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        MoECombine 的通用参数定义。

        Init 参数：
        - ep_group：专家并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - tp_group：张量并行进程组（torch.distributed.ProcessGroup 占位），可选。
        - is_varlen (bool)：为 True 时优先按 TND（逐 token）聚合；为 False 时按 BSND；默认 True。
        - op_name：算子名称占位。

        范围：仅覆盖通用语义，不涉及后端通信与分核细节。
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
        expert_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward 参数（通用层面）：
        - hidden_states：形状 [B, S, D_out] 或专家输出聚合的张量；dtype 浮点。
        - expert_weights：形状 [B, S, K]，dtype 浮点；用于加权合并。
        - expert_ids：形状 [B, S, K]，dtype=int32；与 expert_weights 对齐。
        - active_mask：可选，形状 [B] 或 [B, S]，dtype=bool。
        """

        raise NotImplementedError("MojoMoECombine forward 仅进行通用参数校验，不包含具体合并逻辑")