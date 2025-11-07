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