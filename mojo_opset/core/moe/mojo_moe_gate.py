import torch
from typing import Optional

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
        MoEGating 的通用参数定义。

        Init 参数：
        - gate_weight (torch.Tensor)：门控权重，常见形状 [hidden_dim, num_experts]。
        - top_k (int)：选择的专家数量，正整数。
        - select_method (str)：选择方法枚举，{"TOPKSoftmax", "AuxTC"}；默认 "TOPKSoftmax"。
        - is_varlen (bool)：为 True 时优先按 TND（逐 token）计算；为 False 时按 BSND；默认 True。
        - op_name (str)：算子名称占位。

        范围：仅覆盖通用参数，不涉及后端细分与量化实现。
        """
        super().__init__(op_name)
        if not isinstance(gate_weight, torch.Tensor):
            raise TypeError("gate_weight 必须为 torch.Tensor")
        if gate_weight.ndim not in (2, 3):
            raise ValueError("gate_weight 需为 2-D 或 3-D")
        self.gate_weight = gate_weight

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        self.top_k = top_k

        if select_method not in {"TOPKSoftmax", "AuxTC"}:
            raise ValueError('select_method 需为 {"TOPKSoftmax","AuxTC"}')
        self.select_method = select_method
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")
        self.is_varlen = is_varlen

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward 参数（通用层面）：
        - hidden_states：形状 [B, S, hidden_dim]，dtype 浮点（float16/bfloat16/float32）。
        """

        raise NotImplementedError("MojoMoEGate forward 仅进行通用参数校验，不包含具体 gating 逻辑")

    def forward_ref(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        参考实现（golden）：MoE Gate 计算，严格区分 TND/BNSD 输入。
        输入布局契约：
        - 当 is_varlen=True（TND）：仅接受 [T, D]
        - 当 is_varlen=False（BNSD）：仅接受 [B, S, D]
        公式：
        - logits = H · W_g，形状 [T,E] 或 [B,S,E]（E=num_experts）
        - 稳定化：softmax(logits - max(logits))
        - 若 select_method="TOPKSoftmax"：取 top-k 并 scatter 构造稀疏概率
        注意：返回与输入布局对应的概率张量。
        """
        x = hidden_states
        # 权重维度
        if self.gate_weight.ndim == 2:
            D, E = self.gate_weight.shape
            W = self.gate_weight
        else:
            D, E = self.gate_weight.shape[-2], self.gate_weight.shape[-1]
            W = self.gate_weight
        if x.shape[-1] != D:
            raise ValueError("hidden_states 的最后一维需与 gate_weight 的输入维一致")
        if self.is_varlen:
            # 仅接受 TND [T,D]
            if x.ndim != 2:
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(x.shape)}")
            logits = torch.matmul(x, W)  # [T,E]
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            if self.select_method == "TOPKSoftmax":
                k = min(self.top_k, E)
                values, indices = torch.topk(probs, k=k, dim=-1)
                sparse = torch.zeros_like(probs)
                sparse.scatter_(-1, indices, values)
                return sparse
            else:
                return probs
        else:
            # 仅接受 BNSD [B,S,D]
            if x.ndim != 3:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(x.shape)}")
            B, S, _ = x.shape
            logits = torch.matmul(x, W)  # [B,S,E]
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            if self.select_method == "TOPKSoftmax":
                k = min(self.top_k, E)
                values, indices = torch.topk(probs, k=k, dim=-1)
                sparse = torch.zeros_like(probs)
                sparse.scatter_(-1, indices, values)
                return sparse
            else:
                return probs
