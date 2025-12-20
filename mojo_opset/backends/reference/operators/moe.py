from typing import Optional

import torch

from mojo_opset.core import LAST_PRIORITY
from mojo_opset.core import MojoMoECombine
from mojo_opset.core import MojoMoEDispatch
from mojo_opset.core import MojoMoEGate


class RefMoECombine(MojoMoECombine, default_priority=LAST_PRIORITY):
    def forward_std(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hs = hidden_states
        if self.is_varlen:
            if hs.ndim == 3:
                T, K, D = hs.shape
                w = expert_weights.unsqueeze(-1)  # [T,K,1]
                y = (hs * w).sum(dim=1)  # [T,D]
            elif hs.ndim == 2:
                y = hs
            else:
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(hs.shape)}")
            if active_mask is not None:
                y = torch.where(active_mask[:, None], y, torch.zeros_like(y))
            return y
        else:
            if hs.ndim == 4:
                B, S, K, D = hs.shape
                w = expert_weights.unsqueeze(-1)  # [B,S,K,1]
                y = (hs * w).sum(dim=2)  # [B,S,D]
            elif hs.ndim == 3:
                y = hs
            else:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(hs.shape)}")
            if active_mask is not None:
                if active_mask.ndim == 1:
                    mask = active_mask[:, None, None]
                elif active_mask.ndim == 2:
                    mask = active_mask[:, :, None]
                else:
                    raise ValueError("active_mask shape must be [B] or [B,S]")
                y = torch.where(mask, y, torch.zeros_like(y))
            return y


class RefMoEDispatch(MojoMoEDispatch, default_priority=LAST_PRIORITY):
    def forward_std(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = hidden_states
        if self.is_varlen:
            if x.ndim != 2:
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(x.shape)}")
            T, D = x.shape
            if expert_ids.ndim == 2:
                K = expert_ids.shape[-1]
            elif expert_ids.ndim == 1:
                K = 1
            else:
                raise ValueError("expert_ids should be [T] or [T,K]")
            y = x.unsqueeze(1).expand(T, K, D).contiguous()  # [T,K,D]
            if active_mask is not None:
                if active_mask.ndim != 1 or active_mask.shape[0] != T:
                    raise ValueError("active_mask should be [T]")
                mask = active_mask[:, None, None]
                y = torch.where(mask, y, torch.zeros_like(y))
            return y
        else:
            if x.ndim != 3:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(x.shape)}")
            B, S, D = x.shape
            if expert_ids.ndim == 3:
                K = expert_ids.shape[-1]
            elif expert_ids.ndim == 2:
                K = 1
            else:
                raise ValueError("expert_ids should be [B,S] or [B,S,K]")
            y = x.unsqueeze(2).expand(B, S, K, D).contiguous()
            if active_mask is not None:
                if active_mask.ndim == 1:
                    mask = active_mask[:, None, None, None]
                elif active_mask.ndim == 2:
                    mask = active_mask[:, :, None, None]
                else:
                    raise ValueError("active_mask should be [B] or [B,S]")
                y = torch.where(mask, y, torch.zeros_like(y))
            return y


class RefMoEGate(MojoMoEGate, default_priority=LAST_PRIORITY):
    def forward_std(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        if self.gate_weight.ndim == 2:
            D, E = self.gate_weight.shape
            W = self.gate_weight
        else:
            D, E = self.gate_weight.shape[-2], self.gate_weight.shape[-1]
            W = self.gate_weight
        if self.is_varlen:
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
