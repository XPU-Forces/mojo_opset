from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..operator import MojoOperator


class MojoMoEGatingTopK(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        k: int,
        *,
        bias: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        tid2eid: Optional[torch.Tensor] = None,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        out_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MoE gating top-k operator (reference implementation).

        This operator mirrors the behavior used by Ascend custom kernel
        ``torch.ops.custom.npu_moe_gating_top_k`` in CANN recipes examples.

        Args:
            x: 2D tensor [num_tokens, num_experts].
            k: top-k experts to select.
            bias: optional bias tensor broadcastable to x.
            input_ids: optional token ids [num_tokens].
            tid2eid: optional lookup table [vocab_size, k] (int32). When both
                ``input_ids`` and ``tid2eid`` are provided, expert indices are
                taken from ``tid2eid[input_ids]`` instead of top-k on x.
            k_group/group_count/group_select_mode/renorm: reserved for grouped
                routing modes; only the common simple mode is supported by the
                reference implementation.
            norm_type:
                0: softmax
                1: sigmoid
                2: softplus -> sqrt
            out_flag: whether to return the 3rd tensor output. (Kernel always
                returns it; reference returns a float32 tensor regardless.)
            routed_scaling_factor/eps: output scaling parameters.

        Returns:
            y_out: top-k routing weights [num_tokens, k] with dtype matching x.
            expert_idx_out: expert indices [num_tokens, k] int32.
            norm_out: normalized full routing weights [num_tokens, num_experts] float32.
        """
        if x.dim() != 2:
            raise ValueError(f"MojoMoEGatingTopK: x must be 2D, got shape {tuple(x.shape)}")
        if k <= 0 or k > x.size(-1):
            raise ValueError(f"MojoMoEGatingTopK: k must be in (0, {x.size(-1)}], got {k}")

        if (k_group, group_count, group_select_mode) != (1, 1, 0):
            raise NotImplementedError(
                "MojoMoEGatingTopK reference only supports k_group=1, group_count=1, group_select_mode=0."
            )
        if renorm not in (0, 1):
            raise ValueError(f"MojoMoEGatingTopK: renorm must be 0 or 1, got {renorm}")

        x_fp = x.float()

        if norm_type == 0:
            norm_out = torch.softmax(x_fp, dim=-1)
        elif norm_type == 1:
            norm_out = torch.sigmoid(x_fp)
        elif norm_type == 2:
            norm_out = torch.sqrt(torch.nn.functional.softplus(x_fp))
        else:
            raise ValueError(f"MojoMoEGatingTopK: unsupported norm_type={norm_type}")

        original_norm_out = norm_out
        if bias is not None:
            norm_out = norm_out + bias.float()

        if (input_ids is not None) and (tid2eid is not None):
            if input_ids.dim() != 1 or input_ids.numel() != x.size(0):
                raise ValueError(
                    "MojoMoEGatingTopK: input_ids must be 1D with length equal to x.size(0), "
                    f"got shape {tuple(input_ids.shape)} and x.shape {tuple(x.shape)}."
                )
            expert_idx_out = tid2eid.index_select(0, input_ids.to(dtype=torch.int64)).to(dtype=torch.int32)
            if expert_idx_out.shape != (x.size(0), k):
                raise ValueError(
                    "MojoMoEGatingTopK: tid2eid[input_ids] must have shape [num_tokens, k], "
                    f"got {tuple(expert_idx_out.shape)} expected {(x.size(0), k)}."
                )
        else:
            _, expert_idx_out = torch.topk(norm_out, k, dim=-1, largest=True, sorted=True)
            expert_idx_out = expert_idx_out.to(dtype=torch.int32)

        y_out = torch.gather(original_norm_out, dim=1, index=expert_idx_out.to(dtype=torch.int64))

        # Match example reference behavior: renormalize only for non-softmax norms.
        if norm_type != 0:
            denom = y_out.sum(dim=-1, keepdim=True) + float(eps)
            y_out = y_out / denom

        y_out = (y_out * float(routed_scaling_factor)).to(dtype=x.dtype)

        # Kernel's 3rd output is always present. Keep it to preserve the signature.
        norm_out_fp32 = original_norm_out.to(dtype=torch.float32)
        if not out_flag:
            return y_out, expert_idx_out, norm_out_fp32
        return y_out, expert_idx_out, norm_out_fp32

