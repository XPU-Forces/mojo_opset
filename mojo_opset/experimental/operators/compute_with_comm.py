from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.distributed_c10d import _get_default_group

from mojo_opset.core.operator import MojoOperator


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


class MojoFusedAGScaleQuant(MojoOperator):
    def __init__(
        self,
        *,
        team_size: int = 1,
        quant_mode: str = "per_token",
        norm_mode: str = "none",
        eps: float = 1e-5,
        max_tokens: Optional[int] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        comm_context=None,
        **kwargs,
    ):
        """
        Fused AllGather-scale exchange + optional RMSNorm + per-token int8 quantization.

        Args:
            team_size (int): Communication team size.
            quant_mode (str): Quantization mode. Only ``"per_token"`` is supported.
            norm_mode (str): Normalization mode. Supports ``"none"`` and ``"rmsnorm"``.
            eps (float): Epsilon for RMSNorm.
            max_tokens (Optional[int]): Maximum token count expected by backend
                implementations that initialize communication buffers in ``__init__``.
            process_group (Optional[ProcessGroup]): Distributed group for the torch reference.
                ``None`` means the default group.
            comm_context: Optional runtime/context object for backend implementations.
        """
        super().__init__(**kwargs)
        if quant_mode not in ["per_token"]:
            raise NotImplementedError(f"quant_mode {quant_mode} not supported")
        if norm_mode not in ["none", "rmsnorm"]:
            raise NotImplementedError(f"norm_mode {norm_mode} not supported")
        if team_size < 1:
            raise ValueError(f"team_size must be positive, but got {team_size}")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, but got {max_tokens}")

        self.team_size = team_size
        self.quant_mode = quant_mode
        self.norm_mode = norm_mode
        self.eps = eps
        self.max_tokens = max_tokens
        self.process_group = process_group
        self.comm_context = comm_context

    def _team_max_scale(self, scale: torch.Tensor) -> torch.Tensor:
        if self.team_size == 1 or not _is_dist_initialized():
            return scale

        process_group = self.process_group or _get_default_group()
        world_size = dist.get_world_size(group=process_group)
        if world_size == 1:
            return scale
        if world_size != self.team_size:
            raise ValueError(f"process group world size must match team_size={self.team_size}, but got {world_size}")

        gathered = [torch.empty_like(scale) for _ in range(world_size)]
        dist.all_gather(gathered, scale.contiguous(), group=process_group)
        return torch.stack(gathered, dim=0).amax(dim=0)

    def forward(
        self,
        input: torch.Tensor,
        quant_scale: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
    ):
        if input.dim() not in [3, 4]:
            raise ValueError(f"input must be 3-D or 4-D, but got dim={input.dim()}")

        head_num = input.shape[-2]
        head_dim = input.shape[-1]
        hidden_size = head_num * head_dim
        if quant_scale.numel() != hidden_size:
            raise ValueError(f"quant_scale numel must be {hidden_size}, but got {quant_scale.numel()}")
        if self.norm_mode == "rmsnorm" and norm_weight is not None and norm_weight.numel() != head_dim:
            raise ValueError(f"norm_weight numel must be {head_dim}, but got {norm_weight.numel()}")

        input_fp = input.float()
        if self.norm_mode == "rmsnorm":
            weight = norm_weight.float() if norm_weight is not None else None
            input_fp = F.rms_norm(input_fp, (head_dim,), weight=weight, eps=self.eps)

        rows = input_fp.numel() // hidden_size
        if self.max_tokens is not None and rows > self.max_tokens:
            raise ValueError(f"input token count {rows} exceeds max_tokens={self.max_tokens}")
        scaled = input_fp.reshape(rows, hidden_size) * quant_scale.float().reshape(1, hidden_size)
        scale = scaled.abs().amax(dim=-1).clamp(min=1e-12) / 127
        scale = self._team_max_scale(scale)
        quantized = torch.clamp(torch.round(scaled / scale.unsqueeze(-1)), -128, 127).to(torch.int8)

        return quantized, scale

    def extra_repr(self) -> str:
        return (
            f"{self.team_size=}, {self.quant_mode=}, {self.norm_mode=}, {self.eps=}, {self.max_tokens=}"
        ).replace("self.", "")


__all__ = ["MojoFusedAGScaleQuant"]
