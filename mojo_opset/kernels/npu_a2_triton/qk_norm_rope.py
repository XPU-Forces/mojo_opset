"""Ascend A2 implementation of fused Q/K RMSNorm and half-split RoPE."""

import torch

from . import qk_norm_rope_persistent as _persistent

HEAD_DIM = _persistent.HEAD_DIM
ROTARY_DIM = _persistent.ROTARY_DIM
ROTARY_HALF = _persistent.ROTARY_HALF


def fused_qk_norm_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if q.shape[0] == 0:
        return q_out, k_out
    _persistent._launch_persistent(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        q_out,
        k_out,
        eps,
        compile_options={"num_stages": 1, "ops_reorder": True},
    )
    return q_out, k_out


@torch.library.custom_op("mojo_npu_triton_a2::fused_qk_norm_rope", mutates_args=())
def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_qk_norm_rope_fwd(q, k, q_weight, k_weight, rope_table, eps)


@fused_qk_norm_rope.register_fake
def _fused_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, rope_table, eps
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )
