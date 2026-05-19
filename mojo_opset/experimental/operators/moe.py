from typing import Optional

import torch
import torch.nn.functional as F

from mojo_opset.core.operator import MojoOperator
from mojo_opset.core.operators.moe import _count_expert_tokens


def _validate_moe_token_count(token_count: torch.Tensor, route_count: int) -> torch.Tensor:
    token_count_i64 = token_count.to(dtype=torch.int64, device=token_count.device)
    if token_count_i64.dim() != 1:
        raise ValueError(f"token_count must be 1D, but got shape {tuple(token_count.shape)}")
    if int(token_count_i64.sum().item()) != route_count:
        raise ValueError(
            f"token_count sum must equal total routed token count {route_count}, "
            f"but got {token_count_i64.sum().item()}."
        )
    return token_count_i64


def _expand_grouped_route_param(
    param: Optional[torch.Tensor],
    token_count: torch.Tensor,
    route_shape: tuple[int, int],
) -> Optional[torch.Tensor]:
    if param is None:
        return None

    token_count_i64 = _validate_moe_token_count(token_count, route_shape[0] * route_shape[1])
    param_fp = param.float()

    if param_fp.dim() == 1:
        return param_fp.view(1, 1, -1).expand(*route_shape, -1)
    if param_fp.dim() != 2 or param_fp.size(0) != token_count_i64.numel():
        raise ValueError(
            "Grouped route param must be 2D with the first dimension equal to token_count length, "
            f"but got shape {tuple(param.shape)} and token_count length {token_count_i64.numel()}."
        )

    expanded = param_fp.repeat_interleave(token_count_i64, dim=0)
    return expanded.reshape(*route_shape, param_fp.size(-1))


def _block_dynamic_quant(input_fp: torch.Tensor, quant_block_size: int):
    if input_fp.shape[-1] % quant_block_size != 0:
        raise ValueError(
            f"Last dim {input_fp.shape[-1]} must be divisible by quant_block_size {quant_block_size}."
        )
    input_blocks = input_fp.reshape(*input_fp.shape[:-1], -1, quant_block_size)
    scale = input_blocks.abs().amax(dim=-1).clamp(min=1e-12) / 127
    quantized = torch.clamp(torch.round(input_blocks / scale.unsqueeze(-1)), -128, 127)
    return quantized.reshape_as(input_fp).to(torch.int8), scale


def _sort_moe_routes(
    hidden_states: torch.Tensor,
    top_k_gates: torch.Tensor,
    top_k_indices: torch.Tensor,
):
    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must be 2D, but got shape {tuple(hidden_states.shape)}")
    if top_k_gates.shape != top_k_indices.shape:
        raise ValueError(
            f"top_k_gates and top_k_indices must have the same shape, got "
            f"{tuple(top_k_gates.shape)} vs {tuple(top_k_indices.shape)}."
        )
    if top_k_indices.dim() != 2:
        raise ValueError(f"top_k_indices must be 2D, but got shape {tuple(top_k_indices.shape)}")

    token_num, top_k = top_k_indices.shape
    hidden_dim = hidden_states.shape[-1]

    flat_hidden = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    flat_gates = top_k_gates.reshape(-1, 1)
    flat_experts = top_k_indices.reshape(-1).to(dtype=torch.int64)
    flat_token_indices = (
        torch.arange(token_num, device=top_k_indices.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )

    _, sort_indices = flat_experts.sort(stable=True)
    sorted_experts = flat_experts.index_select(0, sort_indices)
    sorted_hidden = flat_hidden.index_select(0, sort_indices).reshape(token_num, top_k, hidden_dim)
    sorted_gates = flat_gates.index_select(0, sort_indices).reshape(token_num, top_k, 1)
    sorted_token_indices = flat_token_indices.index_select(0, sort_indices).reshape(token_num, top_k, 1)
    return sorted_hidden, sorted_gates, sorted_token_indices, sorted_experts.reshape(token_num, top_k)


class MojoMoEInitRoutingDynamicQuant(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        quant_block_size: int = 8,
        quant_dtype: torch.dtype = torch.int8,
        start_expert_id: int = 0,
        end_expert_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant_block_size = quant_block_size
        self.quant_dtype = quant_dtype
        self.start_expert_id = start_expert_id
        self.end_expert_id = num_experts if end_expert_id is None else end_expert_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
        quant_mode: int = 0,
    ):
        if quant_mode not in (0, 1):
            raise NotImplementedError(f"Unsupported quant_mode: {quant_mode}, expected 0 or 1.")

        sorted_hidden, sorted_gates, sorted_token_indices, sorted_experts = _sort_moe_routes(
            hidden_states,
            top_k_gates,
            top_k_indices,
        )

        route_hidden = sorted_hidden.float()
        if smooth_scale is not None:
            if smooth_scale.dim() != 2 or smooth_scale.size(0) != self.num_experts:
                raise ValueError(
                    "smooth_scale must be 2D with shape (num_experts, hidden_size), "
                    f"but got shape {tuple(smooth_scale.shape)} and num_experts={self.num_experts}."
                )
            route_scale = smooth_scale.index_select(0, sorted_experts.reshape(-1).to(dtype=torch.long))
            route_scale = route_scale.reshape_as(route_hidden)
            route_hidden = route_hidden * route_scale.float()

        quantized, scale = _block_dynamic_quant(route_hidden, self.quant_block_size)
        token_count = _count_expert_tokens(top_k_indices, self.num_experts)
        return (
            quantized.to(self.quant_dtype),
            sorted_gates.float(),
            sorted_token_indices.to(dtype=torch.int32),
            token_count,
            scale,
        )


class MojoFusedSwiGLUMoEScaleDynamicQuantize(MojoOperator):
    def __init__(
        self,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")
        self.quant_dtype = quant_dtype

    def forward(
        self,
        input: torch.Tensor,
        smooth_scale: Optional[torch.Tensor],
        token_count: torch.Tensor,
        beta: float = 1.0,
        quant_mode: int = 0,
    ):
        if input.dim() != 3:
            raise ValueError(f"input must be 3D, but got shape {tuple(input.shape)}")
        if input.shape[-1] % 2 != 0:
            raise ValueError(f"input last dim must be even for SwiGLU, but got {input.shape[-1]}")
        if beta == 0:
            raise ValueError("beta must be non-zero.")
        if quant_mode not in (0, 1):
            raise NotImplementedError(f"Unsupported quant_mode: {quant_mode}, expected 0 or 1.")

        route_shape = input.shape[:2]
        _validate_moe_token_count(token_count, route_shape[0] * route_shape[1])

        left, right = input.float().chunk(2, dim=-1)
        output = (F.silu(left * beta) / beta) * right

        expanded_scale = _expand_grouped_route_param(smooth_scale, token_count, route_shape)
        if expanded_scale is not None:
            output = output * expanded_scale

        scale = output.abs().amax(dim=-1).clamp(min=1e-12) / 127
        quantized = torch.clamp(torch.round(output / scale.unsqueeze(-1)), -128, 127)
        return quantized.to(self.quant_dtype), scale


__all__ = [
    "MojoMoEInitRoutingDynamicQuant",
    "MojoFusedSwiGLUMoEScaleDynamicQuantize",
]
