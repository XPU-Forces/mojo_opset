from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operator import MojoOperator
from .quantize import MojoMoEDynamicQuant


class MojoMoE(MojoOperator):
    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size=None,
        activation: str = "swiglu",
        **kwargs,
    ):
        super().__init__()
        if activation != "swiglu":
            raise NotImplementedError(f"MojoMoe: Activation {activation} is not supported.")

        for k in ("ep_rank", "ep_size"):
            if k in kwargs:
                raise ValueError(f"MojoMoE: {k} is not supported; use ParallelStyle to set expert partition.")

        # NOTE: in some cases, branches may have different expert num or topk
        self.num_experts = num_experts
        if intermediate_size is None:
            raise ValueError("MojoMoE: intermediate_size must be provided.")

        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gating = MojoMoEGating._registry.get(self._backend)(
            hidden_size=self.hidden_size, num_experts=self.num_experts, top_k=self.top_k, **kwargs
        )
        self.dispatch = MojoMoEDispatch._registry.get(self._backend)(num_experts=self.num_experts, **kwargs)
        self.experts = MojoExperts._registry.get(self._backend)(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=activation,
            **kwargs,
        )
        self.combine = MojoMoECombine._registry.get(self._backend)(multiply_by_gates=True, **kwargs)

    def forward(self, hidden_states):
        # hidden_states: [num_tokens, H]
        top_k_indices, top_k_gates = self.gating(hidden_states)
        # top_k_indices, top_k_gates: [num_tokens, top_k]
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states, top_k_gates, top_k_indices
        )
        # sorted_hidden_states: [local_tokens, H]
        # tokens_per_expert: [num_experts]
        # sorted_gates: [local_tokens, 1]
        # token_indices: [local_tokens]
        expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        # expert_outputs: [local_tokens, H]
        output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        # combined: [num_tokens, H]
        return combined


class MojoQuantMoE(MojoOperator):
    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size=None,
        activation: str = "swiglu",
        quant_dtype: torch.dtype = torch.int8,
        quant_group_size: int = -1,
        weight_bits: int = 8,
        **kwargs,
    ):
        super().__init__()
        if activation != "swiglu":
            raise NotImplementedError(f"MojoQuantMoE: Activation {activation} is not supported.")
        if quant_dtype != torch.int8:
            raise NotImplementedError(f"MojoQuantMoE: quant_dtype must be 'int8', got {quant_dtype}.")
        if weight_bits not in (4, 8):
            raise ValueError(f"MojoQuantMoE: weight must be w4 or w8")
        for k in ("ep_rank", "ep_size"):
            if k in kwargs:
                raise ValueError(f"MojoQuantMoE: {k} is not supported; use ParallelStyle to set expert partition.")

        if intermediate_size is None:
            raise ValueError("MojoQuantMoE: intermediate_size must be provided.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quant_dtype = quant_dtype
        self.quant_group_size = quant_group_size
        self.weight_bits = weight_bits

        self.gating = MojoMoEGating._registry.get(self._backend)(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            **kwargs,
        )
        self.dispatch = MojoMoEDispatch._registry.get(self._backend)(num_experts=self.num_experts, **kwargs)
        self.experts = MojoQuantExperts._registry.get(self._backend)(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=activation,
            quant_dtype=quant_dtype,
            quant_group_size=quant_group_size,
            weight_bits=weight_bits,
            **kwargs,
        )
        self.combine = MojoMoECombine._registry.get(self._backend)(multiply_by_gates=True, **kwargs)

    def forward(self, hidden_states):
        top_k_indices, top_k_gates = self.gating(hidden_states)
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states,
            top_k_gates,
            top_k_indices,
        )
        expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
        return self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)


class MojoMoEGating(MojoOperator):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Gating operator.

        Init parameters:
        - gate_weight (torch.Tensor): Gating weight, common shape [hidden_dim, num_experts].
        - top_k (int): Number of experts to select, positive integer.

        Scope: Only covers common parameters, does not involve backend specialization or quantization implementation.
        """
        super().__init__(**kwargs)
        self.gate_weight = torch.nn.Parameter(torch.empty(hidden_size, num_experts, **self.tensor_factory_kwargs))
        self.top_k = top_k
        setattr(self.gate_weight, "force_dtype", torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for MoE Gating operator.

        Input:
        - hidden_states (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].

        Output:
        - top_k_indices (torch.Tensor): Output tensor of shape [num_tokens, top_k].
        - top_k_gates (torch.Tensor): Output tensor of shape [num_tokens, top_k].
        """
        assert self.gate_weight.dtype == torch.float32
        gate_logits = torch.matmul(hidden_states.float(), self.gate_weight)
        gate_logits = torch.softmax(gate_logits, dim=-1)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True)
        return top_k_indices, top_k_gates

    def extra_repr(self) -> str:
        hidden_size = self.gate_weight.size(0)
        num_experts = self.gate_weight.size(1)
        return f"{hidden_size=}, {num_experts=}, {self.top_k=}".replace("self.", "")


def _count_expert_tokens(top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    flat_indices = top_k_indices.reshape(-1).to(dtype=torch.int64, device=top_k_indices.device)
    return torch.bincount(flat_indices, minlength=num_experts).to(dtype=torch.int32, device=top_k_indices.device)


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


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Dispatch operator.

        Init parameters:
        - num_experts (int): Number of experts.

        Scope: Only covers common semantics, does not involve backend communication implementation or core partitioning details.
        """
        super().__init__(**kwargs)
        self.num_experts = num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        """
        Forward pass for MoE Dispatch operator.

        Input:
        - hidden_states (torch.Tensor): Input tensor.
        - top_k_gates (torch.Tensor): Top-k gating weights.
        - top_k_indices (torch.Tensor): Top-k expert indices.

        Output:
        - sorted_hidden_states: Sorted inputs for experts.
        - tokens_per_expert: Count of tokens for each expert.
        - sorted_gates: Packed gating weights.
        - token_indices: Indices for packing/unpacking.
        """
        batch_token_indices = (
            torch.arange(0, hidden_states.shape[0], device=hidden_states.device, dtype=top_k_indices.dtype)
            .unsqueeze(1)
            .repeat(1, top_k_indices.shape[-1])
            .flatten()
        )
        # batch_token_indices: [BS * top_k]
        flat_top_k_gates = top_k_gates.reshape(-1, 1)
        flat_top_k_indices = top_k_indices.flatten()
        sorted_experts, expert_sort_indices = flat_top_k_indices.sort()

        token_indices = batch_token_indices[expert_sort_indices]
        tokens_per_expert = _count_expert_tokens(flat_top_k_indices, self.num_experts)

        sorted_gates = flat_top_k_gates[expert_sort_indices, :]
        sorted_hidden_states = hidden_states[token_indices].squeeze(1)
        return sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices


class MojoExperts(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Experts operator.

        Init parameters:
        - num_experts (int): Number of experts.
        - hidden_size (int): Hidden size of the model.
        - ffn_hidden_size (int): Hidden size of the feed-forward network within each expert.
        - activation (str): Activation function to use.

        Scope: Only covers common parameters, does not involve backend specialization.
        """
        super().__init__(**kwargs)
        if activation != "swiglu":
            raise NotImplementedError(f"MojoExperts: Activation {activation} is not supported.")
        self.activation = activation

        self.up_proj_weight = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, hidden_size, **self.tensor_factory_kwargs)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, **self.tensor_factory_kwargs)
        )

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ):
        # Mocked GroupGemm
        expert_inputs = torch.split(sorted_hidden_states, tokens_per_expert.tolist(), dim=0)
        num_experts = len(expert_inputs)

        fc1_outs = [F.linear(expert_inputs[i].float(), self.up_proj_weight[i].float()) for i in range(num_experts)]
        activated_outs = []
        for fc1_out in fc1_outs:
            gate_proj, up_proj = fc1_out.chunk(2, dim=-1)
            activated_outs.append(F.silu(gate_proj) * up_proj)

        fc2_outs = [F.linear(activated_outs[i], self.down_proj_weight[i].float()) for i in range(num_experts)]
        return torch.cat(fc2_outs, dim=0).to(sorted_hidden_states.dtype)

class MojoQuantExperts(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
        quant_dtype: torch.dtype = torch.int8,
        quant_group_size: int = -1,
        weight_bits: int = 8,
        **kwargs,
    ):
        """
        Quantized MoE Experts reference.

        The input activation is expected to be dynamically quantized before this
        operator. For ``weight_bits=4``, expert weights are signed int4
        values packed two per int8 element along the output/channel dimension,
        matching checkpoint tensors shaped ``[num_experts, output_dim // 2,
        input_dim]``. Weight scales use ``[num_experts, output_dim, group_num]``
        and are expected to be the offline product of per-channel
        ``weight_qscale`` and per-group scales; this module only observes the
        grouped accumulation contract.
        """
        super().__init__(**kwargs)
        if activation != "swiglu":
            raise NotImplementedError(f"MojoQuantExperts: Activation {activation} is not supported.")
        if quant_dtype != torch.int8:
            raise ValueError(f"MojoQuantExperts: quant_dtype must be 'int8', got {quant_dtype}.")
        if weight_bits not in (4, 8):
            raise NotImplementedError("MojoQuantExperts currently only supports w4 or w8.")
        if weight_bits == 4 and (hidden_size % 2 != 0 or intermediate_size % 2 != 0):
            raise ValueError("MojoQuantExperts requires even hidden_size and intermediate_size for int4 packing.")
        
        self.activation = activation
        self.quant_dtype = quant_dtype
        self.quant_group_size = quant_group_size
        self.weight_bits = weight_bits
        assert quant_dtype == torch.int8
        bits = 8
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -(2 ** (bits - 1))
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.up_proj_quantize = MojoMoEDynamicQuant._registry.get(self._backend)(
            num_experts, hidden_size,
        )

        self.down_proj_quantize = MojoMoEDynamicQuant._registry.get(self._backend)(
            num_experts, intermediate_size,
        )

        if weight_bits == 8:
            self.register_buffer(
                "up_proj_weight",
                torch.empty((num_experts, intermediate_size * 2, hidden_size), dtype=torch.int8),
            )
            self.register_buffer(
                "down_proj_weight",
                torch.empty((num_experts, hidden_size, intermediate_size), dtype=torch.int8),
            )
        else:
            self.register_buffer(
                "up_proj_weight",
                torch.empty((num_experts, intermediate_size * 2 // 2, hidden_size), dtype=torch.int8),
            )
            self.register_buffer(
                "down_proj_weight",
                torch.empty((num_experts, hidden_size // 2, intermediate_size), dtype=torch.int8),
            )

        if quant_group_size > 0:
            up_proj_groups = (hidden_size + quant_group_size - 1) // quant_group_size
            self.up_proj_weight_scale = nn.Parameter(
                torch.empty(
                    (num_experts, intermediate_size * 2, up_proj_groups),
                    dtype=torch.bfloat16,
                ),
            )

            down_proj_groups = (intermediate_size + quant_group_size - 1) // quant_group_size
            self.down_proj_weight_scale = nn.Parameter(
                torch.empty(
                    (num_experts, hidden_size, down_proj_groups),
                    dtype=torch.bfloat16,
                ),
            )
        else:
            self.up_proj_weight_scale = nn.Parameter(
                torch.empty(
                    (num_experts, intermediate_size * 2),
                    dtype=torch.bfloat16,
                ),
            )

            self.down_proj_weight_scale = nn.Parameter(
                torch.empty(
                    (num_experts, hidden_size), 
                    dtype=torch.bfloat16,
                ),
            )

    def _unpack_weight(self, weight):
        assert weight.ndim == 2
        unpacked_weight = torch.empty(weight.shape[0] * 2, weight.shape[1], device=weight.device, dtype=torch.int8)
        unpacked_weight[::2] = weight & 0x0F
        unpacked_weight[1::2] = (weight >> 4) & 0x0F
        unpacked_weight = torch.where(unpacked_weight >= 8, unpacked_weight - 16, unpacked_weight)
        return unpacked_weight

    def _quant_linear(
        self, 
        input_int8: torch.Tensor, 
        input_scale: torch.Tensor,
        expert_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        if self.weight_bits == 4:
            expert_weight = self._unpack_weight(expert_weight)
        
        assert input_scale.ndim == 2 and input_scale.shape[1] == 1
        if self.quant_group_size > 0:
            x_int8_groups = torch.split(input_int8, self.quant_group_size, dim=-1)
            weight_int8_groups = torch.split(expert_weight, self.quant_group_size, dim=-1)
            output_groups = [torch.mul(x_int8_group.int().unsqueeze(-2), weight_int8_group.int().unsqueeze(-3)).float().sum(dim=-1)
                             for x_int8_group, weight_int8_group 
                             in zip(x_int8_groups, weight_int8_groups)]
            output = torch.stack(output_groups, dim=-1)
            output = (output * weight_scale * input_scale.unsqueeze(-1)).sum(-1)
        else:
            output = torch.mul(input_int8.int().unsqueeze(-2), expert_weight.int().unsqueeze(-3)).float().sum(dim=-1) * weight_scale * input_scale

        return output.to(output_dtype)

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ):
        """
        Args:
            sorted_hidden_states (torch.Tensor): bf16 activations ``(tokens, H)``.
            tokens_per_expert (torch.Tensor): Token count per expert.

        Returns:
            torch.Tensor: Dequantized bf16/fp output for MoE combine, shape ``(tokens, H)``.
        """
        x_int8, x_scale = self.up_proj_quantize(sorted_hidden_states, tokens_per_expert)

        x_int8_list = torch.split(x_int8, tokens_per_expert.tolist(), dim=0)
        x_scale_list = torch.split(x_scale, tokens_per_expert.tolist(), dim=0)
        num_experts = tokens_per_expert.size(0)

        activated_outs = []
        for expert_idx in range(num_experts):
            x_int8_i = x_int8_list[expert_idx]
            x_scale_i = x_scale_list[expert_idx]
            if x_int8_i.shape[0] == 0:
                activated_outs.append(torch.empty(0, self.intermediate_size, device=sorted_hidden_states.device, dtype=torch.float))
                continue

            fc1_out = self._quant_linear(
                x_int8_i,
                x_scale_i, 
                self.up_proj_weight[expert_idx], 
                self.up_proj_weight_scale[expert_idx],
                sorted_hidden_states.dtype,
            )
            gate_proj, up_proj = fc1_out.float().chunk(2, dim=-1)
            activated_outs.append(F.silu(gate_proj) * up_proj)
        activated = torch.cat(activated_outs, dim=0)

        y_int8, y_scale = self.down_proj_quantize(activated, tokens_per_expert)
        y_int8_list = torch.split(y_int8, tokens_per_expert.tolist(), dim=0)
        y_scale_list = torch.split(y_scale, tokens_per_expert.tolist(), dim=0)
        outputs = []
        for expert_idx in range(num_experts):
            y_int8_i = y_int8_list[expert_idx]
            y_scale_i = y_scale_list[expert_idx]
            if y_int8_i.shape[0] == 0:
                outputs.append(torch.empty(0, self.hidden_size, device=sorted_hidden_states.device, dtype=sorted_hidden_states.dtype))
                continue

            fc2_out = self._quant_linear(
                y_int8_i,
                y_scale_i,
                self.down_proj_weight[expert_idx],
                self.down_proj_weight_scale[expert_idx],
                sorted_hidden_states.dtype,
            )
            outputs.append(fc2_out)

        return torch.cat(outputs, dim=0)

    def extra_repr(self) -> str:
        return f"{self.num_experts=}, {self.intermediate_size=}, {self.hidden_size=}, {self.quant_dtype=}, {self.quant_group_size=}, {self.weight_bits=}".replace("self.", "")


class MojoMoECombine(MojoOperator):
    def __init__(
        self,
        multiply_by_gates: bool = True,
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Combine operator.

        Init parameters:
        - multiply_by_gates (bool): Whether to multiply the expert output by the gating weights.

        Scope: Only covers common semantics, does not involve backend communication or core partitioning details.
        """
        super().__init__(**kwargs)
        self.multiply_by_gates = multiply_by_gates

    def forward(
        self,
        output_buffer: torch.Tensor,
        expert_outputs: torch.Tensor,
        sorted_gates: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for MoE Combine operator.

        Input:
        - output_buffer (torch.Tensor): Initial tensor to combine results into.
        - expert_outputs (torch.Tensor): Output from experts.
        - sorted_gates (torch.Tensor): Packed gating weights.
        - token_indices (torch.Tensor): Indices for packing/unpacking.

        Output:
        - combined: Combined output tensor.
        """
        token_indices = token_indices.to(torch.int64)  # scatter_reduce requires int64 indices
        combined_expert_outputs = expert_outputs.float()
        if self.multiply_by_gates:
            combined_expert_outputs = combined_expert_outputs * sorted_gates.float()

        scatter_indices = token_indices.unsqueeze(-1).expand(-1, output_buffer.size(1))
        output_buffer = output_buffer.float()
        combined = output_buffer.scatter_reduce(
            0, scatter_indices, combined_expert_outputs, reduce="sum", include_self=True
        )
        return combined.to(expert_outputs.dtype)
