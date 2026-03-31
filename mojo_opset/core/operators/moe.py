from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoMoE(MojoOperator):
    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size=None,
        ffn_intermediate_size=None,
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
            intermediate_size = ffn_intermediate_size
        if intermediate_size is None:
            raise ValueError("MojoMoE: intermediate_size must be provided.")

        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gating = MojoMoEGating(hidden_size=self.hidden_size, num_experts=self.num_experts, top_k=self.top_k, **kwargs)
        self.dispatch = MojoMoEDispatch(num_experts=self.num_experts, **kwargs)
        self.experts = MojoExperts(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=activation,
            **kwargs,
        )
        self.combine = MojoMoECombine(multiply_by_gates=True, **kwargs)

        if self.gating.gate_weight is not None:
            setattr(self.gating.gate_weight, "force_dtype", torch.float32)

    def forward(self, hidden_states):
        # hidden_states: [BS, H]
        top_k_indices, top_k_gates = self.gating(hidden_states)
        # top_k_indices, top_k_gates: [BS, top_k]
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(hidden_states, top_k_gates, top_k_indices)
        # sorted_hidden_states: [total_toks, H]
        # tokens_per_expert: [num_experts]
        # sorted_gates: [total_toks, 1]
        # token_indices: [total_toks]
        expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        # expert_outputs: [total_toks, H]
        output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        # combined: [BS, H]
        return combined


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

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for MoE Gating operator.

        Input:
        - hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Output:
        - torch.Tensor: Output tensor of shape [batch_size, seq_len, num_experts].
        """
        assert self.gate_weight.dtype == torch.float32
        gate_logits = torch.matmul(hidden_states.float(), self.gate_weight)
        gate_logits = torch.softmax(gate_logits, dim=-1)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        expert_weights = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True)
        return top_k_indices, expert_weights

    def extra_repr(self) -> str:
        hidden_size = self.gate_weight.size(0)
        num_experts = self.gate_weight.size(1)
        return f"{hidden_size=}, {num_experts=}, {self.top_k=}".replace("self.", "")


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

        self.up_proj_weight = nn.Parameter(torch.empty(num_experts, intermediate_size * 2, hidden_size, **self.tensor_factory_kwargs))
        self.down_proj_weight = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size, **self.tensor_factory_kwargs))

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
        dtype = expert_outputs.dtype
        combined_expert_outputs = expert_outputs.float()
        if self.multiply_by_gates:
            combined_expert_outputs = combined_expert_outputs * sorted_gates.float()

        scatter_indices = token_indices.unsqueeze(-1).expand(-1, output_buffer.size(1))
        output_buffer = output_buffer.float()
        combined = output_buffer.scatter_reduce(0, scatter_indices, combined_expert_outputs, reduce="sum", include_self=True)
        return combined.to(expert_outputs.dtype)


def _validate_moe_token_count(token_count: torch.Tensor, route_count: int) -> torch.Tensor:
    token_count_i64 = token_count.to(dtype=torch.int64, device=token_count.device)
    if token_count_i64.dim() != 1:
        raise ValueError(f"token_count must be 1D, but got shape {tuple(token_count.shape)}")
    if token_count_i64.sum().item() != route_count:
        raise ValueError(
            f"token_count sum must equal total routed token count {route_count}, "
            f"but got {token_count_i64.sum().item()}."
        )
    return token_count_i64


def _count_expert_tokens(top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    flat_indices = top_k_indices.reshape(-1).to(dtype=torch.int64, device=top_k_indices.device)
    return torch.bincount(flat_indices, minlength=num_experts).to(dtype=torch.int32, device=top_k_indices.device)


def _expand_grouped_route_param(
    param: Optional[torch.Tensor],
    token_count: torch.Tensor,
    route_shape: Sequence[int],
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


def _dequant_grouped_input(
    input: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    quant_block_size: int,
) -> torch.Tensor:
    input_fp = input.float()
    if input_scale is None:
        return input_fp

    scale_fp = input_scale.float()

    if scale_fp.shape == input_fp.shape:
        return input_fp * scale_fp

    if scale_fp.shape == input_fp.shape[:-1]:
        return input_fp * scale_fp.unsqueeze(-1)

    if scale_fp.dim() == input_fp.dim() and scale_fp.shape[-1] == 1 and scale_fp.shape[:-1] == input_fp.shape[:-1]:
        return input_fp * scale_fp

    if (
        input_fp.shape[-1] % quant_block_size == 0
        and scale_fp.shape[:-1] == input_fp.shape[:-1]
        and scale_fp.shape[-1] == input_fp.shape[-1] // quant_block_size
    ):
        input_blocks = input_fp.reshape(*input_fp.shape[:-1], -1, quant_block_size)
        return (input_blocks * scale_fp.unsqueeze(-1)).reshape_as(input_fp)

    raise ValueError(
        "input_scale shape must match input, input without the last dim, or per-block grouped scale. "
        f"Got input shape {tuple(input.shape)} and input_scale shape {tuple(input_scale.shape)}."
    )


def _select_group_param(
    param: torch.Tensor,
    expert_idx: int,
) -> torch.Tensor:
    if param.dim() == 1:
        return param.float()
    if param.dim() == 2:
        return param[expert_idx].float()
    raise ValueError(f"Grouped parameter must be 1D or 2D, but got shape {tuple(param.shape)}")


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
    ):
        super().__init__()
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
    ):
        super().__init__()
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
        if quant_mode not in (0, 1):
            raise NotImplementedError(f"Unsupported quant_mode: {quant_mode}, expected 0 or 1.")

        route_shape = input.shape[:2]
        _validate_moe_token_count(token_count, route_shape[0] * route_shape[1])

        left, right = input.float().chunk(2, dim=-1)
        if beta == 0:
            raise ValueError("beta must be non-zero.")
        output = (F.silu(left * beta) / beta) * right

        expanded_scale = _expand_grouped_route_param(smooth_scale, token_count, route_shape)
        if expanded_scale is not None:
            output = output * expanded_scale

        scale = output.abs().amax(dim=-1).clamp(min=1e-12) / 127
        quantized = torch.clamp(torch.round(output / scale.unsqueeze(-1)), -128, 127)
        return quantized.to(self.quant_dtype), scale


class MojoGroupQuantGemmMoE(MojoOperator):
    def __init__(
        self,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = True,
        quant_block_size: int = 8,
        quant_algo: str = "none",
        top_k: Optional[int] = None,
        use_splitk: bool = False,
    ):
        super().__init__()
        self.output_dtype = output_dtype
        self.trans_weight = trans_weight
        self.quant_block_size = quant_block_size
        self.quant_algo = quant_algo
        self.top_k = top_k
        self.use_splitk = use_splitk

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        token_count: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        if input.dim() != 3:
            raise ValueError(f"input must be 3D, but got shape {tuple(input.shape)}")
        if weight.dim() != 3:
            raise ValueError(f"weight must be 3D, but got shape {tuple(weight.shape)}")

        batch_size, top_k, hidden_dim = input.shape
        route_count = batch_size * top_k
        token_count_i64 = _validate_moe_token_count(token_count, route_count)
        # TODO(liuyuan): Refactor this implementation to first perform integer tiled matrix computation, then execute dequantization using the quantization parameters (scale and bias) of the weight, and finally run dequantization with the quantization parameters of the input.
        input_fp = _dequant_grouped_input(input, input_scale, self.quant_block_size).reshape(route_count, hidden_dim)

        outputs = []
        route_start = 0
        for expert_idx, expert_token_count in enumerate(token_count_i64.tolist()):
            expert_input = input_fp[route_start : route_start + expert_token_count]
            expert_weight = weight[expert_idx].float()
            if self.trans_weight:
                expert_weight = expert_weight.transpose(0, 1).contiguous()

            expert_output = expert_input @ expert_weight
            expert_output = expert_output * _select_group_param(weight_scale, expert_idx).unsqueeze(0)
            if bias is not None:
                expert_output = expert_output + _select_group_param(bias, expert_idx).unsqueeze(0)
            outputs.append(expert_output)
            route_start += expert_token_count

        return torch.cat(outputs, dim=0).reshape(batch_size, top_k, -1).to(self.output_dtype)


class MojoGroupQuantGemmCombineMoE(MojoOperator):
    def __init__(
        self,
        output_dtype: torch.dtype = torch.bfloat16,
        trans_weight: bool = False,
        quant_block_size: int = 8,
        shared_expert_rank_num: float = 1.0,
        num_experts_per_rank: Optional[int] = None,
        normalize_top_k_gates: bool = False,
        top_k: Optional[int] = None,
        ep_rank: int = 0,
    ):
        super().__init__()
        self.output_dtype = output_dtype
        self.trans_weight = trans_weight
        self.quant_block_size = quant_block_size
        self.shared_expert_rank_num = shared_expert_rank_num
        self.num_experts_per_rank = num_experts_per_rank
        self.normalize_top_k_gates = normalize_top_k_gates
        self.top_k = top_k
        self.ep_rank = ep_rank

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        top_k_gates: torch.Tensor,
        token_indices: torch.Tensor,
        token_count: torch.Tensor,
        shared_output: Optional[torch.Tensor],
        weight_scale: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        routed = MojoGroupQuantGemmMoE._registry.get("torch")(
            output_dtype=torch.float32,
            trans_weight=self.trans_weight,
            quant_block_size=self.quant_block_size,
        )(
            input=input,
            weight=weight,
            token_count=token_count,
            weight_scale=weight_scale,
            input_scale=input_scale,
            bias=bias,
        )

        gates = top_k_gates.float()
        if gates.dim() == 3 and gates.size(-1) == 1:
            gates = gates.squeeze(-1)
        if gates.dim() != 2 or gates.shape != routed.shape[:2]:
            raise ValueError(
                f"top_k_gates must match routed output leading dims {tuple(routed.shape[:2])}, "
                f"but got shape {tuple(top_k_gates.shape)}."
            )

        if self.normalize_top_k_gates:
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        gather_indices = token_indices
        if gather_indices.dim() == 3 and gather_indices.size(-1) == 1:
            gather_indices = gather_indices.squeeze(-1)
        if gather_indices.dim() != 2 or gather_indices.shape != routed.shape[:2]:
            raise ValueError(
                f"token_indices must match routed output leading dims {tuple(routed.shape[:2])}, "
                f"but got shape {tuple(token_indices.shape)}."
            )

        output_dim = routed.shape[-1]
        if shared_output is None:
            output_buffer = torch.zeros(
                routed.shape[0],
                output_dim,
                dtype=torch.float32,
                device=routed.device,
            )
        else:
            output_buffer = shared_output.float().clone()

        routed_flat = routed.reshape(-1, output_dim)
        gates_flat = gates.reshape(-1, 1)
        indices_flat = gather_indices.reshape(-1).to(dtype=torch.long)
        output_buffer.index_add_(0, indices_flat, routed_flat * gates_flat)
        return output_buffer.to(self.output_dtype if shared_output is None else shared_output.dtype)


class MojoGroupQuantGemmA8W4MSD(MojoOperator):
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_msd_bias: torch.Tensor,
        token_count: torch.Tensor,
        weight_deqscale: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError("Torch reference for MSD-packed W4A8 group quant GEMM is not available.")


class MojoGroupQuantGemmCombineA8W4MSD(MojoOperator):
    def __init__(
        self,
        top_k: int,
        shared_expert_rank_num: float = 1.0,
        ep_rank: int = 0,
    ):
        super().__init__()
        self.top_k = top_k
        self.shared_expert_rank_num = shared_expert_rank_num
        self.ep_rank = ep_rank

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_msd_bias: torch.Tensor,
        token_count: torch.Tensor,
        weight_deqscale: torch.Tensor,
        input_scale: torch.Tensor,
        top_k_gates: torch.Tensor,
        token_indices: torch.Tensor,
        shared_output: torch.Tensor,
    ):
        raise NotImplementedError("Torch reference for MSD-packed W4A8 group quant GEMM combine is not available.")


class MojoGroupedMatmulA8W4MSD(MojoOperator):
    def __init__(
        self,
        transpose_a: bool = False,
        transpose_b: bool = False,
        group_type: int = 0,
        split_item: int = 1,
        expert_ids: Optional[Sequence[int]] = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.group_type = group_type
        self.split_item = split_item
        self.expert_ids = [0] if expert_ids is None else list(expert_ids)
        self.output_dtype = output_dtype

    def forward(
        self,
        inputs,
        weights,
        weight_msd_biases,
        weight_deqscales,
        token_count: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError("Torch reference for MSD-packed grouped_matmul_a8w4 is not available.")
