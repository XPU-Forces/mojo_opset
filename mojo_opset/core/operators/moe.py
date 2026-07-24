from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..operator import MojoOperator
from .quantize import MojoMoEDynamicQuant


class MojoMoE(MojoOperator):
    _use_fused_moe = False

    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size=None,
        activation: str = "swiglu",
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group=None,
        dp_input: bool = False,
        **kwargs,
    ):
        super().__init__()
        if activation != "swiglu":
            raise NotImplementedError(f"MojoMoe: Activation {activation} is not supported.")

        # NOTE: in some cases, branches may have different expert num or topk
        self.num_experts = num_experts
        if intermediate_size is None:
            raise ValueError("MojoMoE: intermediate_size must be provided.")

        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Expert parallelism slice: gating still uses the global expert set, but experts only hold the local slice.
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        base = num_experts // ep_size
        rem = num_experts % ep_size
        self.num_experts_local = base + 1 if ep_rank < rem else base
        self.ep_start = base * ep_rank + min(ep_rank, rem)
        self.ep_end = self.ep_start + self.num_experts_local

        self.dp_input = dp_input

        if not self._use_fused_moe:
            self.gating = MojoMoEGating._registry.get(self._backend)(
                hidden_size=self.hidden_size, num_experts=self.num_experts, top_k=self.top_k, **kwargs
            )
            self.dispatch = MojoMoEDispatch._registry.get(self._backend)(num_experts=self.num_experts, **kwargs)
            self.experts = MojoExperts._registry.get(self._backend)(
                num_experts=self.num_experts_local,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                activation=activation,
                **kwargs,
            )
            self.combine = MojoMoECombine._registry.get(self._backend)(multiply_by_gates=True, **kwargs)
        else:
            # Note: Use Torch MojoOp as a parameter holder, do not use its forward
            self.gating = MojoMoEGating._registry.get("torch")(
                hidden_size=self.hidden_size, num_experts=self.num_experts, top_k=self.top_k, **kwargs
            )
            self.experts = MojoExperts._registry.get("torch")(
                num_experts=self.num_experts_local,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                activation=activation,
                **kwargs,
            )

    def forward(self, hidden_states):
        # hidden_states: [num_tokens, H]
        # DP input: gather peer ranks' shards so gating/dispatch see the full token set.
        if self.dp_input and self.ep_size > 1:
            local_tokens = hidden_states.shape[0]
            full = torch.empty(
                local_tokens * self.ep_size, *hidden_states.shape[1:],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            dist.all_gather_into_tensor(full, hidden_states.contiguous(), group=self.ep_group)
            hidden_states = full

        top_k_indices, top_k_gates = self.gating(hidden_states)
        # top_k_indices, top_k_gates: [num_tokens, top_k]
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states, top_k_gates, top_k_indices
        )
        # sorted_hidden_states: [local_tokens, H]
        # tokens_per_expert: [num_experts]
        # sorted_gates: [local_tokens, 1]
        # token_indices: [local_tokens]

        # EP slice: keep only the token range routed to this rank's local experts.
        if self.ep_size > 1:
            cumsum = tokens_per_expert.cumsum(0)
            tok_start = 0 if self.ep_start == 0 else cumsum[self.ep_start - 1].item()
            tok_end = cumsum[self.ep_end - 1].item()
            sorted_hidden_states = sorted_hidden_states[tok_start:tok_end]
            tokens_per_expert = tokens_per_expert[self.ep_start:self.ep_end]
            sorted_gates = sorted_gates[tok_start:tok_end]
            token_indices = token_indices[tok_start:tok_end]

        expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        # expert_outputs: [local_tokens, H]
        output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        # combined: [num_tokens, H]

        # Sum partial expert outputs across EP ranks; reduce_scatter slices the result back to the rank's DP shard.
        if self.ep_size > 1:
            if self.dp_input:
                local_combined = torch.empty(
                    combined.shape[0] // self.ep_size, *combined.shape[1:],
                    dtype=combined.dtype, device=combined.device,
                )
                dist.reduce_scatter_tensor(local_combined, combined.contiguous(), op=dist.ReduceOp.SUM, group=self.ep_group)
                combined = local_combined
            else:
                dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=self.ep_group)

        return combined


class MojoQuantMoE(MojoOperator):
    _use_fused_moe = False

    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size=None,
        activation: str = "swiglu",
        quant_dtype: torch.dtype = torch.int8,
        up_quant_group_size: int = -1,
        up_weight_dtype: Union[torch.dtype, str] = torch.int8,
        down_quant_group_size: int = -1,
        down_weight_dtype: Union[torch.dtype, str] = torch.int8,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group=None,
        dp_input: bool = False,
        **kwargs,
    ):
        super().__init__()
        if activation != "swiglu":
            raise NotImplementedError(f"MojoQuantMoE: Activation {activation} is not supported.")
        if quant_dtype != torch.int8:
            raise NotImplementedError(f"MojoQuantMoE: quant_dtype must be 'int8', got {quant_dtype}.")
        if up_weight_dtype not in ("int4", torch.int8) or down_weight_dtype not in ("int4", torch.int8):
            raise ValueError("MojoQuantMoE: weight must be w4 or w8")

        if intermediate_size is None:
            raise ValueError("MojoQuantMoE: intermediate_size must be provided.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quant_dtype = quant_dtype
        self.up_quant_group_size = up_quant_group_size
        self.up_weight_dtype = up_weight_dtype
        self.down_quant_group_size = down_quant_group_size
        self.down_weight_dtype = down_weight_dtype

        # Expert parallelism slice: gating still uses the global expert set, but experts only hold the local slice.
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        base = num_experts // ep_size
        rem = num_experts % ep_size
        self.num_experts_local = base + 1 if ep_rank < rem else base
        self.ep_start = base * ep_rank + min(ep_rank, rem)
        self.ep_end = self.ep_start + self.num_experts_local

        # DP input mode: each rank holds 1/ep_size of the tokens (caller guarantees equal counts per rank).
        self.dp_input = dp_input

        if not self._use_fused_moe:
            self.gating = MojoMoEGating._registry.get(self._backend)(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                top_k=self.top_k,
                **kwargs,
            )
            self.dispatch = MojoMoEDispatch._registry.get(self._backend)(num_experts=self.num_experts, **kwargs)
            self.experts = MojoQuantExperts._registry.get(self._backend)(
                num_experts=self.num_experts_local,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                activation=activation,
                quant_dtype=quant_dtype,
                up_quant_group_size=up_quant_group_size,
                up_weight_dtype=up_weight_dtype,
                down_quant_group_size=down_quant_group_size,
                down_weight_dtype=down_weight_dtype,
                **kwargs,
            )
            self.combine = MojoMoECombine._registry.get(self._backend)(multiply_by_gates=True, **kwargs)
        else:
            # Note: Use Torch MojoOp as a parameter holder, do not use its forward
            # FIXME: would there be a better display? Be like: no dummy holders displayed as submodules
            self.gating = MojoMoEGating._registry.get("torch")(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                top_k=self.top_k,
                **kwargs,
            )
            self.experts = MojoQuantExperts._registry.get("torch")(
                num_experts=self.num_experts_local,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                activation=activation,
                quant_dtype=quant_dtype,
                up_quant_group_size=up_quant_group_size,
                up_weight_dtype=up_weight_dtype,
                down_quant_group_size=down_quant_group_size,
                down_weight_dtype=down_weight_dtype,
                **kwargs,
            )

    def forward(self, hidden_states):
        # DP input: gather peer ranks' shards so gating/dispatch see the full token set.
        if self.dp_input and self.ep_size > 1:
            local_tokens = hidden_states.shape[0]
            full = torch.empty(
                local_tokens * self.ep_size, *hidden_states.shape[1:],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            dist.all_gather_into_tensor(full, hidden_states.contiguous(), group=self.ep_group)
            hidden_states = full

        top_k_indices, top_k_gates = self.gating(hidden_states)
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states,
            top_k_gates,
            top_k_indices,
        )

        # EP slice: keep only the token range routed to this rank's local experts.
        if self.ep_size > 1:
            cumsum = tokens_per_expert.cumsum(0)
            tok_start = 0 if self.ep_start == 0 else cumsum[self.ep_start - 1].item()
            tok_end = cumsum[self.ep_end - 1].item()
            sorted_hidden_states = sorted_hidden_states[tok_start:tok_end]
            tokens_per_expert = tokens_per_expert[self.ep_start:self.ep_end]
            sorted_gates = sorted_gates[tok_start:tok_end]
            token_indices = token_indices[tok_start:tok_end]

        expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)

        # Sum partial expert outputs across EP ranks; reduce_scatter slices the result back to the rank's DP shard.
        if self.ep_size > 1:
            if self.dp_input:
                local_combined = torch.empty(
                    combined.shape[0] // self.ep_size, *combined.shape[1:],
                    dtype=combined.dtype, device=combined.device,
                )
                dist.reduce_scatter_tensor(local_combined, combined.contiguous(), op=dist.ReduceOp.SUM, group=self.ep_group)
                combined = local_combined
            else:
                dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=self.ep_group)

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
        return top_k_indices.to(torch.int32), top_k_gates

    def extra_repr(self) -> str:
        hidden_size = self.gate_weight.size(0)
        num_experts = self.gate_weight.size(1)
        return f"{hidden_size=}, {num_experts=}, {self.top_k=}".replace("self.", "")


class MojoMoEGatingTopKHash(MojoOperator):
    """Hash-based expert selection used by DeepSeek-V4 routing."""

    def __init__(
        self,
        k: int,
        *,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        norm_type: int = 1,
        out_flag: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.routed_scaling_factor = routed_scaling_factor
        self.eps = eps
        self.norm_type = norm_type
        self.out_flag = out_flag

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor = None,
        tid2eid: torch.Tensor = None,
    ):
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")

        logits = x.float()
        if self.norm_type == 0:
            scores = torch.softmax(logits, dim=-1)
        elif self.norm_type == 1:
            scores = torch.sigmoid(logits)
        elif self.norm_type == 2:
            scores = torch.sqrt(F.softplus(logits))
        else:
            raise ValueError(f"norm_type must be 0, 1, or 2, got {self.norm_type}")

        if input_ids is not None and tid2eid is not None:
            expert_idx = tid2eid[input_ids].to(torch.int32)
        else:
            _, expert_idx = torch.topk(scores, self.k, dim=-1)
            expert_idx = expert_idx.to(torch.int32)

        selected = torch.gather(scores, 1, expert_idx.long())
        if self.norm_type != 0:
            selected = selected / (selected.sum(dim=-1, keepdim=True) + self.eps)

        y = (selected * self.routed_scaling_factor).to(x.dtype)
        return y, expert_idx, scores if self.out_flag else None

    def extra_repr(self) -> str:
        return (
            f"{self.k=}, {self.routed_scaling_factor=}, {self.eps=}, "
            f"{self.norm_type=}, {self.out_flag=}"
        ).replace("self.", "")


def _count_expert_tokens(top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    flat_indices = top_k_indices.reshape(-1).to(dtype=torch.int64, device=top_k_indices.device)
    return torch.bincount(flat_indices, minlength=num_experts).to(dtype=torch.int32, device=top_k_indices.device)

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
        - top_k_gates (torch.Tensor): Top-k gating weights, must be float32.
        - top_k_indices (torch.Tensor): Top-k expert indices, must be int32.

        Output:
        - sorted_hidden_states: Sorted inputs for experts.
        - tokens_per_expert: Count of tokens for each expert.
        - sorted_gates: Packed gating weights, float32.
        - token_indices: Indices for packing/unpacking, int32.

        Note: ordering of tokens *within* one expert's bucket is not part of the
        contract — backends (e.g. ixformer's fused kernel) are free to permute
        them. Downstream consumers must treat each bucket as an unordered set
        and never rely on `token_indices[start:end]` being sorted by token id
        or by top-k slot. Tests should verify the bucket as a set, not a
        sequence.
        """
        assert top_k_gates.dtype == torch.float32, (
            f"MojoMoEDispatch: top_k_gates must be float32, got {top_k_gates.dtype}."
        )
        assert top_k_indices.dtype == torch.int32, (
            f"MojoMoEDispatch: top_k_indices must be int32, got {top_k_indices.dtype}."
        )
        batch_token_indices = (
            torch.arange(0, hidden_states.shape[0], device=hidden_states.device, dtype=top_k_indices.dtype)
            .unsqueeze(1)
            .repeat(1, top_k_indices.shape[-1])
            .flatten()
        )
        # batch_token_indices: [BS * top_k]
        flat_top_k_gates = top_k_gates.reshape(-1, 1)
        flat_top_k_indices = top_k_indices.flatten()
        # Default torch.sort is non-stable; bucket-internal ordering is
        # intentionally undefined here so backends can pick whatever order
        # their fused kernel produces.
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
        up_quant_group_size: int = -1,
        up_weight_dtype: Union[torch.dtype, str] = torch.int8,
        down_quant_group_size: int = -1,
        down_weight_dtype: Union[torch.dtype, str] = torch.int8,
        **kwargs,
    ):
        """
        Quantized MoE Experts reference.

        The input activation is expected to be dynamically quantized before this
        operator. For ``weight_dtype=4``, expert weights are signed int4
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
        if up_weight_dtype not in ("int4", torch.int8) or down_weight_dtype not in ("int4", torch.int8):
            raise NotImplementedError("MojoQuantExperts currently only supports w4 or w8.")
        if (up_weight_dtype == "int4" and (hidden_size % 2 != 0 or intermediate_size % 2 != 0)) or (down_weight_dtype == "int4" and (intermediate_size % 2 != 0 or hidden_size % 2 != 0)):
            raise ValueError("MojoQuantExperts requires even hidden_size and intermediate_size for int4 packing.")
        
        self.activation = activation
        self.quant_dtype = quant_dtype
        self.up_quant_group_size = up_quant_group_size
        self.up_weight_dtype = up_weight_dtype
        self.down_quant_group_size = down_quant_group_size
        self.down_weight_dtype = down_weight_dtype
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

        if up_weight_dtype == torch.int8:
            self.register_buffer(
                "up_proj_weight",
                torch.empty((num_experts, intermediate_size * 2, hidden_size), dtype=torch.int8),
            )
        else:
            assert up_weight_dtype == "int4"
            self.register_buffer(
                "up_proj_weight",
                torch.empty((num_experts, intermediate_size * 2 // 2, hidden_size), dtype=torch.int8),
            )

        if down_weight_dtype == torch.int8:
            self.register_buffer(
                "down_proj_weight",
                torch.empty((num_experts, hidden_size, intermediate_size), dtype=torch.int8),
            )
        else:
            assert down_weight_dtype == "int4"
            self.register_buffer(
                "down_proj_weight",
                torch.empty((num_experts, hidden_size // 2, intermediate_size), dtype=torch.int8),
            )

        if up_quant_group_size > 0:
            up_proj_groups = (hidden_size + up_quant_group_size - 1) // up_quant_group_size
            self.up_proj_weight_scale = nn.Parameter(
                torch.empty(
                    (num_experts, intermediate_size * 2, up_proj_groups),
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

        if down_quant_group_size > 0:
            down_proj_groups = (intermediate_size + down_quant_group_size - 1) // down_quant_group_size
            self.down_proj_weight_scale = nn.Parameter(
                torch.empty(
                    (num_experts, hidden_size, down_proj_groups),
                    dtype=torch.bfloat16,
                ),
            )
        else:
            self.down_proj_weight_scale = nn.Parameter(
                torch.empty(
                    (num_experts, hidden_size), 
                    dtype=torch.bfloat16,
                ),
            )

    @staticmethod
    def _unpack_weight(weight):
        assert weight.ndim == 2
        unpacked_weight = torch.empty(weight.shape[0] * 2, weight.shape[1], device=weight.device, dtype=torch.int8)
        unpacked_weight[::2] = weight & 0x0F
        unpacked_weight[1::2] = (weight >> 4) & 0x0F
        unpacked_weight = torch.where(unpacked_weight >= 8, unpacked_weight - 16, unpacked_weight)
        return unpacked_weight

    @staticmethod
    def _quant_linear(
        input_int8: torch.Tensor, 
        input_scale: torch.Tensor,
        expert_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        weight_dtype: Union[torch.dtype, str] = torch.int8,
        quant_group_size: int = -1,
    ) -> torch.Tensor:
        if weight_dtype == "int4":
            expert_weight = MojoQuantExperts._unpack_weight(expert_weight)
        
        assert input_scale.ndim == 2 and input_scale.shape[1] == 1
        if quant_group_size > 0:
            x_int8_groups = torch.split(input_int8, quant_group_size, dim=-1)
            weight_int8_groups = torch.split(expert_weight, quant_group_size, dim=-1)
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
                self.up_weight_dtype,
                self.up_quant_group_size,
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
                self.down_weight_dtype,
                self.down_quant_group_size,
            )
            outputs.append(fc2_out)

        return torch.cat(outputs, dim=0)

    def extra_repr(self) -> str:
        return f"{self.num_experts=}, {self.intermediate_size=}, {self.hidden_size=}, {self.quant_dtype=}, {self.up_quant_group_size=}, {self.up_weight_dtype=}, {self.down_quant_group_size=}, {self.down_weight_dtype=}".replace("self.", "")


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
        output_buffer = torch.zeros_like(output_buffer, dtype=torch.float32)
        combined = output_buffer.scatter_reduce(
            0, scatter_indices, combined_expert_outputs, reduce="sum", include_self=True
        )
        return combined.to(expert_outputs.dtype)
