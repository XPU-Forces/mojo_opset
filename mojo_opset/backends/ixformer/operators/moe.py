from typing import Optional

import torch

from mojo_opset.core import MojoMoEGating
from mojo_opset.core import MojoMoEInitRoutingDynamicQuant
from mojo_opset.core import MojoFusedSwiGLUMoEScaleDynamicQuantize
from mojo_opset.core import MojoGroupQuantGemmMoE
from mojo_opset.core import MojoGroupQuantGemmCombineMoE

from ..utils import _get_ixf_and_check_device

class IxformerMoEGating(MojoMoEGating):
    supported_platforms_list = ["ilu"]

    def forward(self, hidden_states: torch.Tensor):
        ixf_f = _get_ixf_and_check_device(hidden_states, self.__class__.__name__)
        assert self.gate_weight.dtype == torch.float32
        gate_logits = ixf_f.mixed_type_linear(hidden_states, self.gate_weight.T.contiguous())
        top_k_gates, top_k_indices = ixf_f.moe_topk_softmax(gate_logits, self.top_k, renormalize=True)

        return top_k_indices.to(torch.int64), top_k_gates

class IxformerMoEInitRoutingDynamicQuant(MojoMoEInitRoutingDynamicQuant):
    supported_platforms_list = ["ilu"]

    def forward(self,
                hidden_states: torch.Tensor,
                top_k_gates: torch.Tensor,
                top_k_indices: torch.Tensor,
                smooth_scale: Optional[torch.Tensor] = None,
                quant_mode: int = 0):
        ixf_f = _get_ixf_and_check_device(hidden_states, self.__class__.__name__)
        top_k_indices = top_k_indices.to(torch.int32).contiguous()

        bs, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.view(bs * seq_len, -1)
        num_tokens, top_k = top_k_indices.shape
        (src_to_dst, 
         sorted_token_ids,
         expert_sizes_gpu, 
         expert_sizes_cpu,
         expand_tokens) = ixf_f.moe_compute_token_index_ep(top_k_indices, 
                                                           self.num_experts,
                                                           self.start_expert_id,
                                                           self.end_expert_id,
                                                           )
        if sorted_token_ids.numel() == 0:
            return hidden_states, None, None, None, None

        i8_hidden_states, quant_scale = ixf_f.moe_expand_input_dynamic_scaled_int8(
                                               hidden_states=hidden_states,
                                               dst_to_src=sorted_token_ids,
                                               dst_tokens=expand_tokens, 
                                               topk=top_k,
                                               src_to_dst=src_to_dst,
                                               topk_ids=top_k_indices,
                                               smooth_scales=smooth_scale)
        return i8_hidden_states.view(-1, dim), top_k_gates, sorted_token_ids, src_to_dst, expert_sizes_cpu, quant_scale
class IxformerFusedSwiGLUMoEScaleDynamicQuantize(MojoFusedSwiGLUMoEScaleDynamicQuantize):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        input: torch.Tensor,
        smooth_scale: Optional[torch.Tensor],
        sorted_token_ids: torch.Tensor,
        topk_indices: torch.Tensor,
        fc1_intermediate_size: int,
        beta: float = 1.0,
        quant_mode: int = 0,
    ):
        ixf_f = _get_ixf_and_check_device(input, self.__class__.__name__)
        quantized_output, quant_scale = ixf_f.activation_dynamic_scaled_int8(
                input=input.view(-1, fc1_intermediate_size),
                smooth_scales=smooth_scale,
                dst_to_src=sorted_token_ids,
                topk_ids=topk_indices,
                act_type="swiglu",)
        return quantized_output, quant_scale

class IxformerGroupQuantGemmMoE(MojoGroupQuantGemmMoE):
    supported_platforms_list = ["ilu"]

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                token_count: torch.Tensor,
                weight_scale: torch.Tensor,
                input_scale: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None):
        ixf_f = _get_ixf_and_check_device(input, self.__class__.__name__)
        quant_gemm_output = ixf_f.moe_w8a8_group_gemm(
                                    input=input,
                                    weight=weight,
                                    i_scales=input_scale,
                                    w_scales=weight_scale,
                                    output_dtype=self.output_dtype,
                                    tokens_per_experts=token_count,
                                    dst_to_src=None,
                                    bias=bias,
                                    format="TN",)
        return quant_gemm_output

class IxformerGroupQuantGemmCombineMoE(MojoGroupQuantGemmCombineMoE):
    supported_platforms_list = ["ilu"]

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                top_k_gates: torch.Tensor,
                token_indices: torch.Tensor,
                src_to_dst: torch.Tensor,
                token_count: torch.Tensor,
                shared_output: Optional[torch.Tensor],
                weight_scale: torch.Tensor,
                input_scale: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,):
        ixf_f = _get_ixf_and_check_device(input, self.__class__.__name__)
        num_tokens, top_k = top_k_gates.shape
        dim = weight.shape[1]
        quant_gemm_output = torch.empty(
            (num_tokens * self.top_k, dim),
            device=input.device,
            dtype=torch.bfloat16,
        )
        ixf_f.moe_w8a8_group_gemm(
                    input=input,
                    weight=weight,
                    i_scales=input_scale,
                    w_scales=weight_scale,
                    output_dtype=torch.bfloat16,
                    tokens_per_experts=token_count,
                    dst_to_src=token_indices,
                    bias=bias,
                    format="TN",
                    output=quant_gemm_output)
        
        reduce_mask = src_to_dst == -1
        combined_output = ixf_f.moe_output_reduce_sum(
            input=quant_gemm_output.view(-1, self.top_k, dim),
            topk_weight=top_k_gates,
            mask=reduce_mask,
            extra_residual=shared_output
        )
        return combined_output


class IxformerQuantMoe(torch.nn.Module):
    """Composable quantized MoE using ixformer backend operators."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
        quant_block_size: int = 8,
        normalize_top_k_gates: bool = False,
        routed_scaling_factor: float = 1.0,
        start_expert_id: int = 0,
        end_expert_id: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dtype = output_dtype
        self.routed_scaling_factor = routed_scaling_factor

        self.gating = IxformerMoEGating(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.init_routing = IxformerMoEInitRoutingDynamicQuant(
            num_experts=num_experts,
            top_k=top_k,
            quant_block_size=quant_block_size,
            start_expert_id=start_expert_id,
            end_expert_id=end_expert_id,
        )
        self.fc1 = IxformerGroupQuantGemmMoE(
            output_dtype=output_dtype,
            trans_weight=False,
            quant_block_size=quant_block_size,
            top_k=top_k,
        )
        self.act_quant = IxformerFusedSwiGLUMoEScaleDynamicQuantize()
        self.fc2 = IxformerGroupQuantGemmCombineMoE(
            output_dtype=output_dtype,
            trans_weight=False,
            quant_block_size=quant_block_size,
            normalize_top_k_gates=normalize_top_k_gates,
            top_k=top_k,
        )

        # W8A8 MoE parameter layout aligned with TN format:
        # w13: [num_experts, 2 * intermediate_size, hidden_size]
        # w2:  [num_experts, hidden_size, intermediate_size]
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.w13_input_scale: Optional[torch.Tensor] = None
        self.w2_input_scale: Optional[torch.Tensor] = None
        self.w13_bias: Optional[torch.Tensor] = None
        self.w2_bias: Optional[torch.Tensor] = None

        if self.gating.gate_weight is not None:
            setattr(self.gating.gate_weight, "force_dtype", torch.float32)

    @staticmethod
    def _pad_last_dim(x: torch.Tensor, target_last_dim: int) -> torch.Tensor:
        if x.shape[-1] == target_last_dim:
            return x
        if x.shape[-1] > target_last_dim:
            raise ValueError(
                f"Input last dim {x.shape[-1]} exceeds target dim {target_last_dim}."
            )
        return torch.nn.functional.pad(x, (0, target_last_dim - x.shape[-1]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: Optional[torch.Tensor] = None,
        top_k_gates: Optional[torch.Tensor] = None,
        shared_output: Optional[torch.Tensor] = None,
        *,
        quant_mode: int = 0,
        beta: float = 1.0,
    ) -> torch.Tensor:
        if hidden_states.dim() not in (2, 3):
            raise ValueError(
                f"hidden_states must be 2D or 3D, but got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dim must be {self.hidden_size}, got {hidden_states.shape[-1]}."
            )

        if hidden_states.dim() == 2:
            hidden_states_3d = hidden_states.unsqueeze(0)
            flatten_hidden_states = hidden_states
            token_shape = hidden_states.shape[:-1]
        else:
            token_shape = hidden_states.shape[:-1]
            flatten_hidden_states = hidden_states.reshape(-1, self.hidden_size)
            hidden_states_3d = hidden_states

        if (top_k_indices is None) != (top_k_gates is None):
            raise ValueError(
                "top_k_indices and top_k_gates must be both provided or both None."
            )

        if top_k_indices is None:
            top_k_indices, top_k_gates = self.gating(flatten_hidden_states)
        else:
            top_k_indices = top_k_indices.reshape(-1, self.top_k)
            top_k_gates = top_k_gates.reshape(-1, self.top_k)

        route_smooth_scale = self.w13_input_scale
        if route_smooth_scale is None:
            route_smooth_scale = torch.ones(
                (self.num_experts, self.hidden_size),
                dtype=torch.float32,
                device=flatten_hidden_states.device,
            )

        (
            routed_input_i8,
            routed_gates,
            sorted_token_ids,
            src_to_dst,
            token_count,
            input_scale,
        ) = self.init_routing(
            hidden_states_3d,
            top_k_gates,
            top_k_indices,
            smooth_scale=route_smooth_scale,
            quant_mode=quant_mode,
        )

        if sorted_token_ids is None or sorted_token_ids.numel() == 0:
            if shared_output is not None:
                return shared_output
            return torch.zeros(
                (*token_shape, self.hidden_size),
                dtype=self.output_dtype,
                device=hidden_states.device,
            )

        routed_input_i8 = self._pad_last_dim(
            routed_input_i8, self.w13_weight.shape[-1]
        )
        fc1_output = self.fc1(
            input=routed_input_i8,
            weight=self.w13_weight,
            token_count=token_count,
            weight_scale=self.w13_weight_scale,
            input_scale=input_scale,
            bias=self.w13_bias,
        )

        act_i8, act_scale = self.act_quant(
            input=fc1_output,
            smooth_scale=self.w2_input_scale,
            sorted_token_ids=sorted_token_ids,
            topk_indices=None,
            fc1_intermediate_size=fc1_output.shape[-1],
            beta=beta,
            quant_mode=quant_mode,
        )
        act_i8 = self._pad_last_dim(act_i8, self.w2_weight.shape[-1])

        shared_output_2d = None
        if shared_output is not None:
            if shared_output.shape[:-1] != token_shape:
                raise ValueError(
                    f"shared_output shape {tuple(shared_output.shape)} does not match token shape {tuple(token_shape)}."
                )
            shared_output_2d = shared_output.reshape(-1, shared_output.shape[-1])

        combined_output = self.fc2(
            input=act_i8,
            weight=self.w2_weight,
            top_k_gates=routed_gates,
            token_indices=sorted_token_ids,
            src_to_dst=src_to_dst,
            token_count=token_count,
            shared_output=shared_output_2d,
            weight_scale=self.w2_weight_scale,
            input_scale=act_scale,
            bias=self.w2_bias,
        )

        if self.routed_scaling_factor != 1.0:
            if shared_output_2d is None:
                combined_output = combined_output * self.routed_scaling_factor
            else:
                combined_output = shared_output_2d + (
                    combined_output - shared_output_2d
                ) * self.routed_scaling_factor

        return combined_output.reshape(*token_shape, self.hidden_size)