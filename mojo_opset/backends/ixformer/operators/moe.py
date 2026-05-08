import torch
from typing import Optional, Tuple, Union

from mojo_opset.core import MojoMoEGating
from mojo_opset.core import MojoMoEDispatch
from mojo_opset.core import MojoMoECombine
from mojo_opset.core import MojoMoEDynamicQuant
from mojo_opset.core import MojoQuantExperts
from mojo_opset.core import MojoQuantMoE

from ixformer import functions as ixf_f

def _repack_int4_tn_to_nn(packed_tn: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """
    Pure layout conversion: TN-packed int4 -> NN-packed int4 with tensor-core swizzle.
    No re-quantization, exact bit-level transformation.

    Args:
        packed_tn: (E, N//2, K) int8 — checkpoint TN format, pairs packed along N dim.
        N: full (unpacked) output dimension.
        K: input dimension.  N and K must both be divisible by 32.

    Returns:
        (E, K, N//2) int8 — NN format with tensor-core swizzle, ready for ixformer kernel.
    """
    device = packed_tn.device
    packed_tn = packed_tn.cuda()
    E = packed_tn.shape[0]
    u8 = packed_tn.to(torch.uint8)
    low = (u8 & 0x0F).to(torch.int8)
    high = ((u8 >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(E, N, K, dtype=torch.int8, device=packed_tn.device)
    unpacked[:, 0::2, :] = low
    unpacked[:, 1::2, :] = high

    out = unpacked.transpose(-2, -1).contiguous()
    out = out.view(E, K // 32, 2, 16, N // 32, 2, 16)
    out = out.permute(0, 1, 5, 3, 4, 2, 6).contiguous().view(E, K, N)

    out = out.view(E, K, N // 32, 32)
    packed = out.new_empty(E, K, N // 32, 16)
    for i in range(16):
        sign_low = (out[:, :, :, i] < 0).to(torch.int8)
        lo = sign_low * 8 + (out[:, :, :, i] & 0x07)
        hi = out[:, :, :, i + 16] << 4
        packed[:, :, :, i] = hi + lo

    return packed.reshape(E, K, N // 2).contiguous().to(device)


def _swizzle_weights_post_hook(module, incompatible_keys):
    """load_state_dict post-hook: convert int4 weights from TN (checkpoint) to NN (ixformer) format."""
    device = module.up_proj_weight.device

    N_up = module.intermediate_size * 2
    K_up = module.hidden_size
    up_nn = _repack_int4_tn_to_nn(module.up_proj_weight.data, N_up, K_up)
    up_scale_nn = module.up_proj_weight_scale.data.permute(0, 2, 1).contiguous()

    N_down = module.hidden_size
    K_down = module.intermediate_size
    down_nn = _repack_int4_tn_to_nn(module.down_proj_weight.data, N_down, K_down)
    down_scale_nn = module.down_proj_weight_scale.data.permute(0, 2, 1).contiguous()

    module.register_buffer("up_proj_weight", up_nn.to(device))
    module.register_buffer("down_proj_weight", down_nn.to(device))
    module.up_proj_weight_scale = torch.nn.Parameter(up_scale_nn.to(device=device, dtype=torch.float32))
    module.down_proj_weight_scale = torch.nn.Parameter(down_scale_nn.to(device=device, dtype=torch.float32))


class IxformerMoEGating(MojoMoEGating):
    supported_platforms_list = ["ilu"]

    def forward(self, hidden_states: torch.Tensor):
        assert self.gate_weight.dtype == torch.float32
        gate_logits = ixf_f.mixed_type_linear(hidden_states, self.gate_weight, format="NN")
        top_k_gates, top_k_indices = ixf_f.moe_topk_softmax(gate_logits, self.top_k, renormalize=True)

        return top_k_indices, top_k_gates


class IxformerMoEDynamicQuant(MojoMoEDynamicQuant):
    """Ixformer placeholder: smooth_scale holder only; actual quant is fused in dispatch."""
    supported_platforms_list = ["ilu"]


class IxformerMoEDispatchQuant(MojoMoEDispatch):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        smooth_scale: torch.Tensor
    ):
        if hidden_states.dim() == 3:
            num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            dim = hidden_states.shape[-1]
            hidden_states = hidden_states.view(num_tokens, dim)
        elif hidden_states.dim() == 2:
            dim = hidden_states.shape[-1]

        num_tokens, top_k = top_k_indices.shape

        (src_to_dst, 
         sorted_token_ids,
         expert_sizes_gpu, 
         expert_sizes_cpu) = ixf_f.moe_compute_token_index(top_k_indices, self.num_experts)
        
        expand_tokens = num_tokens * top_k

        i8_hidden_states, quant_scale = ixf_f.moe_expand_input_dynamic_scaled_int8(
                                               hidden_states=hidden_states,
                                               dst_to_src=sorted_token_ids,
                                               dst_tokens=expand_tokens, 
                                               topk=top_k,
                                               src_to_dst=src_to_dst,
                                               topk_ids=top_k_indices,
                                               smooth_scales=smooth_scale)

        return (
            i8_hidden_states.view(-1, dim),
            sorted_token_ids,
            src_to_dst,
            expert_sizes_gpu,
            quant_scale,
        )


class IxformerQuantExperts(MojoQuantExperts):
    supported_platforms_list = ["ilu"]

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 intermediate_size: int,
                 activation: str = "swiglu",
                 quant_dtype: torch.dtype = torch.int8,
                 quant_group_size: int = -1,
                 weight_dtype: Union[str, torch.dtype] = torch.int8,
                 **kwargs):
        super().__init__(num_experts, hidden_size, intermediate_size, activation, quant_dtype, quant_group_size, weight_dtype, **kwargs)
        
        if self.weight_dtype not in ("int4"):
            raise NotImplementedError(f"IxformerQuantExperts only supports weight_dtype='int4', got {self.weight_dtype}.")
        
        if self.quant_group_size not in [256, 320, 512]:
            raise NotImplementedError(f"IxformerQuantExperts: quant_group_size must be 256, 320, or 512, got {self.quant_group_size}.")

        setattr(self.up_proj_weight_scale, "force_dtype", torch.float32)
        setattr(self.down_proj_weight_scale, "force_dtype", torch.float32)

        if self.weight_dtype in ("int4"):
            self.register_load_state_dict_post_hook(_swizzle_weights_post_hook)
        
        self.output_dtype = torch.bfloat16

    def forward(self, 
                sorted_hidden_states: torch.Tensor,
                input_scale: torch.Tensor,
                tokens_per_expert: torch.Tensor,
                topk_indices: torch.Tensor,
                sorted_token_ids: torch.Tensor):
        
        group_gemm_output1 = ixf_f.moe_w4a8_group_gemm(
            input=sorted_hidden_states,
            weight=self.up_proj_weight,
            i_scales=input_scale,
            w_scales=self.up_proj_weight_scale,
            output_dtype=self.output_dtype,
            tokens_per_experts=tokens_per_expert,
            format=0,
            version=1,
            group_size=self.quant_group_size,
        )

        act_i8, act_scale = ixf_f.activation_dynamic_scaled_int8(
            input=group_gemm_output1,
            smooth_scales=self.down_proj_quantize.inv_smooth_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=topk_indices,
            act_type="swiglu",
        )
        num_tokens, top_k = topk_indices.shape

        group_gemm_output2 = torch.empty(num_tokens * top_k, self.hidden_size, dtype=self.output_dtype, device=sorted_token_ids.device)
        
        ixf_f.moe_w4a8_group_gemm(
            input=act_i8,
            weight=self.down_proj_weight,
            i_scales=act_scale,
            w_scales=self.down_proj_weight_scale,
            output_dtype=self.output_dtype,
            tokens_per_experts=tokens_per_expert,
            dst_to_src=sorted_token_ids,
            format=0,
            version=1,
            group_size=self.quant_group_size,
            output=group_gemm_output2,
        )

        return group_gemm_output2


class IxformerMoECombine(MojoMoECombine):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        expert_outputs: torch.Tensor,
        top_k_gates: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> torch.Tensor:
        reduce_mask = src_to_dst == -1
        combined_output = ixf_f.moe_output_reduce_sum(
            input=expert_outputs,
            topk_weight=top_k_gates,
            mask=reduce_mask,
        )
        return combined_output


class IxformerQuantMoE(MojoQuantMoE):
    supported_platforms_list = ["ilu"]

    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size=None,
        activation: str = "swiglu",
        quant_dtype: torch.dtype = torch.int8,
        quant_group_size: int = -1,
        weight_dtype: Union[torch.dtype, str] = torch.int8,
        enable_cuda_graph: bool = False,
        **kwargs
    ):
        super().__init__(num_experts, top_k, hidden_size, intermediate_size, activation, quant_dtype, quant_group_size, weight_dtype, **kwargs)
        
        if self.quant_group_size not in [256, 320, 512]:
            raise NotImplementedError(f"IxformerQuantMoE: quant_group_size must be 256, 320, or 512, got {self.quant_group_size}.")

        self.enable_cuda_graph = enable_cuda_graph

        if self.enable_cuda_graph:
            raise NotImplementedError("IxformerQuantMoE: enable_cuda_graph is not supported.")
        
        if self.weight_dtype not in ("int4"):
            raise NotImplementedError(f"IxformerQuantMoE: weight_dtype must be 'int4', got {self.weight_dtype}.")


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        top_k_indices, top_k_gates = self.gating(hidden_states)

        i8_hs, sorted_token_ids, src_to_dst, tokens_per_expert, quant_scale = self.dispatch(
            hidden_states,
            top_k_indices,
            self.experts.up_proj_quantize.inv_smooth_scale,
        )

        if not self.enable_cuda_graph:
            tokens_per_expert = tokens_per_expert.cpu()

        expert_outputs = self.experts(
            i8_hs,
            quant_scale,
            tokens_per_expert,
            top_k_indices,
            sorted_token_ids      
        )

        expert_outputs = expert_outputs.view(-1, self.top_k, self.hidden_size)

        combined = self.combine(expert_outputs, top_k_gates, src_to_dst)

        return combined
