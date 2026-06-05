import torch
from typing import Union

from mojo_opset.core import MojoMoEGating
from mojo_opset.core import MojoMoEDispatch
from mojo_opset.core import MojoMoECombine
from mojo_opset.core import MojoMoEDynamicQuant
from mojo_opset.core import MojoExperts
from mojo_opset.core import MojoQuantExperts
from mojo_opset.core import MojoMoE
from mojo_opset.core import MojoQuantMoE

from ixformer import functions as ixf_f

def decompose_fp32_to_3bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    x0 = x.bfloat16()
    r1 = x - x0.float()
    x1 = r1.bfloat16()
    r2 = r1 - x1.float()
    x2 = r2.bfloat16()

    return x0, x1, x2

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
    """load_state_dict post-hook: convert int4/int8 weights from TN (checkpoint) to NN (ixformer) format."""
    device = module.up_proj_weight.device
    module.up_proj_quantize.inv_smooth_scale = torch.nn.Parameter(module.up_proj_quantize.inv_smooth_scale.data.to(dtype=torch.bfloat16))
    
    if module.up_weight_dtype == "int4":
        N_up = module.intermediate_size * 2
        K_up = module.hidden_size
        up_nn = _repack_int4_tn_to_nn(module.up_proj_weight.data, N_up, K_up)
        up_scale_nn = module.up_proj_weight_scale.data.permute(0, 2, 1).contiguous()
        module.register_buffer("up_proj_weight", up_nn.to(device))
        module.up_proj_weight_scale = torch.nn.Parameter(up_scale_nn.to(device=device, dtype=torch.float32))
    elif module.up_weight_dtype == torch.int8:
        up_nn = module.up_proj_weight.data.transpose(1, 2).contiguous()
        up_scale_nn = module.up_proj_weight_scale.data.contiguous()
        module.register_buffer("up_proj_weight", up_nn.to(device))
        module.up_proj_weight_scale = torch.nn.Parameter(up_scale_nn.to(device=device, dtype=torch.float32))

    if module.down_weight_dtype == "int4":
        N_down = module.hidden_size
        K_down = module.intermediate_size
        down_nn = _repack_int4_tn_to_nn(module.down_proj_weight.data, N_down, K_down)
        down_scale_nn = module.down_proj_weight_scale.data.permute(0, 2, 1).contiguous()
        module.register_buffer("down_proj_weight", down_nn.to(device))
        module.down_proj_weight_scale = torch.nn.Parameter(down_scale_nn.to(device=device, dtype=torch.float32))
    elif module.down_weight_dtype == torch.int8:
        down_nn = module.down_proj_weight.data.transpose(1, 2).contiguous()
        down_scale_nn = module.down_proj_weight_scale.data.contiguous()
        module.register_buffer("down_proj_weight", down_nn.to(device))
        module.down_proj_weight_scale = torch.nn.Parameter(down_scale_nn.to(device=device, dtype=torch.float32))


class IxformerMoEGating(MojoMoEGating):
    supported_platforms_list = ["ilu"]

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, **kwargs):
        super().__init__(hidden_size, num_experts, top_k, **kwargs)
        for name in ("gate_weight_bf16_tn_0", "gate_weight_bf16_tn_1", "gate_weight_bf16_tn_2"):
            self.register_buffer(
                name,
                torch.empty((num_experts, hidden_size), dtype=torch.bfloat16, device=self.gate_weight.device),
                persistent=False,
            )
        self.register_load_state_dict_post_hook(self._transform_gate_weight_post_hook)

    @staticmethod
    def _transform_gate_weight_post_hook(module, incompatible_keys):
        gw0, gw1, gw2 = decompose_fp32_to_3bf16(module.gate_weight.data.T.contiguous())
        module.gate_weight_bf16_tn_0.copy_(gw0)
        module.gate_weight_bf16_tn_1.copy_(gw1)
        module.gate_weight_bf16_tn_2.copy_(gw2)

    def forward(self, hidden_states: torch.Tensor):
        gate_logits = ixf_f.triple_gemm_bf16_bf16_fp32(hidden_states, self.gate_weight_bf16_tn_2, self.gate_weight_bf16_tn_1, self.gate_weight_bf16_tn_0)
        top_k_gates, top_k_indices = ixf_f.moe_topk_softmax(gate_logits, self.top_k, renormalize=True)

        return top_k_indices, top_k_gates


class IxformerMoEDynamicQuant(MojoMoEDynamicQuant):
    """Ixformer placeholder: smooth_scale holder only; actual quant is fused in dispatch."""
    supported_platforms_list = ["ilu"]


class IxformerMoEDispatch(MojoMoEDispatch):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        """Sort tokens by expert id and gather into contiguous layout.

        Matches the core ``MojoMoEDispatch`` contract, implemented with a single
        fused ixformer kernel (``moe_dispatch``).

        Returns:
            sorted_hidden_states: (num_tokens * top_k, hidden_size).
            tokens_per_expert:    (num_experts,), int32.
            sorted_gates:         (num_tokens * top_k, 1).
            token_indices:        (num_tokens * top_k,), original token id per slot.
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = ixf_f.moe_dispatch(
            hidden_states,
            top_k_gates,
            top_k_indices,
            self.num_experts,
        )

        return sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices


class IxformerExperts(MojoExperts):
    supported_platforms_list = ["ilu"]
    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 intermediate_size: int,
                 activation: str = "swiglu",
                 **kwargs):
        super().__init__(num_experts, hidden_size, intermediate_size, activation, **kwargs)

        self.hidden_size = hidden_size

    def forward(self,
                sorted_hidden_states: torch.Tensor,
                tokens_per_expert: torch.Tensor):
        enable_cuda_graph = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        
        if not enable_cuda_graph:
            tokens_per_expert = tokens_per_expert.cpu()

            group_gemm_output1 = ixf_f.moe_w16a16_group_gemm(
                input=sorted_hidden_states,
                weight=self.up_proj_weight,
                output_dtype=sorted_hidden_states.dtype,
                tokens_per_experts=tokens_per_expert,
                dst_to_src=None,
                format="TN",
            )
        
        else:
            group_gemm_output1 = ixf_f.moe_w16a16_group_gemv(
                input=sorted_hidden_states,
                weight=self.up_proj_weight,
                output_dtype=sorted_hidden_states.dtype,
                tokens_per_experts_gpu=tokens_per_expert,
                dst_to_src=None,
                format="TN",
            )

        act = ixf_f.silu_and_mul(group_gemm_output1)

        group_gemm_output2 = torch.empty(sorted_hidden_states.shape[0], self.hidden_size, dtype=sorted_hidden_states.dtype, device=sorted_hidden_states.device)

        if not enable_cuda_graph:
            ixf_f.moe_w16a16_group_gemm(
                input=act,
                weight=self.down_proj_weight,
                output_dtype=sorted_hidden_states.dtype,
                tokens_per_experts=tokens_per_expert,
                dst_to_src=None,
                format="TN",
                output=group_gemm_output2,
            )
        else:
            ixf_f.moe_w16a16_group_gemv(
                input=act,
                weight=self.down_proj_weight,
                output_dtype=sorted_hidden_states.dtype,
                tokens_per_experts_gpu=tokens_per_expert,
                dst_to_src=None,
                format="TN",
                output=group_gemm_output2,
            )

        return group_gemm_output2


class IxformerQuantExperts(MojoQuantExperts):
    supported_platforms_list = ["ilu"]

    def __init__(self,
                 num_experts: int,
                 hidden_size: int,
                 intermediate_size: int,
                 activation: str = "swiglu",
                 quant_dtype: torch.dtype = torch.int8,
                 up_quant_group_size: int = -1,
                 up_weight_dtype: Union[str, torch.dtype] = torch.int8,
                 down_quant_group_size: int = -1,
                 down_weight_dtype: Union[str, torch.dtype] = torch.int8,
                 **kwargs):
        super().__init__(
            num_experts,
            hidden_size,
            intermediate_size,
            activation,
            quant_dtype,
            up_quant_group_size,
            up_weight_dtype,
            down_quant_group_size,
            down_weight_dtype,
            **kwargs,
        )

        if self.hidden_size % 64 != 0 or self.intermediate_size % 64 != 0:
            raise NotImplementedError(
                f"IxformerQuantExperts only supports hidden_size and intermediate_size divisible by 64, got {self.hidden_size} and {self.intermediate_size}."
            )
        if self.up_weight_dtype == torch.int8 and self.up_quant_group_size != -1:
            raise NotImplementedError(
                f"IxformerQuantExperts only supports up_weight_dtype='torch.int8' with up_quant_group_size=-1, got {self.up_weight_dtype} and {self.up_quant_group_size}."
            )
        if self.down_weight_dtype == torch.int8 and self.down_quant_group_size != -1:
            raise NotImplementedError(
                f"IxformerQuantExperts only supports down_weight_dtype='torch.int8' with down_quant_group_size=-1, got {self.down_weight_dtype} and {self.down_quant_group_size}."
            )
        if self.up_weight_dtype == "int4":
            if self.up_quant_group_size not in [128, 256, 320, 512]:
                raise NotImplementedError(
                    f"IxformerQuantExperts: up_weight_dtype is 'int4' and up_quant_group_size must be 128, 256, 320, or 512, got {self.up_weight_dtype} and {self.up_quant_group_size}."
                )
            if self.hidden_size % self.up_quant_group_size != 0:
                raise NotImplementedError(
                    f"IxformerQuantExperts: up_weight_dtype is 'int4' and k (hidden_size) must be divisible by up_quant_group_size, got hidden_size={self.hidden_size} and up_quant_group_size={self.up_quant_group_size}."
                )
            if self.intermediate_size * 2 < 256 or self.hidden_size < 256:
                raise NotImplementedError(
                    f"IxformerQuantExperts: up_weight_dtype is 'int4' and intermediate_size * 2 must be >= 256, hidden_size must be >= 256, got {self.hidden_size} and {self.intermediate_size}."
                )
        if self.down_weight_dtype == "int4":
            if self.down_quant_group_size not in [128, 256, 320, 512]:
                raise NotImplementedError(
                    f"IxformerQuantExperts: down_weight_dtype is 'int4' and down_quant_group_size must be 128, 256, 320, or 512, got {self.down_weight_dtype} and {self.down_quant_group_size}."
                )
            if self.intermediate_size % self.down_quant_group_size != 0:
                raise NotImplementedError(
                    f"IxformerQuantExperts: down_weight_dtype is 'int4' and k (intermediate_size) must be divisible by down_quant_group_size, got intermediate_size={self.intermediate_size} and down_quant_group_size={self.down_quant_group_size}."
                )
            if self.intermediate_size < 256 or self.hidden_size < 256:
                raise NotImplementedError(
                    f"IxformerQuantExperts: down_weight_dtype is 'int4' and intermediate_size must be >= 256, hidden_size must be >= 256, got {self.hidden_size} and {self.intermediate_size}."
                )

        setattr(self.up_proj_weight_scale, "force_dtype", torch.float32)
        setattr(self.down_proj_weight_scale, "force_dtype", torch.float32)

        self.register_load_state_dict_post_hook(_swizzle_weights_post_hook)
        
        self.output_dtype = torch.bfloat16

        if self.up_weight_dtype == "int4" and self.down_weight_dtype == "int4": 
            self.gdr_device_buffer1 = torch.zeros([8 * 1024 * 1024], dtype=torch.int8, device="cuda")
            self.gdr_device_buffer2 = torch.zeros([8 * 1024 * 1024], dtype=torch.int8, device="cuda")
            torch.cuda.synchronize()
            self.gdr_buffer_ptr1 = ixf_f.new_gdr_buffer(self.gdr_device_buffer1)
            self.gdr_buffer_ptr2 = ixf_f.new_gdr_buffer(self.gdr_device_buffer2)

    def forward(self, 
                sorted_hidden_states: torch.Tensor,
                input_scale: torch.Tensor,
                tokens_per_expert: torch.Tensor,
                topk_indices: torch.Tensor,
                sorted_token_ids: torch.Tensor,
                enable_cuda_graph: bool = False):
        
        if enable_cuda_graph:
            assert torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

        if self.up_weight_dtype == torch.int8:
            if not enable_cuda_graph:
                group_gemm_output1 = ixf_f.moe_w8a8_group_gemm(
                    input=sorted_hidden_states,
                    weight=self.up_proj_weight,
                    i_scales=input_scale,
                    w_scales=self.up_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    format="NN"
                )
            else:
                group_gemm_output1 = ixf_f.moe_w8a8_group_gemv(
                    input=sorted_hidden_states,
                    weight=self.up_proj_weight,
                    i_scales=input_scale,
                    w_scales=self.up_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    format=0
                )
        elif self.up_weight_dtype == "int4":
            if not enable_cuda_graph:
                group_gemm_output1 = ixf_f.moe_w4a8_group_gemm(
                    input=sorted_hidden_states,
                    weight=self.up_proj_weight,
                    i_scales=input_scale,
                    w_scales=self.up_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    format=0,
                    version=1,
                    group_size=self.up_quant_group_size,
                    gdr_buffer_ptr=self.gdr_buffer_ptr1,
                )
            else:
                group_gemm_output1 = ixf_f.moe_w4a8_group_gemv(
                    input=sorted_hidden_states,
                    weight=self.up_proj_weight,
                    i_scales=input_scale,
                    w_scales=self.up_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    format=0,
                    version=1,
                    group_size=self.up_quant_group_size,
                )
        else:
            raise NotImplementedError(f"IxformerQuantExperts: up_weight_dtype must be 'torch.int8' or 'int4', got {self.up_weight_dtype}.")

        act_i8, act_scale = ixf_f.activation_dynamic_scaled_int8(
            input=group_gemm_output1,
            smooth_scales=self.down_proj_quantize.inv_smooth_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=topk_indices,
            act_type="swiglu",
            output_format=1 if enable_cuda_graph and self.down_weight_dtype == "int4" else 0
        )
        num_tokens, top_k = topk_indices.shape

        group_gemm_output2 = torch.empty(num_tokens * top_k, self.hidden_size, dtype=self.output_dtype, device=sorted_token_ids.device)
        
        if self.down_weight_dtype == torch.int8:
            if not enable_cuda_graph:
                ixf_f.moe_w8a8_group_gemm(
                    input=act_i8,
                    weight=self.down_proj_weight,
                    i_scales=act_scale,
                    w_scales=self.down_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    dst_to_src=sorted_token_ids,
                    format="NN",
                    output=group_gemm_output2,
                )
            else:
                ixf_f.moe_w8a8_group_gemv(
                    input=act_i8,
                    weight=self.down_proj_weight,
                    i_scales=act_scale,
                    w_scales=self.down_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    dst_to_src=sorted_token_ids,
                    format=0,
                    output=group_gemm_output2,
                )
        elif self.down_weight_dtype == "int4":
            if not enable_cuda_graph:
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
                    group_size=self.down_quant_group_size,
                    output=group_gemm_output2,
                    gdr_buffer_ptr=self.gdr_buffer_ptr2,
                )
            else:
                ixf_f.moe_w4a8_group_gemv(
                    input=act_i8,
                    weight=self.down_proj_weight,
                    i_scales=act_scale,
                    w_scales=self.down_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    dst_to_src=sorted_token_ids,
                    format=0,
                    version=1,
                    group_size=self.down_quant_group_size,
                    output=group_gemm_output2,
                )
        else:
            raise NotImplementedError(f"IxformerQuantExperts: down_weight_dtype must be 'torch.int8' or 'int4', got {self.down_weight_dtype}.")
        
        return group_gemm_output2
    
    def __del__(self):
        if hasattr(self, "gdr_buffer_ptr1"):
            ixf_f.delete_gdr_buffer(self.gdr_buffer_ptr1)
        if hasattr(self, "gdr_buffer_ptr2"):
            ixf_f.delete_gdr_buffer(self.gdr_buffer_ptr2)


class IxformerMoECombine(MojoMoECombine):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        output_buffer: torch.Tensor,
        expert_outputs: torch.Tensor,
        sorted_gates: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted scatter-reduce of expert outputs back to token positions.

        Args:
            output_buffer:  (num_tokens, hidden_size), include-self buffer. It is
                converted to float32 accumulator by ixformer.
            expert_outputs: (num_tokens * top_k, hidden_size), fp16/bf16/fp32.
            sorted_gates:   (num_tokens * top_k, 1), float32.
            token_indices:  (num_tokens * top_k,), int32. Maps each expert-sorted
                slot back to its original token id.

        Returns:
            combined: (num_tokens, hidden_size), same dtype as expert_outputs.
        """
        return ixf_f.moe_combine(
            output_buffer,
            expert_outputs,
            sorted_gates,
            token_indices,
            multiply_by_gates=self.multiply_by_gates,
        )


class IxformerMoE(MojoMoE):
    supported_platforms_list = ["ilu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        top_k_indices, top_k_gates = self.gating(hidden_states)

        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states,
            top_k_gates,
            top_k_indices,
        )

        expert_outputs = self.experts(
            sorted_hidden_states,
            tokens_per_expert,
        )
        output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)

        return combined


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
        up_quant_group_size: int = -1,
        up_weight_dtype: Union[torch.dtype, str] = torch.int8,
        down_quant_group_size: int = -1,
        down_weight_dtype: Union[torch.dtype, str] = torch.int8,
        **kwargs
    ):
        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            activation,
            quant_dtype,
            up_quant_group_size,
            up_weight_dtype,
            down_quant_group_size,
            down_weight_dtype,
            **kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            enable_cuda_graph = True
        else:
            enable_cuda_graph = False
        
        if hidden_states.dtype == torch.float16:
            raise NotImplementedError(f"IxformerQuantMoE: hidden_states dtype must be 'torch.bfloat16', got {hidden_states.dtype}.")

        top_k_indices, top_k_gates = self.gating(hidden_states)

        i8_hs, sorted_token_ids, src_to_dst, tokens_per_expert, quant_scale = self.dispatch(
            hidden_states,
            top_k_indices,
            self.experts.up_proj_quantize.inv_smooth_scale,
            weight_dtype=self.up_weight_dtype,
            enable_cuda_graph=enable_cuda_graph
        )

        expert_outputs = self.experts(
            i8_hs,
            quant_scale,
            tokens_per_expert,
            top_k_indices,
            sorted_token_ids,
            enable_cuda_graph=enable_cuda_graph      
        )

        expert_outputs = expert_outputs.view(-1, self.top_k, self.hidden_size)

        combined = self.combine(expert_outputs, top_k_gates, src_to_dst)

        return combined
