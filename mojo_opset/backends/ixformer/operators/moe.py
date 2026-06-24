import os
import torch
import torch.distributed as dist
from typing import Union

from mojo_opset.core import MojoMoE
from mojo_opset.core import MojoQuantMoE

from ixformer import functions as ixf_f
import ixformer.distributed as ixfd

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


def _attach_gating_bf16_buffers(gating_module):
    """Attach gate_weight bf16 TN buffers and post-hook to a parameter-holder gating module."""
    num_experts = gating_module.gate_weight.shape[1]
    hidden_size = gating_module.gate_weight.shape[0]
    for name in ("gate_weight_bf16_tn_0", "gate_weight_bf16_tn_1", "gate_weight_bf16_tn_2"):
        gating_module.register_buffer(
            name,
            torch.empty((num_experts, hidden_size), dtype=torch.bfloat16, device=gating_module.gate_weight.device),
            persistent=False,
        )

    def _transform_gate_weight_post_hook(module, incompatible_keys):
        gate_weight = module.gate_weight.data.T.contiguous()
        gw0, gw1, gw2 = decompose_fp32_to_3bf16(gate_weight)
        module.register_buffer("gate_weight_bf16_tn_0", gw0)
        module.register_buffer("gate_weight_bf16_tn_1", gw1)
        module.register_buffer("gate_weight_bf16_tn_2", gw2)

    gating_module.register_load_state_dict_post_hook(_transform_gate_weight_post_hook)


class IxformerMoE(MojoMoE):
    """Fused bf16 MoE: gating, dispatch, group-gemm experts, and combine in a single forward."""

    supported_platforms_list = ["ilu"]
    _use_fused_moe = True

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
        **kwargs,
    ):
        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            activation,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            **kwargs,
        )
        _attach_gating_bf16_buffers(self.gating)

        # GDR buffer feeds per-expert token counts back to the GPU without a CPU sync.
        self.gdr_device_buffer = torch.zeros([self.num_experts + 1], dtype=torch.int, device="cuda")
        torch.cuda.synchronize()
        self.gdr_buffer_ptr = ixf_f.new_gdr_buffer(self.gdr_device_buffer)
        self.disable_sync = os.environ.get("IXFORMER_DISABLE_SYNC", "0").strip().lower() == "1"

    def __del__(self):
        if hasattr(self, "gdr_buffer_ptr"):
            ixf_f.delete_gdr_buffer(self.gdr_buffer_ptr)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            enable_cuda_graph = True
        else:
            enable_cuda_graph = False

        if enable_cuda_graph and not self.disable_sync:
            raise RuntimeError("IxformerMoE: CUDA graph capture requires disable_sync=True (set IXFORMER_DISABLE_SYNC=1).")

        # DP input: gather peer ranks' shards so gating/dispatch see the full token set.
        if self.dp_input and self.ep_size > 1:
            local_tokens = hidden_states.shape[0]
            full = torch.empty(
                local_tokens * self.ep_size, hidden_states.shape[1],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            ixfd.all_gather_into_tensor(full, hidden_states.contiguous(), group=self.ep_group, async_op=True)
            hidden_states = full

        # triple_gemm uses 3 bf16 components of the fp32 gate weight to emulate fp32 matmul precision on bf16 HW.
        gate_logits = ixf_f.triple_gemm_bf16_bf16_fp32(
            hidden_states,
            self.gating.gate_weight_bf16_tn_2,
            self.gating.gate_weight_bf16_tn_1,
            self.gating.gate_weight_bf16_tn_0,
        )
        top_k_gates, top_k_indices = ixf_f.moe_topk_softmax(gate_logits, self.gating.top_k, renormalize=True)

        num_tokens, dim = hidden_states.shape

        need_gpu_size = self.disable_sync or enable_cuda_graph
        graph_safe_ep = self.ep_size > 1 and need_gpu_size

        if self.ep_size > 1:
            # _ep variant filters tokens to the local expert range [ep_start, ep_end);
            # expand_tokens is the actual local token count after filtering.
            (src_to_dst,
             sorted_token_ids,
             expert_sizes_gpu,
             expert_sizes_cpu,
             expand_tokens) = ixf_f.moe_compute_token_index_ep(
                top_k_indices,
                self.num_experts,
                self.ep_start,
                self.ep_end,
                gdr_buffer_ptr=None if need_gpu_size else self.gdr_buffer_ptr,
                graph_safe=graph_safe_ep,
            )
        else:
            (src_to_dst,
             sorted_token_ids,
             expert_sizes_gpu,
             expert_sizes_cpu) = ixf_f.moe_compute_token_index(
                top_k_indices,
                self.num_experts,
                gdr_buffer_ptr=None if need_gpu_size else self.gdr_buffer_ptr,
            )
            expand_tokens = num_tokens * self.top_k

        dispatch_tokens = num_tokens * self.top_k if graph_safe_ep else expand_tokens

        if not graph_safe_ep and sorted_token_ids.shape[0] == 0:
            combined = torch.zeros(
                num_tokens, self.hidden_size,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
        else:
            tokens_per_expert = expert_sizes_gpu if need_gpu_size else expert_sizes_cpu

            sorted_hidden_states = ixf_f.moe_expand_input(
                hidden_states=hidden_states,
                dst_to_src=sorted_token_ids,
                dst_tokens=dispatch_tokens,
                topk=self.top_k,
                src_to_dst=src_to_dst,
                check_valid=graph_safe_ep,
            ).view(-1, dim)

            # gemv variant takes GPU-side sizes so kernel shapes don't depend on CPU values during graph capture.
            if not need_gpu_size:
                group_gemm_output1 = ixf_f.moe_w16a16_group_gemm(
                    input=sorted_hidden_states,
                    weight=self.experts.up_proj_weight,
                    output_dtype=sorted_hidden_states.dtype,
                    tokens_per_experts=tokens_per_expert,
                    dst_to_src=None,
                    format="TN",
                )
            else:
                group_gemm_output1 = ixf_f.moe_w16a16_group_gemv(
                    input=sorted_hidden_states,
                    weight=self.experts.up_proj_weight,
                    output_dtype=sorted_hidden_states.dtype,
                    tokens_per_experts_gpu=tokens_per_expert,
                    dst_to_src=None,
                    format="TN",
                )

            act = ixf_f.silu_and_mul(group_gemm_output1)

            group_gemm_output2 = torch.empty(
                num_tokens * self.top_k, self.hidden_size,
                dtype=sorted_hidden_states.dtype, device=sorted_hidden_states.device,
            )

            if not need_gpu_size:
                ixf_f.moe_w16a16_group_gemm(
                    input=act,
                    weight=self.experts.down_proj_weight,
                    output_dtype=sorted_hidden_states.dtype,
                    tokens_per_experts=tokens_per_expert,
                    dst_to_src=sorted_token_ids,
                    format="TN",
                    output=group_gemm_output2,
                )
            else:
                ixf_f.moe_w16a16_group_gemv(
                    input=act,
                    weight=self.experts.down_proj_weight,
                    output_dtype=sorted_hidden_states.dtype,
                    tokens_per_experts_gpu=tokens_per_expert,
                    dst_to_src=sorted_token_ids,
                    format="TN",
                    output=group_gemm_output2,
                )

            expert_outputs = group_gemm_output2.view(num_tokens, self.top_k, -1)

            # src_to_dst == -1 marks padding slots that must not be summed.
            reduce_mask = src_to_dst == -1
            combined = ixf_f.moe_output_reduce_sum(
                input=expert_outputs,
                topk_weight=top_k_gates,
                mask=reduce_mask,
            )

        # Sum partial expert outputs across EP ranks; reduce_scatter slices the result back to the rank's DP shard.
        if self.ep_size > 1:
            if self.dp_input:
                local_combined = torch.empty(
                    combined.shape[0] // self.ep_size, combined.shape[1],
                    dtype=combined.dtype, device=combined.device,
                )
                ixfd.reduce_scatter_tensor(local_combined, combined.contiguous(), group=self.ep_group, async_op=True)
                combined = local_combined
            else:
                ixfd.all_reduce(combined, group=self.ep_group, async_op=True)

        return combined


class IxformerQuantMoE(MojoQuantMoE):
    """Fused w4a8/w8a8 quant MoE: gating, smooth-quant dispatch, quant group-gemm experts, and combine in one forward."""

    supported_platforms_list = ["ilu"]
    _use_fused_moe = True

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
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            **kwargs,
        )

        if os.environ.get("IXFORMER_DISABLE_SYNC", "0").strip().lower() == "1" and (self.up_weight_dtype != "int4" or self.down_weight_dtype != "int4"):
            raise NotImplementedError(f"IxformerQuantMoE: up_weight_dtype or down_weight_dtype is not 'int4', is not supported when disable_sync is True.")

        _attach_gating_bf16_buffers(self.gating)

        # Hardware constraints of the ixformer w8a8/w4a8 group-gemm kernels.
        if self.hidden_size % 64 != 0 or self.intermediate_size % 64 != 0:
            raise NotImplementedError(
                f"IxformerQuantMoE only supports hidden_size and intermediate_size divisible by 64, got {self.hidden_size} and {self.intermediate_size}."
            )
        if self.up_weight_dtype == torch.int8 and self.up_quant_group_size != -1:
            raise NotImplementedError(
                f"IxformerQuantMoE only supports up_weight_dtype='torch.int8' with up_quant_group_size=-1, got {self.up_weight_dtype} and {self.up_quant_group_size}."
            )
        if self.down_weight_dtype == torch.int8 and self.down_quant_group_size != -1:
            raise NotImplementedError(
                f"IxformerQuantMoE only supports down_weight_dtype='torch.int8' with down_quant_group_size=-1, got {self.down_weight_dtype} and {self.down_quant_group_size}."
            )
        if self.up_weight_dtype == "int4":
            if self.up_quant_group_size not in [128, 256, 320, 512]:
                raise NotImplementedError(
                    f"IxformerQuantMoE: up_weight_dtype is 'int4' and up_quant_group_size must be 128, 256, 320, or 512, got {self.up_weight_dtype} and {self.up_quant_group_size}."
                )
            if self.hidden_size % self.up_quant_group_size != 0:
                raise NotImplementedError(
                    f"IxformerQuantMoE: up_weight_dtype is 'int4' and k (hidden_size) must be divisible by up_quant_group_size, got hidden_size={self.hidden_size} and up_quant_group_size={self.up_quant_group_size}."
                )
            if self.intermediate_size * 2 < 256 or self.hidden_size < 256:
                raise NotImplementedError(
                    f"IxformerQuantMoE: up_weight_dtype is 'int4' and intermediate_size * 2 must be >= 256, hidden_size must be >= 256, got {self.hidden_size} and {self.intermediate_size}."
                )
        if self.down_weight_dtype == "int4":
            if self.down_quant_group_size not in [128, 256, 320, 512]:
                raise NotImplementedError(
                    f"IxformerQuantMoE: down_weight_dtype is 'int4' and down_quant_group_size must be 128, 256, 320, or 512, got {self.down_weight_dtype} and {self.down_quant_group_size}."
                )
            if self.intermediate_size % self.down_quant_group_size != 0:
                raise NotImplementedError(
                    f"IxformerQuantMoE: down_weight_dtype is 'int4' and k (intermediate_size) must be divisible by down_quant_group_size, got intermediate_size={self.intermediate_size} and down_quant_group_size={self.down_quant_group_size}."
                )
            if self.intermediate_size < 256 or self.hidden_size < 256:
                raise NotImplementedError(
                    f"IxformerQuantMoE: down_weight_dtype is 'int4' and intermediate_size must be >= 256, hidden_size must be >= 256, got {self.hidden_size} and {self.intermediate_size}."
                )

        # ixformer quant gemm kernels require fp32 weight scales, overriding the parent's default bf16 storage.
        setattr(self.experts.up_proj_weight_scale, "force_dtype", torch.float32)
        setattr(self.experts.down_proj_weight_scale, "force_dtype", torch.float32)
        # Convert checkpoint weights from TN layout to NN + tensor-core swizzle expected by the kernel.
        self.experts.register_load_state_dict_post_hook(_swizzle_weights_post_hook)
        self.output_dtype = torch.bfloat16

        self.gdr_device_buffer = torch.zeros([self.num_experts + 1], dtype=torch.int, device="cuda")
        torch.cuda.synchronize()
        self.gdr_buffer_ptr = ixf_f.new_gdr_buffer(self.gdr_device_buffer)
        self.disable_sync = os.environ.get("IXFORMER_DISABLE_SYNC", "0").strip().lower() == "1"

        # w4a8 kernel needs its own GDR scratch (per up/down) for async expert-routing metadata when sync is off.
        if self.up_weight_dtype == "int4" and self.down_weight_dtype == "int4":
            self.gdr_device_buffer1 = torch.zeros([8 * 1024 * 1024], dtype=torch.int8, device="cuda")
            self.gdr_device_buffer2 = torch.zeros([8 * 1024 * 1024], dtype=torch.int8, device="cuda")
            torch.cuda.synchronize()
            self.gdr_buffer_ptr1 = ixf_f.new_gdr_buffer(self.gdr_device_buffer1)
            self.gdr_buffer_ptr2 = ixf_f.new_gdr_buffer(self.gdr_device_buffer2)

    def __del__(self):
        if hasattr(self, "gdr_buffer_ptr"):
            ixf_f.delete_gdr_buffer(self.gdr_buffer_ptr)
        if hasattr(self, "gdr_buffer_ptr1"):
            ixf_f.delete_gdr_buffer(self.gdr_buffer_ptr1)
        if hasattr(self, "gdr_buffer_ptr2"):
            ixf_f.delete_gdr_buffer(self.gdr_buffer_ptr2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            enable_cuda_graph = True
        else:
            enable_cuda_graph = False

        if enable_cuda_graph and not self.disable_sync:
            raise RuntimeError("IxformerQuantMoE: CUDA graph capture requires disable_sync=True (set IXFORMER_DISABLE_SYNC=1).")

        if hidden_states.dtype == torch.float16:
            raise NotImplementedError(f"IxformerQuantMoE: hidden_states dtype must be 'torch.bfloat16', got {hidden_states.dtype}.")

        # DP input: gather peer ranks' shards so gating/dispatch see the full token set.
        # AG(h) is launched on torch.distributed's NCCL comm stream so it overlaps with
        # local gating; gating is per-token, so we run it on the LOCAL shard first and
        # all_gather the (tiny) routing tensors afterwards. dispatch then sees the
        # full hidden_states + full routing.
        if self.dp_input and self.ep_size > 1:
            local_tokens = hidden_states.shape[0]
            full = torch.empty(
                local_tokens * self.ep_size, hidden_states.shape[1],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            h_hdl = dist.all_gather_into_tensor(
                full, hidden_states.contiguous(), group=self.ep_group, async_op=True,
            )

            # gating on local hidden_states — runs on compute stream while AG(h) flies.
            local_gate_logits = ixf_f.triple_gemm_bf16_bf16_fp32(
                hidden_states,
                self.gating.gate_weight_bf16_tn_2,
                self.gating.gate_weight_bf16_tn_1,
                self.gating.gate_weight_bf16_tn_0,
            )
            local_top_k_gates, local_top_k_indices = ixf_f.moe_topk_softmax(
                local_gate_logits, self.gating.top_k, renormalize=True,
            )

            full_top_k_indices = torch.empty(
                local_tokens * self.ep_size, *local_top_k_indices.shape[1:],
                dtype=local_top_k_indices.dtype, device=local_top_k_indices.device,
            )
            idx_hdl = dist.all_gather_into_tensor(
                full_top_k_indices, local_top_k_indices.contiguous(),
                group=self.ep_group, async_op=True,
            )
            full_top_k_gates = torch.empty(
                local_tokens * self.ep_size, *local_top_k_gates.shape[1:],
                dtype=local_top_k_gates.dtype, device=local_top_k_gates.device,
            )
            gates_hdl = dist.all_gather_into_tensor(
                full_top_k_gates, local_top_k_gates.contiguous(),
                group=self.ep_group, async_op=True,
            )

            h_hdl.wait()
            idx_hdl.wait()
            gates_hdl.wait()

            hidden_states = full
            top_k_indices = full_top_k_indices
            top_k_gates = full_top_k_gates
        else:
            # triple_gemm uses 3 bf16 components of the fp32 gate weight to emulate fp32 matmul precision on bf16 HW.
            gate_logits = ixf_f.triple_gemm_bf16_bf16_fp32(
                hidden_states,
                self.gating.gate_weight_bf16_tn_2,
                self.gating.gate_weight_bf16_tn_1,
                self.gating.gate_weight_bf16_tn_0,
            )
            top_k_gates, top_k_indices = ixf_f.moe_topk_softmax(gate_logits, self.gating.top_k, renormalize=True)

        num_tokens, dim = hidden_states.shape

        need_gpu_size = self.disable_sync or enable_cuda_graph
        graph_safe_ep = self.ep_size > 1 and need_gpu_size

        if self.ep_size > 1:
            # _ep variant filters tokens to the local expert range [ep_start, ep_end);
            # expand_tokens is the actual local token count after filtering.
            (src_to_dst,
             sorted_token_ids,
             expert_sizes_gpu,
             expert_sizes_cpu,
             expand_tokens) = ixf_f.moe_compute_token_index_ep(
                top_k_indices,
                self.num_experts,
                self.ep_start,
                self.ep_end,
                gdr_buffer_ptr=None if need_gpu_size else self.gdr_buffer_ptr,
                graph_safe=graph_safe_ep,
            )
        else:
            (src_to_dst,
             sorted_token_ids,
             expert_sizes_gpu,
             expert_sizes_cpu) = ixf_f.moe_compute_token_index(
                top_k_indices,
                self.num_experts,
                gdr_buffer_ptr=None if need_gpu_size else self.gdr_buffer_ptr,
            )
            expand_tokens = num_tokens * self.top_k

        dispatch_tokens = num_tokens * self.top_k if graph_safe_ep else expand_tokens

        if sorted_token_ids.shape[0] == 0:
            combined = torch.zeros(
                num_tokens, self.hidden_size,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
        else:

            tokens_per_expert = expert_sizes_gpu if need_gpu_size else expert_sizes_cpu

            # Fuses expert-sort, expansion, smooth-quant scaling, and dynamic-int8 quantization into one kernel.
            i8_hs, quant_scale = ixf_f.moe_expand_input_dynamic_scaled_int8(
                hidden_states=hidden_states,
                dst_to_src=sorted_token_ids,
                dst_tokens=dispatch_tokens,
                topk=self.top_k,
                src_to_dst=src_to_dst,
                topk_ids=top_k_indices,
                smooth_scales=self.experts.up_proj_quantize.inv_smooth_scale,
                check_valid=graph_safe_ep,
            )
            i8_hs = i8_hs.view(-1, dim)

            # gemv variant takes GPU-side sizes so kernel shapes don't depend on CPU values during graph capture.
            if self.up_weight_dtype == torch.int8:
                if not need_gpu_size:
                    group_gemm_output1 = ixf_f.moe_w8a8_group_gemm(
                        input=i8_hs,
                        weight=self.experts.up_proj_weight,
                        i_scales=quant_scale,
                        w_scales=self.experts.up_proj_weight_scale,
                        output_dtype=self.output_dtype,
                        tokens_per_experts=tokens_per_expert,
                        format="NN",
                    )
                else:
                    group_gemm_output1 = ixf_f.moe_w8a8_group_gemv(
                        input=i8_hs,
                        weight=self.experts.up_proj_weight,
                        i_scales=quant_scale,
                        w_scales=self.experts.up_proj_weight_scale,
                        output_dtype=self.output_dtype,
                        tokens_per_experts=tokens_per_expert,
                        format=0,
                    )
            elif self.up_weight_dtype == "int4":
                group_gemm_output1 = torch.empty(
                    (i8_hs.shape[0], self.experts.up_proj_weight_scale.shape[-1]),
                    dtype=self.output_dtype,
                    device=i8_hs.device,
                )
                ixf_f.moe_w4a8_group_gemm(
                    input=i8_hs,
                    weight=self.experts.up_proj_weight,
                    i_scales=quant_scale,
                    w_scales=self.experts.up_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    format=0,
                    version=1,
                    group_size=self.up_quant_group_size,
                    gdr_buffer_ptr=self.gdr_buffer_ptr1 if not need_gpu_size else None,
                    output=group_gemm_output1,
                    enable_cuda_graph=need_gpu_size,
                )
            else:
                raise NotImplementedError(f"IxformerQuantMoE: up_weight_dtype must be 'torch.int8' or 'int4', got {self.up_weight_dtype}.")

            # Fuses swiglu activation, smooth-quant scaling, and dynamic-int8 quant ahead of the down projection.
            act_i8, act_scale = ixf_f.activation_dynamic_scaled_int8(
                input=group_gemm_output1,
                smooth_scales=self.experts.down_proj_quantize.inv_smooth_scale,
                dst_to_src=sorted_token_ids,
                topk_ids=top_k_indices,
                act_type="swiglu",
                check_valid=graph_safe_ep,
            )

            group_gemm_output2 = torch.empty(
                num_tokens * self.top_k, self.hidden_size,
                dtype=self.output_dtype, device=sorted_token_ids.device,
            )

            if self.down_weight_dtype == torch.int8:
                if not need_gpu_size:
                    ixf_f.moe_w8a8_group_gemm(
                        input=act_i8,
                        weight=self.experts.down_proj_weight,
                        i_scales=act_scale,
                        w_scales=self.experts.down_proj_weight_scale,
                        output_dtype=self.output_dtype,
                        tokens_per_experts=tokens_per_expert,
                        dst_to_src=sorted_token_ids,
                        format="NN",
                        output=group_gemm_output2,
                    )
                else:
                    ixf_f.moe_w8a8_group_gemv(
                        input=act_i8,
                        weight=self.experts.down_proj_weight,
                        i_scales=act_scale,
                        w_scales=self.experts.down_proj_weight_scale,
                        output_dtype=self.output_dtype,
                        tokens_per_experts=tokens_per_expert,
                        dst_to_src=sorted_token_ids,
                        format=0,
                        output=group_gemm_output2,
                    )
            elif self.down_weight_dtype == "int4":
                ixf_f.moe_w4a8_group_gemm(
                    input=act_i8,
                    weight=self.experts.down_proj_weight,
                    i_scales=act_scale,
                    w_scales=self.experts.down_proj_weight_scale,
                    output_dtype=self.output_dtype,
                    tokens_per_experts=tokens_per_expert,
                    dst_to_src=sorted_token_ids,
                    format=0,
                    version=1,
                    group_size=self.down_quant_group_size,
                    output=group_gemm_output2,
                    gdr_buffer_ptr=self.gdr_buffer_ptr2 if not need_gpu_size else None,
                    enable_cuda_graph=need_gpu_size,
                )

            else:
                raise NotImplementedError(f"IxformerQuantMoE: down_weight_dtype must be 'torch.int8' or 'int4', got {self.down_weight_dtype}.")

            expert_outputs = group_gemm_output2.view(-1, self.top_k, self.hidden_size)

            # src_to_dst == -1 marks padding slots that must not be summed.
            reduce_mask = src_to_dst == -1

            combined = ixf_f.moe_output_reduce_sum(
                input=expert_outputs,
                topk_weight=top_k_gates,
                mask=reduce_mask,
            )

        # Sum partial expert outputs across EP ranks; reduce_scatter slices the result back to the rank's DP shard.
        if self.ep_size > 1:
            if self.dp_input:
                local_combined = torch.empty(
                    combined.shape[0] // self.ep_size, combined.shape[1],
                    dtype=combined.dtype, device=combined.device,
                )
                ixfd.reduce_scatter_tensor(local_combined, combined.contiguous(), group=self.ep_group, async_op=True)
                combined = local_combined
            else:
                ixfd.all_reduce(combined, group=self.ep_group, async_op=True)

        return combined
