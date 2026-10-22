import os
import torch
import torch.distributed as dist
from typing import Union

from mojo_opset.core import MojoMoE
from mojo_opset.core import MojoQuantMoE

from ixformer import functions as ixf_f
import ixformer.distributed as ixfd


def _w4a8_ixmm_swizzle_columns(x: torch.Tensor) -> torch.Tensor:
    """ixmm's within-32-column interleave on the last dim.

    Forward of ixformer's ``_w4a8_ixmm_unswizzle_columns``, which gathers by the
    argsort of this map: ``out[..., c] = x[..., col_idx[c]]``.
    """
    n = x.shape[-1]
    j = torch.arange(32, device=x.device)
    perm = j // 2 + (j & 1) * 16
    col = torch.arange(n, device=x.device)
    col_idx = (col // 32) * 32 + perm[col % 32]
    return x.index_select(-1, col_idx)


def _w4a8_ixmm_swap_quadrants(x: torch.Tensor) -> torch.Tensor:
    """Exchange the 16x16 quadrants inside every 32x32 (k, n) block.

    Its own inverse, so this is also ixformer's ``_w4a8_ixmm_unswap_quadrants``.
    """
    *lead, k, n = x.shape
    blocks = x.reshape(*lead, k // 32, 32, n // 32, 32).clone()
    top_right = blocks[..., 0:16, :, 16:32].clone()
    bottom_left = blocks[..., 16:32, :, 0:16].clone()
    blocks[..., 0:16, :, 16:32] = bottom_left
    blocks[..., 16:32, :, 0:16] = top_right
    return blocks.reshape(*lead, k, n)


def _pack_w4a8_ixmm_weight(logical: torch.Tensor) -> torch.Tensor:
    """Logical signed int4 (E, k, n) -> the int8 (E, k, n // 2) buffer that
    moe_w4a8_group_gemm_ixmm reads.

    Forward of ``ixf_f.unpack_w4a8_ixmm_weight``. ixformer ships the inverse
    only, for its torch reference, so the pairing is kept explicit here rather
    than reaching into ixmm for its packer.
    """
    swizzled = _w4a8_ixmm_swizzle_columns(
        _w4a8_ixmm_swap_quadrants(logical.to(torch.int64))
    )
    low = (swizzled[..., 0::2] & 0xF).to(torch.uint8)
    high = (swizzled[..., 1::2] & 0xF).to(torch.uint8)
    return (low | (high << 4)).view(torch.int8)


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


def _repack_int4_nn_to_tn(packed_nn: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of :func:`_repack_int4_tn_to_nn`.

    Takes NN-packed int4 (E, K, N//2) used by the ixformer kernel and
    returns TN-packed int4 (E, N//2, K) matching the checkpoint layout.
    Used by the state_dict post-hook dual to keep state_dict round-trip
    safe.
    """
    device = packed_nn.device
    packed_nn = packed_nn.cuda()
    E = packed_nn.shape[0]

    # Unpack the i / i+16 pair packing.
    packed = packed_nn.view(E, K, N // 32, 16)
    out = torch.empty(E, K, N // 32, 32, dtype=torch.int8, device=packed.device)
    for i in range(16):
        byte = packed[:, :, :, i].to(torch.uint8)
        lo_nib = (byte & 0x0F).to(torch.int8)
        hi_nib = ((byte >> 4) & 0x0F).to(torch.int8)
        lo_nib = torch.where(lo_nib >= 8, lo_nib - 16, lo_nib)
        hi_nib = torch.where(hi_nib >= 8, hi_nib - 16, hi_nib)
        out[:, :, :, i] = lo_nib
        out[:, :, :, i + 16] = hi_nib
    out = out.view(E, K, N)

    # Inverse permute (the original permutation [0, 1, 5, 3, 4, 2, 6] is
    # self-inverse, so applying it again undoes the swizzle).
    out = out.view(E, K // 32, 2, 16, N // 32, 2, 16)
    out = out.permute(0, 1, 5, 3, 4, 2, 6).contiguous().view(E, K, N)

    # Transpose K <-> N back to checkpoint orientation.
    unpacked = out.transpose(-2, -1).contiguous()  # (E, N, K)

    # Pack pairs along N (low nibble = even index, high nibble = odd index).
    low = unpacked[:, 0::2, :]
    high = unpacked[:, 1::2, :]
    low_u8 = (low & 0x0F).to(torch.uint8)
    high_u8 = ((high & 0x0F) << 4).to(torch.uint8)
    packed_tn = (high_u8 | low_u8).to(torch.int8)

    return packed_tn.contiguous().to(device)


def _unpack_int4_tn(packed_tn: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """TN-packed int4 (E, N//2, K) -> logical signed int4 (E, K, N) as int8."""
    E = packed_tn.shape[0]
    u8 = packed_tn.to(torch.uint8)
    low = (u8 & 0x0F).to(torch.int8)
    high = ((u8 >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(E, N, K, dtype=torch.int8, device=packed_tn.device)
    unpacked[:, 0::2, :] = low
    unpacked[:, 1::2, :] = high
    return unpacked.transpose(-2, -1).contiguous()


def _pack_int4_tn(logical: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_unpack_int4_tn`: (E, K, N) -> (E, N//2, K)."""
    unpacked = logical.transpose(-2, -1).contiguous()
    low = (unpacked[:, 0::2, :] & 0x0F).to(torch.uint8)
    high = ((unpacked[:, 1::2, :] & 0x0F) << 4).to(torch.uint8)
    return (high | low).to(torch.int8)


def _repack_int4_tn_to_ixmm(packed_tn: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Checkpoint TN-packed int4 -> the buffer moe_w4a8_group_gemm_ixmm reads.

    ixmm pre-swizzles into a different layout from the ixinfer one that
    :func:`_repack_int4_tn_to_nn` builds, so the two packed buffers are not
    interchangeable. ``unpack_w4a8_ixmm_weight`` is the documented inverse.
    """
    device = packed_tn.device
    logical = _unpack_int4_tn(packed_tn.cuda(), N, K)
    return _pack_w4a8_ixmm_weight(logical).to(device)


def _repack_int4_ixmm_to_tn(packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of :func:`_repack_int4_tn_to_ixmm`, for the state_dict dual."""
    device = packed.device
    logical = ixf_f.unpack_w4a8_ixmm_weight(packed.cuda()).to(torch.int8)
    return _pack_int4_tn(logical).to(device)


def _ixmm_weight_scale(scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Checkpoint weight scale -> the shape the ixmm op validates.

    The grouped checkpoint is (E, n, k // group_size) against an op that wants
    (E, k // group_size, n). The per-channel one has no k axis to move: the
    checkpoint drops it, so (E, n) is already what the op takes.
    """
    if group_size <= 0:
        return scale.contiguous()
    return scale.permute(0, 2, 1).contiguous()


def _checkpoint_weight_scale(scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of :func:`_ixmm_weight_scale`; the swap is its own inverse."""
    return _ixmm_weight_scale(scale, group_size)


def _group_layout_from_expert_sizes(tokens_per_expert: torch.Tensor, device) -> torch.Tensor:
    """Per-expert [start, end) token ranges, the layout the ixmm op indexes with.

    start is the exclusive prefix sum of the expert sizes, which is exact here
    because the NN format packs expert rows back to back with no padding.
    """
    counts = tokens_per_expert.to(device=device, dtype=torch.int64)
    ends = torch.cumsum(counts, dim=0)
    return torch.stack((ends - counts, ends), dim=1).to(torch.int32)


def _swizzle_weights_post_hook(module, incompatible_keys):
    """load_state_dict post-hook: convert int4/int8 weights from TN (checkpoint) to NN (ixformer) format."""
    device = module.up_proj_weight.device
    module.up_proj_quantize.inv_smooth_scale = torch.nn.Parameter(module.up_proj_quantize.inv_smooth_scale.data.to(dtype=torch.bfloat16))

    # The up-projection smooth scale is indexed per expert, so it can normally only
    # be applied after routing. When every local expert carries the same row the
    # scale is effectively per-channel, and the EP expand can then smooth and
    # quantize once per token instead of once per expanded row. Decide it here:
    # the comparison is a host sync and must not happen inside forward.
    inv_s = module.up_proj_quantize.inv_smooth_scale.data
    module.up_smooth_per_channel = bool(
        inv_s.shape[0] > 0 and torch.equal(inv_s, inv_s[:1].expand_as(inv_s))
    )
    module.up_smooth_row = inv_s[0].contiguous() if module.up_smooth_per_channel else None


    if module.up_weight_dtype == "int4":
        N_up = module.intermediate_size * 2
        K_up = module.hidden_size
        up_nn = _repack_int4_tn_to_ixmm(module.up_proj_weight.data, N_up, K_up)
        up_scale_nn = _ixmm_weight_scale(module.up_proj_weight_scale.data, module.up_quant_group_size)
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
        down_nn = _repack_int4_tn_to_ixmm(module.down_proj_weight.data, N_down, K_down)
        down_scale_nn = _ixmm_weight_scale(module.down_proj_weight_scale.data, module.down_quant_group_size)
        module.register_buffer("down_proj_weight", down_nn.to(device))
        module.down_proj_weight_scale = torch.nn.Parameter(down_scale_nn.to(device=device, dtype=torch.float32))
    elif module.down_weight_dtype == torch.int8:
        down_nn = module.down_proj_weight.data.transpose(1, 2).contiguous()
        down_scale_nn = module.down_proj_weight_scale.data.contiguous()
        module.register_buffer("down_proj_weight", down_nn.to(device))
        module.down_proj_weight_scale = torch.nn.Parameter(down_scale_nn.to(device=device, dtype=torch.float32))


def _unswizzle_weights_state_dict_post_hook(module, state_dict, prefix, local_metadata):
    """Dual of :func:`_swizzle_weights_post_hook`.

    Reverses the runtime NN/swizzled layout back to checkpoint TN layout so
    state_dict() output can be re-loaded via load_state_dict() and have the
    forward post-hook re-swizzle to runtime form. Covers up/down proj weights
    and weight scales for both int4 and int8 paths. ``inv_smooth_scale``
    dtype cast (bf16) is idempotent on round-trip and needs no inverse.
    """
    up_key = prefix + "up_proj_weight"
    up_scale_key = prefix + "up_proj_weight_scale"
    down_key = prefix + "down_proj_weight"
    down_scale_key = prefix + "down_proj_weight_scale"

    if module.up_weight_dtype == "int4":
        N_up = module.intermediate_size * 2
        K_up = module.hidden_size
        if up_key in state_dict:
            state_dict[up_key] = _repack_int4_ixmm_to_tn(state_dict[up_key], N_up, K_up)
        if up_scale_key in state_dict:
            state_dict[up_scale_key] = _checkpoint_weight_scale(
                state_dict[up_scale_key], module.up_quant_group_size
            )
    elif module.up_weight_dtype == torch.int8:
        if up_key in state_dict:
            state_dict[up_key] = state_dict[up_key].transpose(1, 2).contiguous()
        # int8 up_proj_weight_scale: forward only .contiguous() + dtype cast,
        # both idempotent, no shape inverse needed.

    if module.down_weight_dtype == "int4":
        N_down = module.hidden_size
        K_down = module.intermediate_size
        if down_key in state_dict:
            state_dict[down_key] = _repack_int4_ixmm_to_tn(state_dict[down_key], N_down, K_down)
        if down_scale_key in state_dict:
            state_dict[down_scale_key] = _checkpoint_weight_scale(
                state_dict[down_scale_key], module.down_quant_group_size
            )
    elif module.down_weight_dtype == torch.int8:
        if down_key in state_dict:
            state_dict[down_key] = state_dict[down_key].transpose(1, 2).contiguous()


def _cast_gate_weight_post_hook(module, incompatible_keys):
    # fp32 (hidden, num_experts) -> bf16 (num_experts, hidden), the TN layout.
    module.gate_weight = torch.nn.Parameter(
        module.gate_weight.detach().to(torch.bfloat16).transpose(0, 1).contiguous(),
        requires_grad=module.gate_weight.requires_grad,
    )


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
        self.gating.register_load_state_dict_post_hook(_cast_gate_weight_post_hook)

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

        gate_logits = ixf_f.gemm_bf16_bf16_fp32(
            hidden_states,
            self.gating.gate_weight,
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

        self.gating.register_load_state_dict_post_hook(_cast_gate_weight_post_hook)

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
            if self.up_quant_group_size not in [-1,128, 256, 320, 512]:
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
            # ixmm packs two int4 output channels per byte, so n moves in 128s,
            # and k feeds the 64-wide mainloop.
            if (self.intermediate_size * 2) % 128 or self.hidden_size % 64:
                raise NotImplementedError(
                    f"IxformerQuantMoE: up_weight_dtype is 'int4' and ixmm needs n (intermediate_size * 2) % 128 == 0 and k (hidden_size) % 64 == 0, got intermediate_size={self.intermediate_size} and hidden_size={self.hidden_size}."
                )
        if self.down_weight_dtype == "int4":
            if self.down_quant_group_size not in [-1, 128, 256, 320, 512]:
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
            if self.hidden_size % 128 or self.intermediate_size % 64:
                raise NotImplementedError(
                    f"IxformerQuantMoE: down_weight_dtype is 'int4' and ixmm needs n (hidden_size) % 128 == 0 and k (intermediate_size) % 64 == 0, got hidden_size={self.hidden_size} and intermediate_size={self.intermediate_size}."
                )

        # ixformer quant gemm kernels require fp32 weight scales, overriding the parent's default bf16 storage.
        setattr(self.experts.up_proj_weight_scale, "force_dtype", torch.float32)
        setattr(self.experts.down_proj_weight_scale, "force_dtype", torch.float32)
        # Convert checkpoint weights from TN layout to NN + tensor-core swizzle expected by the kernel.
        self.experts.register_load_state_dict_post_hook(_swizzle_weights_post_hook)
        self.experts.register_state_dict_post_hook(_unswizzle_weights_state_dict_post_hook)
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
            local_gate_logits = ixf_f.gemm_bf16_bf16_fp32(
                hidden_states,
                self.gating.gate_weight,
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
            gate_logits = ixf_f.gemm_bf16_bf16_fp32(
                hidden_states,
                self.gating.gate_weight,
            )
            top_k_gates, top_k_indices = ixf_f.moe_topk_softmax(gate_logits, self.gating.top_k, renormalize=True)

        num_tokens, dim = hidden_states.shape

        need_gpu_size = self.disable_sync or enable_cuda_graph
        graph_safe_ep = self.ep_size > 1 and need_gpu_size
        needs_group_layout = "int4" in (self.up_weight_dtype, self.down_weight_dtype)

        if self.ep_size > 1:
            # _ep_packed filters tokens to the local expert range [ep_start, ep_end)
            # exactly like _ep -- src_to_expert_row is the classic slot -> expert-row
            # map, -1 on non-local slots -- but it additionally emits the token-major
            # packed metadata the reduce consumes, so the CSR, the per-row weights and
            # the group layout no longer get rebuilt here on every forward.
            #
            # dst_to_src, group_layout and packed_to_expert_row are out-params the
            # same routing kernel fills with no extra launch. It writes only
            # [0, expand_tokens) of the index tensors and never slices them, so the
            # -1 prefill that _ep did internally under graph_safe has to happen here:
            # the fixed-capacity consumers below identify padding rows by -1.
            sorted_token_ids = top_k_indices.new_empty(num_tokens * self.top_k)
            if graph_safe_ep:
                sorted_token_ids.fill_(-1)
            group_layout = (
                torch.empty((self.num_experts_local, 2), dtype=torch.int32, device=hidden_states.device)
                if needs_group_layout
                else None
            )
            packed_to_expert_row = top_k_indices.new_empty(num_tokens * self.top_k)
            (src_to_dst,
             _dst_to_packed,
             weights_packed,
             token_row_offsets,
             expert_sizes,
             expand_tokens) = ixf_f.moe_compute_token_index_ep_packed(
                topk_ids=top_k_indices.contiguous(),
                num_experts=self.num_experts,
                start_expert_id=self.ep_start,
                end_expert_id=self.ep_end,
                topk_weights=top_k_gates,
                dst_to_src=sorted_token_ids,
                group_layout=group_layout,
                packed_to_expert_row=packed_to_expert_row,
                graph_safe=graph_safe_ep,
            )
            # graph_safe already decides the residency of the size outputs, and in
            # this branch graph_safe_ep is need_gpu_size, so the one returned tensor
            # is whichever of the two tokens_per_expert goes on to select.
            expert_sizes_gpu = expert_sizes_cpu = expert_sizes
            # _ep handed back a dst_to_src already sliced down to the local token
            # count unless graph_safe asked for fixed capacity. The packed op always
            # returns full capacity, and the consumers below require numel to equal
            # their own row count, so reproduce the slice (a view, no copy).
            if not graph_safe_ep:
                sorted_token_ids = sorted_token_ids[:expand_tokens]
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

        # The fused smooth kernels index smooth_scales by expert id, but that
        # parameter only holds this rank's local experts while top_k_indices carries
        # global ids. The two coincide only at ep_start == 0, so every other rank was
        # reading past the end of smooth_scales. The routing op above still wants the
        # global ids, so shift into a separate tensor rather than in place.
        local_expert_ids = top_k_indices - self.ep_start if self.ep_start else top_k_indices

        # _ep_packed returns a full-capacity dst_to_src instead of one sliced down to
        # the local token count, so "nothing routed here" has to be read off
        # expand_tokens. Under graph_safe that count lives on device and must not be
        # inspected, which is fine: the path is fixed-capacity by construction.
        if not graph_safe_ep and expand_tokens == 0:
            combined = torch.zeros(
                num_tokens, self.hidden_size,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
        else:

            tokens_per_expert = expert_sizes_gpu if need_gpu_size else expert_sizes_cpu
            # Both w4a8 GEMMs read the same expert ranges. Under EP the routing op
            # already filled group_layout as an out-param, which also keeps the
            # layout off the host; without EP there is no such op, so build it here.
            if self.ep_size == 1:
                group_layout = (
                    _group_layout_from_expert_sizes(tokens_per_expert, hidden_states.device)
                    if needs_group_layout
                    else None
                )

            if self.ep_size > 1 and self.experts.up_smooth_per_channel:
                # Smooth and quantize once per token, then scatter int8 rows into
                # expert-major order. Each of a token's topk copies is the same row,
                # so per-expanded-row dynamic quant would compute the same int8
                # values and scale anyway; doing it first touches num_tokens rows
                # instead of dispatch_tokens and lets the scatter move 1 byte per
                # element instead of 2. Only valid when the smooth scale is
                # per-channel rather than per-expert, which the load hook decided.
                #
                # src_to_expert_row carries -1 on non-local slots and the scatter
                # skips them, so rows that no slot claims stay untouched. Nothing
                # downstream reads them: the group GEMMs are bounded by the expert
                # ranges and the reduce by token_row_offsets.
                i8_hidden, hs_scale = ixf_f.scaled_int8_quant(
                    hidden_states * self.experts.up_smooth_row
                )
                i8_hs = torch.empty((dispatch_tokens, dim), dtype=torch.int8, device=hidden_states.device)
                quant_scale = torch.empty((dispatch_tokens,), dtype=torch.float32, device=hidden_states.device)
                ixf_f.moe_expand_input_ep(
                    hidden_states=i8_hidden,
                    src_to_expert_row=src_to_dst,
                    expand_tokens=dispatch_tokens,
                    topk=self.top_k,
                    output=i8_hs,
                    input_scale=hs_scale,
                    output_scale=quant_scale,
                )
            else:
                # Per-expert smooth scale (or no EP): the scale is only known after
                # routing, so expert-sort, expansion, smooth scaling and dynamic
                # int8 quant all have to happen in the one fused kernel.
                i8_hs, quant_scale = ixf_f.moe_expand_input_dynamic_scaled_int8(
                    hidden_states=hidden_states,
                    dst_to_src=sorted_token_ids,
                    dst_tokens=dispatch_tokens,
                    topk=self.top_k,
                    src_to_dst=src_to_dst,
                    topk_ids=local_expert_ids,
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
                ixf_f.moe_w4a8_group_gemm_ixmm(
                    a=i8_hs,
                    weight=self.experts.up_proj_weight,
                    scale_a=quant_scale,
                    weight_scale=self.experts.up_proj_weight_scale,
                    group_layout=group_layout,
                    output=group_gemm_output1,
                    group_size=self.up_quant_group_size,
                )
            else:
                raise NotImplementedError(f"IxformerQuantMoE: up_weight_dtype must be 'torch.int8' or 'int4', got {self.up_weight_dtype}.")

            # Fuses swiglu activation, smooth-quant scaling, and dynamic-int8 quant
            # ahead of the down projection. The down smooth scale is per expert, but
            # these rows are already in expert-major order, so dst_to_src + topk_ids
            # resolve the expert per row and the scale survives the EP path intact.
            #
            # Under graph-safe EP the up GEMM wrote full capacity while only
            # [0, expand_tokens) carry routed rows, so bound the work on the
            # device-side count: no host sync, and the bound does not depend on the
            # -1 sentinel surviving in dst_to_src. check_valid stays on because
            # expand_tokens_gpu is documented to bound the writes, not the topk_ids
            # reads. Outside graph-safe the buffer is already exactly sized.
            act_i8, act_scale = ixf_f.activation_dynamic_scaled_int8(
                input=group_gemm_output1,
                smooth_scales=self.experts.down_proj_quantize.inv_smooth_scale,
                dst_to_src=sorted_token_ids,
                topk_ids=local_expert_ids,
                act_type="swiglu",
                check_valid=graph_safe_ep,
                expand_tokens_gpu=expand_tokens if graph_safe_ep else None,
            )

            group_gemm_output2 = torch.empty(
                num_tokens * self.top_k, self.hidden_size,
                dtype=self.output_dtype, device=sorted_token_ids.device,
            )
            # The int8 ops fold dst_to_src into their epilogue and land in token
            # order; moe_w4a8_group_gemm_ixmm has no such index and stays in
            # expert-major order, which the reduce below has to account for.
            expert_major = self.down_weight_dtype == "int4"

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
                # Row 0 is where the reduce below parks slots that carry no
                # routed row. The GEMM leaves it untouched when nothing routes
                # to a local expert, so it must not stay allocator garbage:
                # 0 * NaN would survive the weighted sum.
                group_gemm_output2[0].zero_()
                ixf_f.moe_w4a8_group_gemm_ixmm(
                    a=act_i8,
                    weight=self.experts.down_proj_weight,
                    scale_a=act_scale,
                    weight_scale=self.experts.down_proj_weight_scale,
                    group_layout=group_layout,
                    output=group_gemm_output2[: act_i8.shape[0]],
                    group_size=self.down_quant_group_size,
                )

            else:
                raise NotImplementedError(f"IxformerQuantMoE: down_weight_dtype must be 'torch.int8' or 'int4', got {self.down_weight_dtype}.")

            if expert_major:
                # Gather the expert-major rows inside the reduce instead of
                # scattering them to token order first: the scatter moved the
                # whole activation twice for nothing.
                if self.ep_size > 1:
                    # The routing op emitted the CSR, the packed-order weights and
                    # the packed -> expert-row map directly on device, so nothing is
                    # rebuilt here. The CSR spans only the rows that actually routed,
                    # so unrouted slots are absent rather than carried with a zero
                    # weight, and every shape still comes from host-side constants.
                    combined = ixf_f.moe_output_reduce_sum_ep(
                        input=group_gemm_output2,
                        token_row_offsets=token_row_offsets,
                        weights=weights_packed,
                        packed_to_expert_row=packed_to_expert_row,
                    )
                else:
                    # Non-EP routing has no packed metadata. Every slot owns one
                    # expert row there, so the CSR is a plain arange and packed row
                    # == slot; slots with no routed row are pointed at row 0 and
                    # neutralised by a zero weight, which keeps every shape
                    # independent of device data.
                    slots = src_to_dst.reshape(-1)
                    routed = slots >= 0
                    combined = ixf_f.moe_output_reduce_sum_ep(
                        input=group_gemm_output2,
                        token_row_offsets=torch.arange(
                            0, (num_tokens + 1) * self.top_k, self.top_k,
                            dtype=torch.int32, device=slots.device,
                        ),
                        weights=torch.where(routed, top_k_gates.reshape(-1), 0.0),
                        packed_to_expert_row=torch.where(routed, slots, 0),
                    )
            else:
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
