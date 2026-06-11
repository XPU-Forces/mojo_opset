import os
from typing import Union

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh

from mojo_opset import MojoExperts
from mojo_opset import MojoMoE
from mojo_opset import MojoMoECombine
from mojo_opset import MojoMoEDispatch
from mojo_opset import MojoMoEGating
from mojo_opset import MojoQuantMoE
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _get_world_size():
    # When launched via torchrun, WORLD_SIZE is set; otherwise fall back to single-rank
    # (EP=1, no torch.distributed init) so the file is also runnable via plain pytest.
    return int(os.environ.get("WORLD_SIZE", "1"))


def _set_current_device(device_type: str, world_size: int) -> str:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "npu":
        n_devices = torch.npu.device_count()
        if n_devices < world_size:
            pytest.skip(
                f"Not enough NPU devices: need {world_size}, but only {n_devices} visible. "
                f"Set ASCEND_RT_VISIBLE_DEVICES to expose more devices "
                f"(e.g. ASCEND_RT_VISIBLE_DEVICES=0,1)."
            )
        torch.npu.set_device(local_rank)
        return f"npu:{local_rank}"
    if device_type == "mlu":
        n_devices = torch.mlu.device_count()
        if n_devices < world_size:
            pytest.skip(f"Not enough MLU devices: need {world_size}, but only {n_devices} visible.")
        torch.mlu.set_device(local_rank)
        return f"mlu:{local_rank}"
    if device_type == "cuda":
        n_devices = torch.cuda.device_count()
        if n_devices < world_size:
            pytest.skip(f"Not enough CUDA devices: need {world_size}, but only {n_devices} visible.")
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    raise ValueError(f"Unsupported device type for distributed test: {device_type}")


# Copied from mojo_opset/tests/accuracy/operators/test_moe_quant.py — produces
# quantization-consistent weights/scales so a fused-kernel backend (e.g. ixformer)
# matches the torch reference under forward_diff_with(mixed_tol=True).
def _pack_int4_to_int8_along_output(input: torch.Tensor) -> torch.Tensor:
    input_u8 = input.to(torch.uint8)
    packed = ((input_u8[..., 1::2, :] & 0x0F) << 4) | (input_u8[..., 0::2, :] & 0x0F)
    return packed.to(torch.int8)


def _quantize_weight_per_group(weight: torch.Tensor, quant_group_size: int, weight_dtype):
    if quant_group_size > 0:
        weight_groups = weight.float().split(quant_group_size, dim=-1)
    else:
        weight_groups = [weight]
    scales, quantizeds = [], []
    for weight_group in weight_groups:
        scale = (weight_group.abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-12)
        quantizeds.append(torch.clamp(torch.round(weight_group / scale), -8, 7).to(torch.int8))
        scales.append(scale)
    quantized = torch.cat(quantizeds, dim=-1)
    scale = torch.cat(scales, dim=-1)
    if quant_group_size <= 0:
        scale = scale.squeeze(-1)
    return _pack_int4_to_int8_along_output(quantized) if weight_dtype == "int4" else quantized, scale


def _make_quant_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    up_quant_group_size: int,
    up_weight_dtype: Union[torch.dtype, str],
    down_quant_group_size: int,
    down_weight_dtype: Union[torch.dtype, str],
):
    up_weight_fp = torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=torch.float32) * 0.01
    down_weight_fp = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float32) * 0.01
    up_weight, up_weight_scale = _quantize_weight_per_group(up_weight_fp, up_quant_group_size, up_weight_dtype)
    down_weight, down_weight_scale = _quantize_weight_per_group(down_weight_fp, down_quant_group_size, down_weight_dtype)
    return up_weight, up_weight_scale.bfloat16(), down_weight, down_weight_scale.bfloat16()


def _broadcast_state(state, src: int = 0):
    """In-place broadcast of every persistent tensor in `state` from `src` to all ranks,
    so the ref module is byte-identical across ranks. No-op when torch.distributed isn't initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return
    for k in sorted(state.keys()):
        dist.broadcast(state[k], src=src)


def _slice_state_for_ep(full_state, ep_start, ep_end, expert_dim0_keys):
    """Return a state_dict for an EP rank: expert-indexed tensors are sliced along dim 0,
    everything else (notably gating weights) is replicated."""
    sliced = {}
    for k, v in full_state.items():
        if k in expert_dim0_keys:
            sliced[k] = v[ep_start:ep_end].clone().contiguous()
        else:
            sliced[k] = v.clone()
    return sliced


def _replicated_input(num_tokens: int, hidden_size: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    # Generate on CPU under a fixed seed so every rank receives an identical tensor,
    # independent of the per-rank device RNG state.
    cpu_gen = torch.Generator(device="cpu").manual_seed(42)
    return torch.randn(num_tokens, hidden_size, generator=cpu_gen, dtype=dtype).to(device)


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, intermediate_size, num_tokens",
    [
        (16, 4, 1024, 2048, 64),
        (32, 8, 1024, 4096, 128),
    ],
)
@pytest.mark.parametrize("dp_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_moe_ep(num_experts, top_k, hidden_size, intermediate_size, num_tokens, dp_input, dtype):
    world_size = _get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    device_type = get_torch_device()
    device = _set_current_device(device_type, world_size)
    if world_size > 1:
        init_device_mesh(device_type, (world_size,))

    if num_experts % world_size != 0:
        pytest.skip(f"num_experts={num_experts} not divisible by world_size={world_size}.")

    # Build full weights directly as tensors on rank 0, then broadcast so the full
    # state is byte-identical across ranks.
    torch.manual_seed(0)
    full_state = {
        "gating.gate_weight": (torch.randn(hidden_size, num_experts, dtype=torch.float32, device=device) * 0.02),
        "experts.up_proj_weight": (
            torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=dtype, device=device) * 0.02
        ),
        "experts.down_proj_weight": (
            torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device) * 0.02
        ),
    }
    _broadcast_state(full_state, src=0)

    expert_dim0_keys = {"experts.up_proj_weight", "experts.down_proj_weight"}

    # Torch-backend EP reference.
    ref = MojoMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=None,
        dp_input=dp_input,
    ).to(dtype).to(device)
    ref.gating.gate_weight.data = ref.gating.gate_weight.data.float()
    ref_state = _slice_state_for_ep(full_state, ref.ep_start, ref.ep_end, expert_dim0_keys)
    ref.load_state_dict(ref_state)

    # Active-backend EP MoE — same expert split.
    moe = MojoMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=None,
        dp_input=dp_input,
    ).to(dtype).to(device)
    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()
    moe_state = _slice_state_for_ep(full_state, moe.ep_start, moe.ep_end, expert_dim0_keys)
    moe.load_state_dict(moe_state)

    x = _replicated_input(num_tokens, hidden_size, dtype, device)
    if dp_input:
        # Pad up to a multiple of world_size, then take this rank's contiguous slice.
        pad = (-num_tokens) % world_size
        if pad:
            x = torch.cat([x, x.new_zeros(pad, hidden_size)], dim=0)
        tokens_per_rank = x.shape[0] // world_size
        x = x[rank * tokens_per_rank : (rank + 1) * tokens_per_rank].contiguous()
    moe.forward_diff_with(ref, x, mixed_tol=True)


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, intermediate_size, num_tokens",
    [
        (16, 2, 512, 1280, 33),
        (24, 4, 512, 1280, 97),
    ],
)
@pytest.mark.parametrize(
    "up_weight_dtype,up_quant_group_size,down_weight_dtype,down_quant_group_size",
    [
        ("int4", 512, "int4", 320),
        (torch.int8, -1, torch.int8, -1),
    ],
)
@pytest.mark.parametrize("dp_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_quant_moe_ep(
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    num_tokens,
    up_weight_dtype,
    up_quant_group_size,
    down_weight_dtype,
    down_quant_group_size,
    dp_input,
    dtype,
):
    world_size = _get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    device_type = get_torch_device()
    device = _set_current_device(device_type, world_size)
    if world_size > 1:
        init_device_mesh(device_type, (world_size,))

    if num_experts % world_size != 0:
        pytest.skip(f"num_experts={num_experts} not divisible by world_size={world_size}.")

    # Build quant-consistent state on rank 0 (so int8 weights match their fp scales),
    # then broadcast every tensor so every rank's `ref` is byte-identical.
    torch.manual_seed(0)
    gate_weight = torch.randn(hidden_size, num_experts, dtype=torch.float32, device=device) * 0.2
    fc1_input_smooth_scale = torch.rand(num_experts, hidden_size, dtype=torch.float32, device=device) + 0.5
    fc2_input_smooth_scale = torch.rand(num_experts, intermediate_size, dtype=torch.float32, device=device) + 0.5
    up_weight, up_weight_scale, down_weight, down_weight_scale = _make_quant_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        up_quant_group_size,
        up_weight_dtype,
        down_quant_group_size,
        down_weight_dtype,
    )

    full_state = {
        "gating.gate_weight": gate_weight,
        "experts.up_proj_weight": up_weight.to(device),
        "experts.down_proj_weight": down_weight.to(device),
        "experts.up_proj_weight_scale": up_weight_scale.to(device),
        "experts.down_proj_weight_scale": down_weight_scale.to(device),
        "experts.up_proj_quantize.inv_smooth_scale": (1.0 / fc1_input_smooth_scale).to(device),
        "experts.down_proj_quantize.inv_smooth_scale": (1.0 / fc2_input_smooth_scale).to(device),
    }
    _broadcast_state(full_state, src=0)

    ref = MojoQuantMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_dtype=torch.int8,
        up_quant_group_size=up_quant_group_size,
        up_weight_dtype=up_weight_dtype,
        down_quant_group_size=down_quant_group_size,
        down_weight_dtype=down_weight_dtype,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=None,
        dp_input=dp_input,
    ).to(device)

    # Active-backend EP QuantMoE — same expert split.
    moe = MojoQuantMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quant_dtype=torch.int8,
        up_quant_group_size=up_quant_group_size,
        up_weight_dtype=up_weight_dtype,
        down_quant_group_size=down_quant_group_size,
        down_weight_dtype=down_weight_dtype,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=None,
        dp_input=dp_input,
    ).to(device)

    expert_dim0_keys = {
        "experts.up_proj_weight",
        "experts.down_proj_weight",
        "experts.up_proj_weight_scale",
        "experts.down_proj_weight_scale",
        "experts.up_proj_quantize.inv_smooth_scale",
        "experts.down_proj_quantize.inv_smooth_scale",
    }
    ref.load_state_dict(_slice_state_for_ep(full_state, ref.ep_start, ref.ep_end, expert_dim0_keys))
    moe.load_state_dict(_slice_state_for_ep(full_state, moe.ep_start, moe.ep_end, expert_dim0_keys))

    x = _replicated_input(num_tokens, hidden_size, dtype, device)
    if dp_input:
        # Pad up to a multiple of world_size, then take this rank's contiguous slice.
        pad = (-num_tokens) % world_size
        if pad:
            x = torch.cat([x, x.new_zeros(pad, hidden_size)], dim=0)
        tokens_per_rank = x.shape[0] // world_size
        x = x[rank * tokens_per_rank : (rank + 1) * tokens_per_rank].contiguous()
    moe.forward_diff_with(ref, x, mixed_tol=True)


class _SmallOpMoEModule(nn.Module):
    """Plain nn.Module assembled from MoE small operators.

    Mirrors MojoMoE's interface and parameter layout (gating.gate_weight,
    experts.up_proj_weight, experts.down_proj_weight) but is itself a regular
    nn.Module rather than a MojoOperator. The four small operators are picked
    from their default registry (active backend), so this validates that the
    backend's small ops compose correctly into a full MoE.
    """

    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group=None,
        dp_input: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        self.dp_input = dp_input

        base = num_experts // ep_size
        rem = num_experts % ep_size
        self.num_experts_local = base + 1 if ep_rank < rem else base
        self.ep_start = base * ep_rank + min(ep_rank, rem)
        self.ep_end = self.ep_start + self.num_experts_local

        self.gating = MojoMoEGating(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
        self.dispatch = MojoMoEDispatch(num_experts=num_experts)
        self.experts = MojoExperts(
            num_experts=self.num_experts_local,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.combine = MojoMoECombine(multiply_by_gates=True)

    def forward(self, hidden_states):
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
            hidden_states, top_k_gates, top_k_indices,
        )

        if self.ep_size > 1:
            cumsum = tokens_per_expert.cumsum(0)
            tok_start = 0 if self.ep_start == 0 else cumsum[self.ep_start - 1].item()
            tok_end = cumsum[self.ep_end - 1].item()
            local_sorted = sorted_hidden_states[tok_start:tok_end]
            local_tokens_per_expert = tokens_per_expert[self.ep_start:self.ep_end]
            local_expert_outputs = self.experts(local_sorted, local_tokens_per_expert)
            # ixformer's moe_combine kernel requires expert_outputs of shape
            # [num_tokens * top_k, hidden]. Pad non-local positions with zeros
            # so they contribute nothing to the per-token reduction.
            expert_outputs = local_expert_outputs.new_zeros(
                sorted_hidden_states.shape[0], local_expert_outputs.shape[-1]
            )
            expert_outputs[tok_start:tok_end] = local_expert_outputs
        else:
            expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        output_buffer = torch.zeros_like(hidden_states, memory_format=torch.contiguous_format)
        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)

        if self.ep_size > 1:
            if self.dp_input:
                local_combined = torch.empty(
                    combined.shape[0] // self.ep_size, *combined.shape[1:],
                    dtype=combined.dtype, device=combined.device,
                )
                dist.reduce_scatter_tensor(
                    local_combined, combined.contiguous(),
                    op=dist.ReduceOp.SUM, group=self.ep_group,
                )
                combined = local_combined
            else:
                dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=self.ep_group)

        return combined


@pytest.mark.parametrize(
    "num_experts, top_k, hidden_size, intermediate_size, num_tokens",
    [
        (16, 4, 1024, 2048, 64),
        (32, 8, 1024, 4096, 128),
    ],
)
@pytest.mark.parametrize("dp_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@bypass_not_implemented
def test_small_op_moe_vs_mojo_torch(num_experts, top_k, hidden_size, intermediate_size, num_tokens, dp_input, dtype):
    """A small-op-composed plain nn.Module must match MojoMoE(backend='torch') under EP=1 and EP>1."""
    world_size = _get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    device_type = get_torch_device()
    device = _set_current_device(device_type, world_size)
    if world_size > 1:
        init_device_mesh(device_type, (world_size,))

    if num_experts % world_size != 0:
        pytest.skip(f"num_experts={num_experts} not divisible by world_size={world_size}.")

    torch.manual_seed(0)
    full_state = {
        "gating.gate_weight": (torch.randn(hidden_size, num_experts, dtype=torch.float32, device=device) * 0.02),
        "experts.up_proj_weight": (
            torch.randn(num_experts, intermediate_size * 2, hidden_size, dtype=dtype, device=device) * 0.02
        ),
        "experts.down_proj_weight": (
            torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device) * 0.02
        ),
    }
    _broadcast_state(full_state, src=0)

    expert_dim0_keys = {"experts.up_proj_weight", "experts.down_proj_weight"}

    # Reference: MojoMoE with backend='torch'.
    ref = MojoMoE._registry.get("torch")(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=None,
        dp_input=dp_input,
    ).to(dtype).to(device)
    ref.gating.gate_weight.data = ref.gating.gate_weight.data.float()
    ref.load_state_dict(_slice_state_for_ep(full_state, ref.ep_start, ref.ep_end, expert_dim0_keys))

    # Active-backend small ops composed in a plain nn.Module.
    moe = _SmallOpMoEModule(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=None,
        dp_input=dp_input,
    ).to(dtype).to(device)
    moe.gating.gate_weight.data = moe.gating.gate_weight.data.float()
    moe.load_state_dict(_slice_state_for_ep(full_state, moe.ep_start, moe.ep_end, expert_dim0_keys))

    x = _replicated_input(num_tokens, hidden_size, dtype, device)
    if dp_input:
        pad = (-num_tokens) % world_size
        if pad:
            x = torch.cat([x, x.new_zeros(pad, hidden_size)], dim=0)
        tokens_per_rank = x.shape[0] // world_size
        x = x[rank * tokens_per_rank : (rank + 1) * tokens_per_rank].contiguous()

    out = moe(x.clone())
    out_ref = ref(x.clone())
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=2e-2)
