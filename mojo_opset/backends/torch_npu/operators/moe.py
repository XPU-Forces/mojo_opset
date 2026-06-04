"""
Copyright (c) 2026 Bytedance. All Rights Reserved.
"""

import logging
from typing import Sequence

import torch
import torch.nn.functional as F
import torch_npu

from mojo_opset.core import MojoExperts
from mojo_opset.core import MojoMoE
from mojo_opset.core import MojoMoECombine
from mojo_opset.core import MojoMoEDispatch
from mojo_opset.core import MojoMoEGating
from mojo_opset.core import MojoQuantMoE

from .mxfp8 import _mx_e8m0_dtype
from .mxfp8 import mx_dequant_weight
from .mxfp8 import mx_dynamic_quant_activation
from .mxfp8 import prepare_mx_expert_scale_for_grouped_matmul

logger = logging.getLogger(__name__)

MAX_NPU_GROUPED_MATMUL_EXPERT_GROUPS = 128


def _tokens_per_expert_to_group_list_cumsum(tokens_per_expert: torch.Tensor, total_tokens: int) -> list[int]:
    counts = [max(0, int(x)) for x in tokens_per_expert.reshape(-1).tolist()]
    group_list = []
    acc = 0
    for count in counts:
        acc += count
        group_list.append(acc)
    if sum(counts) != total_tokens:
        raise ValueError(f"tokens_per_expert sum {sum(counts)} != chunk rows {total_tokens}")
    if group_list and group_list[-1] != total_tokens:
        raise ValueError(f"group_list tail {group_list[-1]} != chunk rows {total_tokens}")
    if len(group_list) < 2:
        raise ValueError(f"npu_grouped_matmul needs len(group_list)>=2 in chunk, got {len(group_list)}")
    return group_list


def _npu_grouped_matmul_chunk(
    x: torch.Tensor,
    weights: Sequence[torch.Tensor],
    tokens_per_expert_chunk: torch.Tensor,
) -> torch.Tensor:
    if len(weights) > MAX_NPU_GROUPED_MATMUL_EXPERT_GROUPS:
        raise ValueError(f"chunk has {len(weights)} weights > {MAX_NPU_GROUPED_MATMUL_EXPERT_GROUPS}")
    total_tokens = int(x.shape[0])
    if total_tokens == 0:
        raise ValueError("empty input to npu_grouped_matmul_chunk")
    return torch_npu.npu_grouped_matmul(
        [x],
        list(weights),
        group_list=_tokens_per_expert_to_group_list_cumsum(tokens_per_expert_chunk, total_tokens),
        group_type=0,
        split_item=2,
        group_list_type=0,
    )[0]


def _single_expert_ffn(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate_up = F.linear(x.float(), up_weight.float())
    activated = torch_npu.npu_swiglu(gate_up, dim=-1)
    return F.linear(activated, down_weight.float())


def _experts_swiglu_ffn_torch_loop(
    sorted_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    counts = tokens_per_expert.reshape(-1).to("cpu", dtype=torch.int64)
    outputs = []
    row = 0
    for expert_idx, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        x_e = sorted_hidden_states[row : row + count].float()
        row += count
        gate_up = F.linear(x_e, up_proj_weight[expert_idx].float())
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        outputs.append(F.linear(activated, down_proj_weight[expert_idx].float()))
    if row != int(tokens_per_expert.sum().item()):
        raise RuntimeError(f"MoE row walk mismatch: consumed {row} != {int(tokens_per_expert.sum().item())}")
    if not outputs:
        return sorted_hidden_states
    return torch.cat(outputs, dim=0).to(dtype=sorted_hidden_states.dtype)


def _npu_experts_swiglu_ffn_chunked(
    sorted_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    counts = tokens_per_expert.reshape(-1)
    num_experts = int(counts.numel())
    total_tokens = int(sorted_hidden_states.shape[0])
    if total_tokens == 0:
        return sorted_hidden_states

    outputs = []
    row = 0
    dtype = sorted_hidden_states.dtype

    for chunk_start in range(0, num_experts, MAX_NPU_GROUPED_MATMUL_EXPERT_GROUPS):
        chunk_end = min(chunk_start + MAX_NPU_GROUPED_MATMUL_EXPERT_GROUPS, num_experts)
        chunk_counts = counts[chunk_start:chunk_end]
        chunk_tokens = int(chunk_counts.sum().item())
        if chunk_tokens == 0:
            continue

        x_chunk = sorted_hidden_states[row : row + chunk_tokens]
        row += chunk_tokens
        n_local = chunk_end - chunk_start

        if n_local == 1:
            out_chunk = _single_expert_ffn(
                x_chunk,
                up_proj_weight[chunk_start],
                down_proj_weight[chunk_start],
            )
        else:
            fc1_w = [up_proj_weight[i].t().contiguous() for i in range(chunk_start, chunk_end)]
            gate_up = _npu_grouped_matmul_chunk(x_chunk, fc1_w, chunk_counts)
            activated = torch_npu.npu_swiglu(gate_up, dim=-1)
            fc2_w = [down_proj_weight[i].t().contiguous() for i in range(chunk_start, chunk_end)]
            out_chunk = _npu_grouped_matmul_chunk(activated, fc2_w, chunk_counts)

        outputs.append(out_chunk.to(dtype=dtype))

    if row != total_tokens:
        raise RuntimeError(f"MoE row walk mismatch: consumed {row} != total {total_tokens}")
    if not outputs:
        return sorted_hidden_states
    return torch.cat(outputs, dim=0)


def _swiglu_mx_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch_npu.npu_swiglu_mx_quant(
        x,
        axis=-1,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        scale_alg=0,
        max_dtype_value=0.0,
    )
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError(f"unexpected npu_swiglu_mx_quant return: type={type(out)}")
    return out[0], out[1]


def _expert_count_group_list(tokens_per_expert_chunk: torch.Tensor) -> torch.Tensor:
    return tokens_per_expert_chunk.reshape(-1).to(dtype=torch.int64).contiguous()


def _mx_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_list: torch.Tensor,
    per_token_scale: torch.Tensor | None = None,
    split_item: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    e8 = _mx_e8m0_dtype()
    kwargs = {
        "scale": [scale],
        "group_list": group_list,
        "group_type": 0,
        "split_item": split_item,
        "group_list_type": 1,
        "scale_dtype": e8,
        "output_dtype": output_dtype,
    }
    if per_token_scale is not None:
        kwargs["per_token_scale"] = [per_token_scale]
        kwargs["per_token_scale_dtype"] = e8
    return torch_npu.npu_grouped_matmul([x], weight=[weight], **kwargs)[0]


def _npu_experts_swiglu_ffn_mxfp8_chunked(
    sorted_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    up_proj_weight_fp8: torch.Tensor,
    up_proj_weight_scale: torch.Tensor,
    down_proj_weight_fp8: torch.Tensor,
    down_proj_weight_scale: torch.Tensor,
) -> torch.Tensor:
    counts = tokens_per_expert.reshape(-1)
    num_experts = int(counts.numel())
    total_tokens = int(sorted_hidden_states.shape[0])
    if total_tokens == 0:
        return sorted_hidden_states

    hidden_size = int(sorted_hidden_states.shape[-1])
    outputs = []
    row = 0
    out_dtype = sorted_hidden_states.dtype
    use_native_gmm = True

    for chunk_start in range(0, num_experts, MAX_NPU_GROUPED_MATMUL_EXPERT_GROUPS):
        chunk_end = min(chunk_start + MAX_NPU_GROUPED_MATMUL_EXPERT_GROUPS, num_experts)
        chunk_counts = counts[chunk_start:chunk_end]
        chunk_tokens = int(chunk_counts.sum().item())
        if chunk_tokens == 0:
            continue

        x_chunk = sorted_hidden_states[row : row + chunk_tokens]
        row += chunk_tokens

        if use_native_gmm:
            try:
                up_w = up_proj_weight_fp8[chunk_start:chunk_end]
                up_s = up_proj_weight_scale[chunk_start:chunk_end]
                inter_size = int(up_w.shape[1]) // 2
                x_fp8, x_scale = mx_dynamic_quant_activation(x_chunk)
                gate_up = _mx_grouped_matmul(
                    x_fp8,
                    up_w.transpose(1, 2),
                    prepare_mx_expert_scale_for_grouped_matmul(up_s, in_features=hidden_size),
                    group_list=_expert_count_group_list(chunk_counts),
                    per_token_scale=x_scale,
                    split_item=3,
                    output_dtype=out_dtype,
                )
                gate_h, up_h = gate_up[..., :inter_size], gate_up[..., inter_size:]
                gate_up = torch.cat((up_h, gate_h), dim=-1)
                act_fp8, act_scale = _swiglu_mx_quant(gate_up)
                down_w = down_proj_weight_fp8[chunk_start:chunk_end]
                down_s = down_proj_weight_scale[chunk_start:chunk_end]
                out_chunk = _mx_grouped_matmul(
                    act_fp8,
                    down_w.transpose(1, 2),
                    prepare_mx_expert_scale_for_grouped_matmul(down_s, in_features=inter_size),
                    group_list=_expert_count_group_list(chunk_counts),
                    per_token_scale=act_scale,
                    split_item=2,
                    output_dtype=out_dtype,
                )
                outputs.append(out_chunk.to(dtype=out_dtype))
                continue
            except Exception as exc:
                logger.warning("MXFP8 npu_grouped_matmul path failed (%s); fallback to dequant+BF16 MoE", exc)
                use_native_gmm = False

        up_bf16 = mx_dequant_weight(
            up_proj_weight_fp8[chunk_start:chunk_end],
            up_proj_weight_scale[chunk_start:chunk_end],
            out_dtype=out_dtype,
        )
        down_bf16 = mx_dequant_weight(
            down_proj_weight_fp8[chunk_start:chunk_end],
            down_proj_weight_scale[chunk_start:chunk_end],
            out_dtype=out_dtype,
        )
        try:
            outputs.append(_npu_experts_swiglu_ffn_chunked(x_chunk, chunk_counts, up_bf16, down_bf16))
        except Exception as exc:
            logger.warning("BF16 dequant MoE path failed (%s); fallback to torch MoE", exc)
            outputs.append(_experts_swiglu_ffn_torch_loop(x_chunk, chunk_counts, up_bf16, down_bf16))

    if row != total_tokens:
        raise RuntimeError(f"MoE row walk mismatch: consumed {row} != total {total_tokens}")
    if not outputs:
        return sorted_hidden_states
    return torch.cat(outputs, dim=0)


class TorchNpuMoEGating(MojoMoEGating, default_priority=0):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_states: torch.Tensor):
        if hidden_states.device.type != "npu":
            return super().forward(hidden_states)

        gate_logits = torch.matmul(hidden_states.float(), self.gate_weight)
        try:
            topk_weights, topk_indices, _ = torch_npu.npu_moe_gating_top_k_softmax(gate_logits, k=self.top_k)
            topk_gates = topk_weights / torch.sum(topk_weights, dim=-1, keepdim=True).clamp(min=1e-20)
            return topk_indices.to(torch.int64), topk_gates.float()
        except Exception as exc:
            logger.warning("TorchNpuMoEGating: npu_moe_gating_top_k_softmax failed (%s), fallback torch topk", exc)
            return super().forward(hidden_states)


class TorchNpuMoEDispatch(MojoMoEDispatch, default_priority=0):
    """No dedicated torch_npu kernel; keep torch dispatch semantics under the torch_npu MoE route."""

    supported_platforms_list = ["npu"]


class TorchNpuExperts(MojoExperts, default_priority=0):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if sorted_hidden_states.device.type != "npu":
            return super().forward(sorted_hidden_states, tokens_per_expert)

        try:
            if (
                self.up_proj_weight.dtype == torch.float8_e4m3fn
                and hasattr(self, "up_proj_weight_mx_scale")
                and hasattr(self, "down_proj_weight_mx_scale")
            ):
                return _npu_experts_swiglu_ffn_mxfp8_chunked(
                    sorted_hidden_states,
                    tokens_per_expert,
                    self.up_proj_weight,
                    self.up_proj_weight_mx_scale,
                    self.down_proj_weight,
                    self.down_proj_weight_mx_scale,
                )
            return _npu_experts_swiglu_ffn_chunked(
                sorted_hidden_states,
                tokens_per_expert,
                self.up_proj_weight,
                self.down_proj_weight,
            )
        except Exception as exc:
            logger.warning("TorchNpuExperts: npu_grouped_matmul/npu_swiglu failed (%s), fallback torch loop", exc)
            return super().forward(sorted_hidden_states, tokens_per_expert)


class TorchNpuMoECombine(MojoMoECombine, default_priority=0):
    """No dedicated torch_npu kernel; keep torch combine semantics under the torch_npu MoE route."""

    supported_platforms_list = ["npu"]


class TorchNpuMoE(MojoMoE, default_priority=0):
    """Composite MoE shell so gating/experts resolve to torch_npu implementations."""

    supported_platforms_list = ["npu"]


class TorchNpuQuantMoE(MojoQuantMoE, default_priority=0):
    """Composite quant MoE shell; quant experts may be provided by external torch_npu extensions."""

    supported_platforms_list = ["npu"]
