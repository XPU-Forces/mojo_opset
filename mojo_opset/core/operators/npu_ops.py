from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence

import torch
import torch.nn.functional as F

from ..operator import MojoOperator


def _maybe_import_torch_npu():
    try:
        import torch_npu

        return torch_npu
    except Exception:
        return None


def _as_group_counts(group_list: torch.Tensor | Sequence[int], group_list_type: int) -> list[int]:
    if isinstance(group_list, torch.Tensor):
        counts = group_list.to(dtype=torch.int64).cpu().tolist()
    else:
        counts = [int(x) for x in group_list]
    if group_list_type == 0:
        prev = 0
        raw_counts = counts
        counts = []
        for value in raw_counts:
            counts.append(int(value - prev))
            prev = int(value)
    return counts


def _repeat_by_group(param: torch.Tensor, group_counts: list[int]) -> torch.Tensor:
    if param.dim() == 1:
        return param.float().view(1, -1).expand(sum(group_counts), -1)
    if param.dim() != 2 or param.shape[0] != len(group_counts):
        raise ValueError(
            f"grouped param must be 2D with first dim {len(group_counts)}, got shape {tuple(param.shape)}"
        )
    return param.float().repeat_interleave(torch.tensor(group_counts, dtype=torch.int64), dim=0)


class MojoQuantMatmul(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        pertoken_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_quant_matmul"):
            return torch_npu.npu_quant_matmul(
                x,
                weight,
                scale,
                pertoken_scale=pertoken_scale,
                bias=bias,
                output_dtype=output_dtype,
            )

        out = torch.matmul(x.float(), weight.float())
        if bias is not None:
            out = out + bias.float()
        if output_dtype == torch.int32:
            return out.round().to(torch.int32)
        out = out * scale.float().reshape(1, -1)
        if pertoken_scale is not None:
            out = out * pertoken_scale.float().reshape(-1, 1)
        return out.to(output_dtype)


class MojoFunctionalDequantSwiGLUQuant(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        *,
        weight_scale: torch.Tensor,
        quant_scale: torch.Tensor,
        activation_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
        group_index: Optional[torch.Tensor] = None,
        activate_left: bool = False,
        quant_mode: int = 1,
        swiglu_mode: int = 0,
        clamp_limit: Optional[float] = None,
        glu_alpha: float = 1.0,
        glu_bias: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if quant_offset is not None:
            raise NotImplementedError("quant_offset is not supported.")
        if quant_mode != 1:
            raise NotImplementedError(f"Only quant_mode=1 is supported, got {quant_mode}.")

        torch_npu = _maybe_import_torch_npu()
        use_clamp_kernel = clamp_limit is not None or swiglu_mode != 0 or glu_alpha != 1.0 or glu_bias != 0.0
        if torch_npu is not None:
            if use_clamp_kernel and hasattr(torch_npu, "npu_dequant_swiglu_clamp_quant"):
                return torch_npu.npu_dequant_swiglu_clamp_quant(
                    x,
                    weight_scale=weight_scale,
                    quant_scale=quant_scale,
                    activation_scale=activation_scale,
                    bias=bias,
                    quant_offset=quant_offset,
                    group_index=group_index,
                    activate_left=activate_left,
                    quant_mode=quant_mode,
                    swiglu_mode=swiglu_mode,
                    clamp_limit=clamp_limit,
                    glu_alpha=glu_alpha,
                    glu_bias=glu_bias,
                )
            if not use_clamp_kernel and hasattr(torch_npu, "npu_dequant_swiglu_quant"):
                return torch_npu.npu_dequant_swiglu_quant(
                    x,
                    weight_scale=weight_scale,
                    quant_scale=quant_scale,
                    activation_scale=activation_scale,
                    bias=bias,
                    quant_offset=quant_offset,
                    group_index=group_index,
                    activate_left=activate_left,
                    quant_mode=quant_mode,
                )

        x_fp = x.float()
        if group_index is not None:
            group_counts = _as_group_counts(group_index, 1)
            x_fp = x_fp * _repeat_by_group(weight_scale, group_counts)
        else:
            x_fp = x_fp * weight_scale.float().view(1, -1)
        if activation_scale is not None:
            x_fp = x_fp * activation_scale.float().reshape(-1, 1)
        if bias is not None:
            x_fp = x_fp + bias.float()

        left, right = x_fp.chunk(2, dim=-1)
        if activate_left:
            out_fp = F.silu(left * glu_alpha + glu_bias) * right
        else:
            out_fp = F.silu(right * glu_alpha + glu_bias) * left
        if clamp_limit is not None:
            out_fp = out_fp.clamp(min=-float(clamp_limit), max=float(clamp_limit))

        if group_index is not None:
            group_counts = _as_group_counts(group_index, 1)
            out_fp = out_fp * _repeat_by_group(quant_scale, group_counts)
        else:
            out_fp = out_fp * quant_scale.float().view(1, -1)

        scale = out_fp.abs().amax(dim=-1).clamp(min=1e-12) / 127.0
        output = torch.clamp(torch.round(out_fp / scale.unsqueeze(-1)), -128, 127).to(torch.int8)
        return output, scale


class MojoGroupedMatmul(MojoOperator):
    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        weights: Sequence[torch.Tensor],
        *,
        scale: Optional[Sequence[torch.Tensor]] = None,
        per_token_scale: Optional[Sequence[torch.Tensor]] = None,
        group_list: torch.Tensor | Sequence[int],
        split_item: int = 3,
        output_dtype: torch.dtype = torch.bfloat16,
        group_type: int = 0,
        group_list_type: int = 1,
        tuning_config: Optional[Sequence[Any]] = None,
    ) -> list[torch.Tensor]:
        del split_item, group_type, tuning_config
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_grouped_matmul"):
            return torch_npu.npu_grouped_matmul(
                list(inputs),
                list(weights),
                scale=scale,
                per_token_scale=per_token_scale,
                group_list=group_list,
                split_item=3,
                output_dtype=output_dtype,
                group_type=0,
                group_list_type=group_list_type,
                tuning_config=[0] if tuning_config is None else tuning_config,
            )

        counts = _as_group_counts(group_list, group_list_type)
        outputs: list[torch.Tensor] = []
        scale_list = list(scale) if scale is not None else [None] * len(inputs)
        per_token_scale_list = list(per_token_scale) if per_token_scale is not None else [None] * len(inputs)

        for input_tensor, weight_tensor, weight_scale, token_scale in zip(
            inputs, weights, scale_list, per_token_scale_list
        ):
            if weight_tensor.dim() == 3:
                current_outputs = []
                start = 0
                for expert_idx, count in enumerate(counts):
                    end = start + count
                    x_slice = input_tensor[start:end]
                    w_slice = weight_tensor[expert_idx]
                    out = torch.matmul(x_slice.float(), w_slice.float())
                    if output_dtype != torch.int32:
                        if weight_scale is not None:
                            out = out * weight_scale[expert_idx].float().view(1, -1)
                        if token_scale is not None:
                            out = out * token_scale[start:end].float().reshape(-1, 1)
                    current_outputs.append(out.to(output_dtype))
                    start = end
                outputs.append(
                    torch.cat(current_outputs, dim=0) if current_outputs else input_tensor.new_empty((0, weight_tensor.shape[-1]))
                )
            else:
                out = torch.matmul(input_tensor.float(), weight_tensor.float())
                if output_dtype != torch.int32:
                    if weight_scale is not None:
                        out = out * weight_scale.float().view(1, -1)
                    if token_scale is not None:
                        out = out * token_scale.float().reshape(-1, 1)
                outputs.append(out.to(output_dtype))
        return outputs


class MojoFormatCast(MojoOperator):
    def forward(self, x: torch.Tensor, acl_format: int) -> torch.Tensor:
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_format_cast"):
            return torch_npu.npu_format_cast(x, acl_format)
        return x.contiguous()


def _expand_moe_routes(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    expert_num: int,
    smooth_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hidden_size = x.shape
    top_k = expert_ids.shape[1]
    flat_experts = expert_ids.reshape(-1).to(dtype=torch.int64)
    expanded_x = x.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
    route_indices = torch.arange(flat_experts.numel(), device=x.device, dtype=torch.int64)
    tokens_per_expert = torch.bincount(flat_experts, minlength=expert_num).to(dtype=torch.int32)

    if smooth_scale is not None:
        expanded_scale = smooth_scale.index_select(0, flat_experts).float()
        scaled_x = expanded_x.float() * expanded_scale
        pertoken_scale = scaled_x.abs().amax(dim=-1).clamp(min=1e-12) / 127.0
        expanded_x = torch.clamp(torch.round(scaled_x / pertoken_scale.unsqueeze(-1)), -128, 127).to(torch.int8)
    else:
        pertoken_scale = torch.ones(route_indices.shape[0], dtype=torch.float32, device=x.device)

    perm = torch.argsort(flat_experts, stable=True)
    return expanded_x[perm], route_indices[perm].to(torch.int32), tokens_per_expert, pertoken_scale[perm]


class MojoMoEInitRoutingV2(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        *,
        expert_idx: torch.Tensor,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range: Optional[Sequence[int]] = None,
        quant_mode: int = 1,
        scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del active_num, expert_tokens_num_type, expert_tokens_num_flag, active_expert_range, quant_mode
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_moe_init_routing_v2"):
            return torch_npu.npu_moe_init_routing_v2(
                x,
                expert_idx=expert_idx,
                active_num=active_num,
                expert_num=expert_num,
                expert_tokens_num_type=expert_tokens_num_type,
                expert_tokens_num_flag=expert_tokens_num_flag,
                active_expert_range=active_expert_range,
                quant_mode=quant_mode,
                scale=scale,
            )
        return _expand_moe_routes(x, expert_idx, expert_num=expert_num, smooth_scale=scale)


class MojoMoEReRouting(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert_group: torch.Tensor,
        *,
        per_token_scales: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_moe_re_routing"):
            return torch_npu.npu_moe_re_routing(
                x,
                tokens_per_expert_group,
                per_token_scales=per_token_scales,
            )

        tokens_per_local_expert = tokens_per_expert_group.sum(dim=0).to(dtype=torch.int32)
        gathered_ids_unsort = torch.arange(x.shape[0], device=x.device, dtype=torch.int32)
        return x, per_token_scales, gathered_ids_unsort, tokens_per_local_expert


class MojoMoEFinalizeRouting(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        *,
        skip1: Optional[torch.Tensor] = None,
        skip2: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        scales: torch.Tensor,
        expanded_src_to_dst_row: torch.Tensor,
        export_for_source_row: Optional[torch.Tensor] = None,
        drop_pad_mode: int = 2,
    ) -> torch.Tensor:
        del export_for_source_row, drop_pad_mode
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_moe_finalize_routing"):
            return torch_npu.npu_moe_finalize_routing(
                x,
                skip1=skip1,
                skip2=skip2,
                bias=bias,
                scales=scales,
                expanded_src_to_dst_row=expanded_src_to_dst_row,
                export_for_source_row=export_for_source_row,
                drop_pad_mode=drop_pad_mode,
            )

        top_k = scales.shape[1]
        route_order = expanded_src_to_dst_row.to(dtype=torch.int64)
        token_indices = torch.div(route_order, top_k, rounding_mode="floor")
        route_scales = scales.reshape(-1).index_select(0, route_order).float()
        out = torch.zeros(
            (scales.shape[0], x.shape[-1]),
            dtype=torch.float32,
            device=x.device,
        )
        contrib = x.float() * route_scales.unsqueeze(-1)
        out.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(contrib), contrib)
        if skip1 is not None:
            out = out + skip1.float()
        if skip2 is not None:
            out = out + skip2.float()
        if bias is not None:
            out = out + bias.float()
        return out.to(x.dtype)


class MojoMoEDistributeDispatchV2(MojoOperator):
    def forward(
        self,
        *,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        x_active_mask: Optional[torch.Tensor] = None,
        expert_shard_type: int = 0,
        shared_expert_rank_num: int = 0,
        moe_expert_num: int = 0,
        global_bs: int = 0,
        scales: Optional[torch.Tensor] = None,
        quant_mode: int = 0,
        group_ep: Optional[str] = None,
        ep_world_size: int = 1,
        ep_rank_id: int = 0,
        group_tp: Optional[str] = None,
        tp_world_size: int = 1,
        tp_rank_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del x_active_mask, expert_shard_type, shared_expert_rank_num, global_bs
        del quant_mode, group_ep, ep_world_size, ep_rank_id, group_tp, tp_world_size, tp_rank_id
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            return torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                x_active_mask=x_active_mask,
                expert_shard_type=expert_shard_type,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                global_bs=global_bs,
                scales=scales,
                quant_mode=quant_mode,
                group_ep=group_ep,
                ep_world_size=ep_world_size,
                ep_rank_id=ep_rank_id,
                group_tp=group_tp,
                tp_world_size=tp_world_size,
                tp_rank_id=tp_rank_id,
            )
        return _expand_moe_routes(x, expert_ids, expert_num=moe_expert_num, smooth_scale=scales)


class MojoMoEDistributeCombineV2(MojoOperator):
    def forward(
        self,
        *,
        expand_x: torch.Tensor,
        shared_expert_x: Optional[torch.Tensor],
        expert_ids: torch.Tensor,
        assist_info_for_combine: torch.Tensor,
        expert_scales: torch.Tensor,
        x_active_mask: Optional[torch.Tensor] = None,
        expert_shard_type: int = 0,
        shared_expert_rank_num: int = 0,
        moe_expert_num: int = 0,
        global_bs: int = 0,
        group_ep: Optional[str] = None,
        ep_world_size: int = 1,
        ep_rank_id: int = 0,
        group_tp: Optional[str] = None,
        tp_world_size: int = 1,
        tp_rank_id: int = 0,
        ep_send_counts: Optional[torch.Tensor] = None,
        tp_send_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del x_active_mask, expert_shard_type, shared_expert_rank_num, moe_expert_num, global_bs
        del group_ep, ep_world_size, ep_rank_id, group_tp, tp_world_size, tp_rank_id
        del ep_send_counts, tp_send_counts
        torch_npu = _maybe_import_torch_npu()
        if torch_npu is not None and hasattr(torch_npu, "npu_moe_distribute_combine_v2"):
            return torch_npu.npu_moe_distribute_combine_v2(
                expand_x=expand_x,
                shared_expert_x=shared_expert_x,
                expert_ids=expert_ids,
                assist_info_for_combine=assist_info_for_combine,
                expert_scales=expert_scales,
                x_active_mask=x_active_mask,
                expert_shard_type=expert_shard_type,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                global_bs=global_bs,
                group_ep=group_ep,
                ep_world_size=ep_world_size,
                ep_rank_id=ep_rank_id,
                group_tp=group_tp,
                tp_world_size=tp_world_size,
                tp_rank_id=tp_rank_id,
                ep_send_counts=ep_send_counts,
                tp_send_counts=tp_send_counts,
            )

        top_k = expert_ids.shape[1]
        route_order = assist_info_for_combine.to(dtype=torch.int64)
        token_indices = torch.div(route_order, top_k, rounding_mode="floor")
        route_scales = expert_scales.reshape(-1).index_select(0, route_order).float()
        out = torch.zeros(
            (expert_scales.shape[0], expand_x.shape[-1]),
            dtype=torch.float32,
            device=expand_x.device,
        )
        contrib = expand_x.float() * route_scales.unsqueeze(-1)
        out.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(contrib), contrib)
        if shared_expert_x is not None:
            out = out + shared_expert_x.float()
        return out.to(expand_x.dtype)
