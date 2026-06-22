import torch
import torch_npu

from mojo_opset.core import MojoFormatCast
from mojo_opset.core import MojoFunctionalDequantSwiGLUQuant
from mojo_opset.core import MojoGroupedMatmul
from mojo_opset.core import MojoMoEDistributeCombineV2
from mojo_opset.core import MojoMoEDistributeDispatchV2
from mojo_opset.core import MojoMoEFinalizeRouting
from mojo_opset.core import MojoMoEInitRoutingV2
from mojo_opset.core import MojoMoEReRouting
from mojo_opset.core import MojoQuantMatmul


class TorchNpuQuantMatmul(MojoQuantMatmul):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        pertoken_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        return torch_npu.npu_quant_matmul(
            x,
            weight,
            scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=output_dtype,
        )


class TorchNpuFunctionalDequantSwiGLUQuant(MojoFunctionalDequantSwiGLUQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        *,
        weight_scale: torch.Tensor,
        quant_scale: torch.Tensor,
        activation_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        quant_offset: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        activate_left: bool = False,
        quant_mode: int = 1,
        swiglu_mode: int = 0,
        clamp_limit: float | None = None,
        glu_alpha: float = 1.0,
        glu_bias: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_clamp_kernel = clamp_limit is not None or swiglu_mode != 0 or glu_alpha != 1.0 or glu_bias != 0.0
        if use_clamp_kernel:
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


class TorchNpuGroupedMatmul(MojoGroupedMatmul):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        inputs,
        weights,
        *,
        scale=None,
        per_token_scale=None,
        group_list=None,
        split_item: int = 3,
        output_dtype: torch.dtype = torch.bfloat16,
        group_type: int = 0,
        group_list_type: int = 1,
        tuning_config=None,
    ):
        return torch_npu.npu_grouped_matmul(
            list(inputs),
            list(weights),
            scale=scale,
            per_token_scale=per_token_scale,
            group_list=group_list,
            split_item=split_item,
            output_dtype=output_dtype,
            group_type=group_type,
            group_list_type=group_list_type,
            tuning_config=[0] if tuning_config is None else tuning_config,
        )


class TorchNpuFormatCast(MojoFormatCast):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor, acl_format: int) -> torch.Tensor:
        return torch_npu.npu_format_cast(x, acl_format)


class TorchNpuMoEInitRoutingV2(MojoMoEInitRoutingV2):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        *,
        expert_idx: torch.Tensor,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range=None,
        quant_mode: int = 1,
        scale: torch.Tensor | None = None,
    ):
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


class TorchNpuMoEReRouting(MojoMoEReRouting):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert_group: torch.Tensor,
        *,
        per_token_scales: torch.Tensor | None = None,
    ):
        return torch_npu.npu_moe_re_routing(
            x,
            tokens_per_expert_group,
            per_token_scales=per_token_scales,
        )


class TorchNpuMoEFinalizeRouting(MojoMoEFinalizeRouting):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        *,
        skip1: torch.Tensor | None = None,
        skip2: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        expanded_src_to_dst_row: torch.Tensor | None = None,
        export_for_source_row: torch.Tensor | None = None,
        drop_pad_mode: int = 2,
    ) -> torch.Tensor:
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


class TorchNpuMoEDistributeDispatchV2(MojoMoEDistributeDispatchV2):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        *,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        x_active_mask: torch.Tensor | None = None,
        expert_shard_type: int = 0,
        shared_expert_rank_num: int = 0,
        moe_expert_num: int = 0,
        global_bs: int = 0,
        scales: torch.Tensor | None = None,
        quant_mode: int = 0,
        group_ep: str | None = None,
        ep_world_size: int = 1,
        ep_rank_id: int = 0,
        group_tp: str | None = None,
        tp_world_size: int = 1,
        tp_rank_id: int = 0,
    ):
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


class TorchNpuMoEDistributeCombineV2(MojoMoEDistributeCombineV2):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        *,
        expand_x: torch.Tensor,
        shared_expert_x: torch.Tensor | None,
        expert_ids: torch.Tensor,
        assist_info_for_combine: torch.Tensor,
        expert_scales: torch.Tensor,
        x_active_mask: torch.Tensor | None = None,
        expert_shard_type: int = 0,
        shared_expert_rank_num: int = 0,
        moe_expert_num: int = 0,
        global_bs: int = 0,
        group_ep: str | None = None,
        ep_world_size: int = 1,
        ep_rank_id: int = 0,
        group_tp: str | None = None,
        tp_world_size: int = 1,
        tp_rank_id: int = 0,
        ep_send_counts: torch.Tensor | None = None,
        tp_send_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
