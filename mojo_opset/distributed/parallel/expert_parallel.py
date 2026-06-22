from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc

from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Shard

from mojo_opset.core.operators.moe import MojoMoE
from mojo_opset.core.operators.moe import MojoMoECombine
from mojo_opset.core.operators.moe import MojoMoEDispatch
from mojo_opset.core.operators.moe import MojoQuantMoE
from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle
from mojo_opset.distributed.parallel.utils import shard_tensor
from mojo_opset.distributed.parallel.utils import stat_dict_rename_hook


# DeepSeek V4 MoE-EP runtime helpers.
#
# The model keeps local expert computation methods and operator instances; this
# module owns the EP routing, communication, and MC2 dispatch/combine orchestration.
def build_deepseek_v4_moe_mc2_kwargs(moe):
    global_rank = dist.get_rank()
    moe_ep_group_name = moe.hccl_comm_dict.get("moe_ep_group_mc2_name", None)
    quant_mode = moe.dispatch_quant_mode.get(moe.gmm_quant_mode, moe.dispatch_quant_mode["w16a16"])
    enable_smooth_scale = quant_mode == moe.dispatch_quant_mode["w8a8int8"]
    dispatch_kwargs = {
        "x_active_mask": None,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe.n_routed_experts,
        "global_bs": 0,
        "scales": moe.smooth_scale_1 if enable_smooth_scale else None,
        "quant_mode": quant_mode,
        "group_ep": moe_ep_group_name,
        "ep_world_size": moe.ep_size,
        "ep_rank_id": global_rank,
        "group_tp": moe_ep_group_name,
        "tp_world_size": 1,
        "tp_rank_id": 0,
    }
    combine_kwargs = {
        "x_active_mask": None,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe.n_routed_experts,
        "global_bs": 0,
        "group_ep": moe_ep_group_name,
        "ep_world_size": moe.ep_size,
        "ep_rank_id": global_rank,
        "group_tp": moe_ep_group_name,
        "tp_world_size": 1,
        "tp_rank_id": 0,
    }
    return dispatch_kwargs, combine_kwargs


def _build_deepseek_v4_moe_token_splits(tokens_per_expert_group, tokens_per_expert, ep_size: int):
    """Build all-to-all split lists for DeepSeek V4 prefill MoE-EP routing."""

    combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
    combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
    all_tokens = combine_tokens[0].sum()
    output_splits, input_splits = combine_tokens.cpu().tolist()
    return all_tokens, input_splits, output_splits


def deepseek_v4_moe_infer_ep_prefill(
    moe,
    hidden_states_flat,
    topk_idx,
    topk_weight,
    *,
    shared_expert_out=None,
    shared_expert_event=None,
):
    """Run DeepSeek V4 prefill MoE-EP with double all-to-all routing."""

    moe_ep_group = moe.ep_group
    n_tokens = hidden_states_flat.shape[0]

    expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = moe.moe_init_routing_v2(
        hidden_states_flat,
        expert_idx=topk_idx,
        active_num=n_tokens * moe.top_k,
        expert_num=moe.n_routed_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, moe.n_routed_experts],
        quant_mode=1,
        scale=moe.smooth_scale_1,
    )

    tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
    dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)

    all_tokens, input_splits, output_splits = _build_deepseek_v4_moe_token_splits(
        tokens_per_expert_group,
        tokens_per_expert,
        moe.ep_size,
    )

    gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
    dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)

    gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
    dist.all_to_all_single(gathered_pertoken_scale, pertoken_scale, output_splits, input_splits, group=moe_ep_group)

    hidden_states_ordered, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = (
        moe.moe_re_routing(
            gathered_tokens,
            tokens_per_expert_group.view(moe.ep_size, -1),
            per_token_scales=gathered_pertoken_scale,
        )
    )

    expert_out = moe.forward_expert_gmm(
        hidden_states_ordered,
        tokens_per_local_expert,
        pertoken_scale=gathered_pertoken_scale,
        group_list_type=1,
    )

    new_x = torch.index_select(expert_out, 0, gathered_ids_unsort.float().argsort().int())

    combined_tokens = new_x.new_empty(expanded_x.shape[0], new_x.shape[1])
    dist.all_to_all_single(combined_tokens, new_x, input_splits, output_splits, group=moe_ep_group)

    moe._wait_shared_expert(shared_expert_out, shared_expert_event)
    return moe.moe_finalize_routing(
        combined_tokens,
        skip1=shared_expert_out,
        skip2=None,
        bias=None,
        scales=topk_weight.to(combined_tokens.dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=None,
        drop_pad_mode=2,
    )


def deepseek_v4_moe_infer_ep_decode(
    moe,
    hidden_states_flat,
    topk_idx,
    topk_weight,
    *,
    shared_expert_out=None,
    shared_expert_event=None,
):
    """Run DeepSeek V4 decode MoE-EP with MC2 distribute dispatch/combine."""

    if moe.dispatch_kwargs is None:
        moe.set_mc2_kwargs()

    dispatch_output = moe.moe_distribute_dispatch_v2(
        x=hidden_states_flat,
        expert_ids=topk_idx,
        **moe.dispatch_kwargs,
    )
    expand_x, dynamic_scale, expand_idx, expert_token_num = dispatch_output[:4]
    ep_recv_counts = dispatch_output[4] if len(dispatch_output) > 4 else None
    tp_recv_counts = dispatch_output[5] if len(dispatch_output) > 5 else None

    expert_out = moe.forward_expert_gmm(
        expand_x,
        expert_token_num,
        pertoken_scale=dynamic_scale,
        group_list_type=1,
    )

    moe._wait_shared_expert(shared_expert_out, shared_expert_event)
    combine_input = {
        "expand_x": expert_out,
        "shared_expert_x": shared_expert_out,
        "expert_ids": topk_idx,
        "assist_info_for_combine": expand_idx,
        "expert_scales": topk_weight.to(torch.float32),
    }
    if ep_recv_counts is not None:
        combine_input["ep_send_counts"] = ep_recv_counts
    if tp_recv_counts is not None:
        combine_input["tp_send_counts"] = tp_recv_counts

    return moe.moe_distribute_combine_v2(
        **combine_input,
        **moe.combine_kwargs,
    )


def gather_deepseek_v4_moe_smooth_scale_1(model, layer_iter):
    if getattr(model, "ep_size", 1) <= 1 or not dist.is_initialized():
        return
    moe_ep_group = getattr(model, "hccl_comm_dict", {}).get("moe_ep_group")
    if moe_ep_group is None:
        return
    for layer in layer_iter:
        mlp = layer.mlp
        if not hasattr(mlp, "smooth_scale_1") or mlp.smooth_scale_1 is None:
            continue
        all_smooth_scale_1 = mlp.smooth_scale_1.data.new_empty(
            mlp.smooth_scale_1.data.shape[0] * model.ep_size,
            mlp.smooth_scale_1.data.shape[1],
        )
        dist.all_gather_into_tensor(all_smooth_scale_1, mlp.smooth_scale_1.data, group=moe_ep_group)
        mlp.smooth_scale_1.data = all_smooth_scale_1


class _EPDispatchWrapper(nn.Module):
    """Wraps MojoMoEDispatch to slice dispatch output to a local expert partition."""

    def __init__(self, dispatch: nn.Module, ep_mesh: DeviceMesh):
        super().__init__()
        assert isinstance(dispatch, MojoMoEDispatch)
        self._dispatch = dispatch
        ep_size = ep_mesh.size()
        ep_rank = ep_mesh.get_local_rank()
        base = dispatch.num_experts // ep_size
        rem = dispatch.num_experts % ep_size
        local = base + 1 if ep_rank < rem else base
        self.ep_start = base * ep_rank + min(ep_rank, rem)
        self.ep_end = self.ep_start + local

    def forward(self, hidden_states, top_k_gates, top_k_indices):
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self._dispatch(
            hidden_states, top_k_gates, top_k_indices
        )
        cumsum = tokens_per_expert.cumsum(0)
        tok_start = 0 if self.ep_start == 0 else cumsum[self.ep_start - 1].item()
        tok_end = cumsum[self.ep_end - 1].item()
        return (
            sorted_hidden_states[tok_start:tok_end],
            tokens_per_expert[self.ep_start : self.ep_end],
            sorted_gates[tok_start:tok_end],
            token_indices[tok_start:tok_end],
        )


class _EPCombineWrapper(nn.Module):
    """Wraps MojoMoECombine to combine partial output of each local expert partition."""

    def __init__(self, combine: nn.Module, ep_mesh: DeviceMesh):
        super().__init__()
        assert isinstance(combine, MojoMoECombine)
        self._combine = combine
        self.ep_mesh = ep_mesh

    def forward(self, output_buffer, expert_outputs, sorted_gates, token_indices):
        output = self._combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return fc.all_reduce(output, "sum", (self.ep_mesh, 0))
        return output


def _ep_partition_fn(src_data_rank, name, module, device_mesh):
    del name
    from mojo_opset.core.operators.moe import MojoExperts
    from mojo_opset.core.operators.moe import MojoMoE
    from mojo_opset.core.operators.moe import MojoQuantExperts
    from mojo_opset.core.operators.moe import MojoQuantMoE

    if isinstance(module, (MojoMoE, MojoQuantMoE)):
        module.dispatch = _EPDispatchWrapper(module.dispatch, device_mesh)
        module.combine = _EPCombineWrapper(module.combine, device_mesh)

    elif isinstance(module, MojoExperts):
        module.register_parameter(
            "up_proj_weight",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_weight)),
        )
        module.register_parameter(
            "down_proj_weight",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_weight)),
        )
        module.register_state_dict_post_hook(
            partial(stat_dict_rename_hook, ("up_proj_weight", "down_proj_weight"), device_mesh)
        )
    elif isinstance(module, MojoQuantExperts):
        module.register_buffer(
            "up_proj_weight",
            shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_weight),
        )
        module.register_buffer(
            "down_proj_weight",
            shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_weight),
        )
        module.register_parameter(
            "up_proj_weight_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_weight_scale)),
        )
        module.register_parameter(
            "down_proj_weight_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_weight_scale)),
        )
        module.up_proj_quantize.register_parameter(
            "inv_smooth_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_quantize.inv_smooth_scale)),
        )
        module.down_proj_quantize.register_parameter(
            "inv_smooth_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_quantize.inv_smooth_scale)),
        )
        module.register_state_dict_post_hook(
            partial(
                stat_dict_rename_hook,
                (
                    "up_proj_weight",
                    "down_proj_weight",
                    "up_proj_weight_scale",
                    "down_proj_weight_scale",
                    "up_proj_quantize.inv_smooth_scale",
                    "down_proj_quantize.inv_smooth_scale",
                ),
                device_mesh,
            )
        )


class MojoExpertParallel(MojoRegisterableParallelStyle):
    def __init__(
        self,
        *,
        hccl_comm_dict: dict | None = None,
        ep_size: int | None = None,
        ep_rank: int | None = None,
        global_rank: int | None = None,
    ):
        super().__init__()
        self.hccl_comm_dict = hccl_comm_dict
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.global_rank = global_rank

    def _deepseek_v4_moe_partition_fn(self, src_data_rank, name, module, device_mesh):
        del src_data_rank
        del device_mesh
        if name or module.__class__.__name__ != "DeepseekV4MoE":
            return
        if self.hccl_comm_dict is not None:
            module.hccl_comm_dict = self.hccl_comm_dict
            module.ep_group = self.hccl_comm_dict.get("moe_ep_group")
        if self.ep_size is not None:
            module.ep_size = self.ep_size
        if self.ep_rank is not None:
            module.ep_rank = self.ep_rank
        if self.ep_size is not None and self.ep_rank is not None:
            module.experts_per_rank = module.n_routed_experts // module.ep_size
            module.ep_start = module.ep_rank * module.experts_per_rank
            module.ep_end = module.ep_start + module.experts_per_rank
        module.dispatch_kwargs = None
        module.combine_kwargs = None

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        partition_fn, _, _, _, _ = self.get_dist_info(module)
        if partition_fn is None and module.__class__.__name__ == "DeepseekV4MoE":
            partition_fn = self._deepseek_v4_moe_partition_fn

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            None,
            None,
            parallel_style_name=self.__class__.__name__,
        )


MojoExpertParallel.register_dist_info(
    (MojoMoE, MojoQuantMoE),
    partiton_fn=_ep_partition_fn,
)
