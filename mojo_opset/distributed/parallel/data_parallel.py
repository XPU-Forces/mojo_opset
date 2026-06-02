from typing import Dict
from typing import List

import torch
import torch.distributed as dist

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement

from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle


class MojoDataParallel(MojoRegisterableParallelStyle):
    def __init__(
        self,
        *,
        desired_args_input_layouts: List[Placement] = [],  # Layouts used only for non-DTensor inputs
        desired_kwargs_input_layouts: Dict[str, Placement] = {},  # Layouts used only for non-DTensor inputs
        desired_output_layouts: List[Placement] = [],  # This layout is the placement used to convert the local tensor
        # output by the operator into the corresponding DTensor according to the distributed semantics required by the operator
        use_local_output: bool = True,
    ):
        super().__init__()
        assert desired_args_input_layouts or desired_kwargs_input_layouts or desired_output_layouts
        self.desired_args_input_layouts = desired_args_input_layouts
        self.desired_kwargs_input_layouts = desired_kwargs_input_layouts
        self.desired_output_layouts = desired_output_layouts
        self.use_local_output = use_local_output

    def prepare_input_fn(
        self,
        device_mesh,
        *args,
        **kwargs,
    ):
        def mapping(tensor, desired_input_layout):  # desired_input_layout is used only for non-DTensor inputs
            if not isinstance(tensor, torch.Tensor):
                return tensor
            desired_input_layout = (
                [desired_input_layout] if isinstance(desired_input_layout, Placement) else desired_input_layout
            )

            if not isinstance(tensor, DTensor):
                tensor = DTensor.from_local(tensor, device_mesh, desired_input_layout, run_check=False)

            if tensor.placements != desired_input_layout:
                tensor = tensor.redistribute(placements=desired_input_layout, async_op=True)
            return tensor.to_local()

        args = list(args)
        for (idx, input_tensor), desired_input_layout in zip(enumerate(args), self.desired_args_input_layouts):
            args[idx] = mapping(input_tensor, desired_input_layout)
        for key, input_tensor in kwargs.items():
            if key in self.desired_kwargs_input_layouts:
                kwargs[key] = mapping(input_tensor, self.desired_kwargs_input_layouts[key])

        return (tuple(args), kwargs)

    def prepare_output_fn(self, device_mesh, outputs):
        is_single = False
        is_tuple = False
        if isinstance(outputs, (list, tuple)):
            is_tuple = isinstance(outputs, tuple)
            outputs = list(outputs)
        else:
            outputs = [outputs]
            is_single = True

        # desired_output_layout is used only for non-DTensor outputs
        for (idx, output_tensor), desired_output_layout in zip(
            enumerate(outputs),
            self.desired_output_layouts,
        ):
            if not isinstance(output_tensor, torch.Tensor):
                continue
            desired_output_layout = (
                [desired_output_layout] if isinstance(desired_output_layout, Placement) else desired_output_layout
            )

            if not isinstance(output_tensor, DTensor):
                output_tensor = DTensor.from_local(
                    output_tensor,
                    device_mesh,
                    desired_output_layout,
                    run_check=False,
                )
            if output_tensor.placements != desired_output_layout:
                output_tensor = output_tensor.redistribute(
                    placements=desired_output_layout,
                    async_op=True,
                )
            outputs[idx] = output_tensor.to_local() if self.use_local_output else output_tensor
            if isinstance(outputs[idx], AsyncCollectiveTensor):
                outputs[idx] = outputs[idx].wait()

        return outputs[0] if is_single else tuple(outputs) if is_tuple else outputs

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh):

        return MojoDistributedModule(
            module,
            device_mesh,
            None,
            self.prepare_input_fn,
            self.prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )


def get_attn_dp_shard_range(total_batch, *, global_rank, world_size, attn_tp_size, cp_size=1):
    if total_batch <= 0:
        return 0, 0, 1, 0
    if attn_tp_size <= 0:
        raise ValueError(f"attn_tp_size must be > 0, got {attn_tp_size}")
    if cp_size <= 0:
        raise ValueError(f"cp_size must be > 0, got {cp_size}")

    # Golden-style split: decode/batch ownership is controlled by attention DP
    # only. CP prefill uses a separate prefill_dp_size and, when cp_size > 1,
    # requires cp_size == world_size so all ranks form one CP group.
    if cp_size > 1 and world_size != cp_size:
        raise ValueError(
            f"Golden-style CP requires cp_size == world_size when CP is enabled, "
            f"got cp_size={cp_size}, world_size={world_size}."
        )
    if world_size % attn_tp_size != 0:
        raise ValueError(f"world_size={world_size} is not divisible by attn_tp_size={attn_tp_size}")

    if cp_size > 1:
        if world_size % (cp_size * attn_tp_size) != 0:
            raise ValueError(
                f"world_size={world_size} must be divisible by cp_size*attn_tp_size="
                f"{cp_size * attn_tp_size} when CP is enabled"
            )
        # Golden CP prefill semantics: CP ranks cooperate on the same sequence,
        # so batch ownership is controlled by prefill DP, not decode attention DP.
        dp_group_count = world_size // cp_size // attn_tp_size
        dp_rank = global_rank // (cp_size * attn_tp_size)
    else:
        dp_group_count = world_size // attn_tp_size
        dp_rank = global_rank // attn_tp_size
    if total_batch % dp_group_count != 0:
        raise ValueError(
            f"DeepseekV4 DP requires batch_size={total_batch} to be divisible by "
            f"attn_dp_size={dp_group_count} (world_size={world_size}, attn_tp_size={attn_tp_size})."
        )

    shard_size = total_batch // dp_group_count
    start = dp_rank * shard_size
    end = start + shard_size
    return start, end, dp_group_count, dp_rank


def gather_decode_shard_tensor(tensor, total_batch, shard_start, attn_dp_size):
    del shard_start
    if not dist.is_initialized() or attn_dp_size <= 1:
        return tensor
    if tensor.shape[0] == total_batch:
        return tensor
    local_batch = tensor.shape[0]
    if total_batch != local_batch * attn_dp_size:
        raise ValueError(
            "Cannot gather decode shards with uneven local batch: "
            f"total_batch={total_batch}, local_batch={local_batch}, attn_dp_size={attn_dp_size}"
        )
    gathered = tensor.new_empty((attn_dp_size, *tensor.shape))
    dist.all_gather_into_tensor(gathered, tensor.contiguous())
    return gathered.reshape(total_batch, *tensor.shape[1:])


def shard_batch_for_attn_dp(
    input_ids,
    attention_mask,
    lengths,
    prompts,
    rendered,
    *,
    global_rank,
    world_size,
    attn_tp_size,
    cp_size=1,
):
    start, end, _, _ = get_attn_dp_shard_range(
        input_ids.shape[0],
        global_rank=global_rank,
        world_size=world_size,
        attn_tp_size=attn_tp_size,
        cp_size=cp_size,
    )
    if start >= end:
        return input_ids, attention_mask, lengths, prompts, rendered

    return (
        input_ids[start:end],
        attention_mask[start:end],
        lengths[start:end],
        prompts[start:end],
        rendered[start:end],
    )
