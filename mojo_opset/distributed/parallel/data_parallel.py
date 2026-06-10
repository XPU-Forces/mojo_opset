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


def prepare_lmhead_input_for_attn_dp(
    hidden_states,
    *,
    attn_dp_size: int,
    lmhead_tp_size: int,
    lmhead_tp_group=None,
):
    """Gather Attention-DP hidden shards inside each LMHead-TP group before lm_head."""

    if not dist.is_initialized() or attn_dp_size <= 1 or lmhead_tp_size <= 1:
        return hidden_states
    if lmhead_tp_group is None:
        raise ValueError("lmhead_tp_group is required when attn_dp_size > 1 and lmhead_tp_size > 1.")
    gathered_hidden = hidden_states.new_empty(
        lmhead_tp_size * hidden_states.shape[0],
        hidden_states.shape[1],
        hidden_states.shape[2],
    )
    dist.all_gather_into_tensor(gathered_hidden, hidden_states.contiguous(), group=lmhead_tp_group)
    return gathered_hidden


def gather_lmhead_logits_for_attn_dp(
    logits,
    *,
    local_batch: int,
    q_len: int,
    vocab_size: int,
    attn_dp_size: int,
    lmhead_tp_size: int,
    lmhead_tp_group=None,
):
    """Gather LMHead-TP vocab shards while preserving Attention-DP batch ownership."""

    if not dist.is_initialized() or lmhead_tp_size <= 1:
        return logits
    if lmhead_tp_group is None:
        raise ValueError("lmhead_tp_group is required when lmhead_tp_size > 1.")
    if attn_dp_size == 1:
        gathered_logits = logits.new_empty(
            lmhead_tp_size * logits.shape[0],
            logits.shape[1],
            logits.shape[2],
        )
        dist.all_gather_into_tensor(gathered_logits, logits.contiguous(), group=lmhead_tp_group)
    else:
        gathered_logits = logits.new_empty(logits.numel()).view(-1)
        dist.all_to_all_single(gathered_logits, logits.contiguous().view(-1), group=lmhead_tp_group)
        gathered_logits = gathered_logits.view_as(logits)

    gathered_logits = gathered_logits.reshape(
        lmhead_tp_size, local_batch * q_len, logits.shape[1], -1
    ).permute(1, 2, 0, 3)
    return gathered_logits.reshape(local_batch * q_len, logits.shape[1], -1)[..., :vocab_size]


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


def prepare_deepseek_v4_attn_dp_inputs(
    full_input_ids,
    full_attention_mask,
    full_lengths,
    prompts,
    rendered,
    *,
    global_rank,
    world_size,
    attn_tp_size,
    cp_size=1,
    pad_token_id=None,
    cp_prefill_pad_fn=None,
):
    """Prepare DeepSeek-V4 Attention-DP batch ownership and optional CP padding."""

    shard_start, shard_end, attn_dp_size, dp_rank = get_attn_dp_shard_range(
        full_input_ids.shape[0],
        global_rank=global_rank,
        world_size=world_size,
        attn_tp_size=attn_tp_size,
        cp_size=cp_size,
    )
    if cp_prefill_pad_fn is not None:
        if pad_token_id is None:
            raise ValueError("pad_token_id is required when cp_prefill_pad_fn is provided.")
        full_input_ids, full_attention_mask = cp_prefill_pad_fn(
            full_input_ids,
            full_attention_mask,
            pad_token_id,
            cp_size,
        )

    input_ids, attention_mask, lengths, local_prompts, local_rendered = shard_batch_for_attn_dp(
        full_input_ids,
        full_attention_mask,
        full_lengths,
        prompts,
        rendered,
        global_rank=global_rank,
        world_size=world_size,
        attn_tp_size=attn_tp_size,
        cp_size=cp_size,
    )
    return {
        "full_input_ids": full_input_ids,
        "full_attention_mask": full_attention_mask,
        "full_lengths": full_lengths,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "prompts": local_prompts,
        "rendered": local_rendered,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "attn_dp_size": attn_dp_size,
        "dp_rank": dp_rank,
    }
