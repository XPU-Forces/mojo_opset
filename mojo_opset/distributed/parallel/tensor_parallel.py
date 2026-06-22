import inspect

from functools import partial
from typing import List

import torch

from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed.tensor.placement_types import Shard

from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle
from mojo_opset.distributed.parallel.data_parallel import gather_lmhead_logits_for_attn_dp
from mojo_opset.distributed.parallel.data_parallel import prepare_lmhead_input_for_attn_dp


# DeepSeek V4 LMHead-TP helpers.
#
# These helpers keep model files free from LMHead-TP communication details while
# the generic MojoTensorParallel styles below remain reusable for regular modules.
def split_even_range(total_size: int, num_partitions: int, partition_rank: int) -> tuple[int, int]:
    base = total_size // num_partitions
    rem = total_size % num_partitions
    start = base * partition_rank + min(partition_rank, rem)
    end = start + base + (1 if partition_rank < rem else 0)
    return start, end


def get_deepseek_v4_lmhead_tp_partition(
    vocab_size: int,
    *,
    global_rank: int = 0,
    lmhead_tp_size: int = 1,
    use_parallelize_module_tp: bool = False,
) -> tuple[int, int, int, int]:
    """Return rank-local LMHead vocab partition information for DeepSeek V4."""

    init_tp_size = 1 if use_parallelize_module_tp and lmhead_tp_size > 1 else lmhead_tp_size
    tp_rank = global_rank % init_tp_size if init_tp_size > 1 else 0
    vocab_start, vocab_end = split_even_range(vocab_size, init_tp_size, tp_rank)
    return tp_rank, vocab_start, vocab_end, vocab_end - vocab_start


def deepseek_v4_lmhead_tp_forward(
    *,
    lm_head,
    hidden_states,
    local_vocab_size: int,
    vocab_size: int,
    attn_dp_size: int,
    lmhead_tp_size: int,
    lmhead_tp_group=None,
):
    """Run DeepSeek V4 LMHead with LMHead-TP and Attention-DP aware communication."""

    local_bs, q_len, hidden_size = hidden_states.shape
    hidden_states_for_lm = hidden_states.view(local_bs * q_len, 1, hidden_size).to(torch.bfloat16)
    hidden_states_for_lm = prepare_lmhead_input_for_attn_dp(
        hidden_states_for_lm,
        attn_dp_size=attn_dp_size,
        lmhead_tp_size=lmhead_tp_size,
        lmhead_tp_group=lmhead_tp_group,
    )

    logits_flat = lm_head(hidden_states_for_lm.view(-1, hidden_size))
    logits = logits_flat.view(*hidden_states_for_lm.shape[:-1], local_vocab_size)
    logits = gather_lmhead_logits_for_attn_dp(
        logits,
        local_batch=local_bs,
        q_len=q_len,
        vocab_size=vocab_size,
        attn_dp_size=attn_dp_size,
        lmhead_tp_size=lmhead_tp_size,
        lmhead_tp_group=lmhead_tp_group,
    )
    return logits.reshape(local_bs, q_len, -1)


class MojoTensorParallel(MojoRegisterableParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Placement,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        output_layouts: Placement,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts
        self.use_local_output = use_local_output

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        (
            partition_fn,
            prepare_input_fn,
            prepare_output_fn,
            desired_input_layouts,
            desired_output_layouts,
        ) = self.get_dist_info(module)

        prepare_input_fn = prepare_input_fn if prepare_input_fn else self.prepare_input_fn
        prepare_output_fn = prepare_output_fn if prepare_output_fn else self.prepare_output_fn

        if desired_input_layouts:
            prepare_input_fn = partial(prepare_input_fn, desired_input_layouts)
        else:
            try:
                if inspect.signature(prepare_input_fn).parameters["desired_input_layouts"].default == inspect._empty:
                    prepare_input_fn = partial(prepare_input_fn, None)
            except KeyError:
                ...
        if desired_output_layouts:
            prepare_output_fn = partial(prepare_output_fn, desired_output_layouts)
        else:
            try:
                if inspect.signature(prepare_output_fn).parameters["desired_output_layouts"].default == inspect._empty:
                    prepare_output_fn = partial(prepare_output_fn, None)
            except KeyError:
                ...

        # WARNING(liuyuan): we should follow the positional parameter order.
        prepare_input_fn = partial(prepare_input_fn, self.input_layouts)
        prepare_output_fn = partial(prepare_output_fn, self.output_layouts, self.use_local_output)

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            prepare_input_fn,
            prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )


class MojoRowwiseParallel(MojoTensorParallel):
    def __init__(
        self,
        *,
        input_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        output_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts or (Shard(-1),),
            output_layouts=output_layouts or (Replicate(),),
            use_local_output=use_local_output,
        )


class MojoColwiseParallel(MojoTensorParallel):
    def __init__(
        self,
        *,
        input_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        output_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts or (Replicate(),),
            output_layouts=output_layouts or (Shard(-1),),
            use_local_output=use_local_output,
        )


class MojoSwiGLUParallel(MojoTensorParallel):
    def __init__(self, **kwargs):
        super().__init__(
            input_layouts=kwargs.get("input_layouts") or (Replicate(),),
            output_layouts=kwargs.get("output_layouts") or (Replicate(),),
            use_local_output=kwargs.get("use_local_output", True),
        )


class MojoQKVColwiseParallel(MojoColwiseParallel):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        partition_fn, _, _, desired_input_layouts, desired_output_layouts = self.get_dist_info(module)

        if partition_fn is not None:
            partition_fn = partial(
                partition_fn,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )

        prepare_input_fn = self.prepare_input_fn
        prepare_output_fn = self.prepare_output_fn
        if desired_input_layouts:
            prepare_input_fn = partial(prepare_input_fn, desired_input_layouts)
        else:
            prepare_input_fn = partial(prepare_input_fn, None)
        if desired_output_layouts:
            prepare_output_fn = partial(prepare_output_fn, desired_output_layouts)
        else:
            prepare_output_fn = partial(prepare_output_fn, None)
        prepare_input_fn = partial(prepare_input_fn, self.input_layouts)
        prepare_output_fn = partial(prepare_output_fn, self.output_layouts, self.use_local_output)

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            prepare_input_fn,
            prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )
