import inspect

from functools import partial
from typing import List

import torch
import torch.nn as nn

from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Partial
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed.tensor.placement_types import Shard

from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle
from mojo_opset.distributed.parallel.utils import shard_tensor
from mojo_opset.distributed.parallel.utils import stat_dict_rename_hook


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


class MojoAttnHeadParallel(MojoTensorParallel):
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
            output_layouts=output_layouts or (Shard(-2),),
            use_local_output=use_local_output,
        )


class MojoSwiGLUParallel(MojoTensorParallel):
    """Tensor-parallel style for fused SwiGLU MLP whose fc1 packs [gate | up].

    Applies ColwiseParallel to fc1 with gate/up-aware sharding (each TP rank
    gets the *paired* gate and up slices so that ``chunk(2)`` gives correct
    results), and RowwiseParallel to fc2.  The all-reduce is performed on the
    combined module output.

    Usage in a parallelise plan::

        "layers.*.experts_share": MojoSwiGLUParallel()
    """

    def __init__(self, **kwargs):
        super().__init__(
            input_layouts=kwargs.get("input_layouts") or (Replicate(),),
            output_layouts=kwargs.get("output_layouts") or (Replicate(),),
            use_local_output=kwargs.get("use_local_output", True),
        )

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        def swiglu_partition_fn(src_data_rank, name, mod, mesh):
            if not isinstance(mod, nn.Linear):
                return
            rank = mesh.get_local_rank()
            size = mesh.size()

            if name == "fc1":
                weight = shard_tensor(mesh, [Replicate()], src_data_rank, mod.weight)
                half = weight.shape[0] // 2
                per_tp = half // size
                gate_shard = weight[rank * per_tp : (rank + 1) * per_tp]
                up_shard = weight[half + rank * per_tp : half + (rank + 1) * per_tp]
                new_weight = torch.cat([gate_shard, up_shard], dim=0)
                mod.register_parameter("weight", nn.Parameter(new_weight))
            elif name == "fc2":
                new_weight = shard_tensor(mesh, [Shard(1)], src_data_rank, mod.weight)
                mod.register_parameter("weight", nn.Parameter(new_weight))
            else:
                return

            mod.register_state_dict_post_hook(partial(stat_dict_rename_hook, ("weight", "bias"), mesh))

        prepare_input_fn = partial(
            MojoRegisterableParallelStyle.prepare_input_fn,
            None,
            self.input_layouts,
        )
        prepare_output_fn = partial(
            MojoRegisterableParallelStyle.prepare_output_fn,
            [Partial()],
            self.output_layouts,
            self.use_local_output,
        )

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(swiglu_partition_fn, self.src_data_rank),
            prepare_input_fn,
            prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )


class MojoQKVColwiseParallel(MojoColwiseParallel):
    """Head-aware QKV column-wise parallel style for GQA models.

    When num_kv_heads < tp_size, KV heads are replicated so that each rank
    gets at least one KV head, while Q heads are split evenly.
    """

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
        num_q_heads = self.num_q_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        def qkv_partition_fn(src_data_rank, name, mod, mesh):
            if not isinstance(mod, nn.Linear):
                return
            rank = mesh.get_local_rank()
            size = mesh.size()

            q_total_dim = num_q_heads * head_dim
            kv_total_dim = num_kv_heads * head_dim

            # Replicate full weight from src_data_rank so every rank can
            # perform head-aware local slicing on valid data.  On meta
            # tensors the monkey-patched shard_tensor is a no-op identity.
            weight = shard_tensor(mesh, [Replicate()], src_data_rank, mod.weight)

            q_per_rank = num_q_heads // size
            q_start = rank * q_per_rank * head_dim
            q_end = q_start + q_per_rank * head_dim
            local_q = weight[q_start:q_end, :]

            replicate = max(1, size // num_kv_heads)
            kv_idx = rank // replicate
            k_offset = q_total_dim
            k_start = k_offset + kv_idx * head_dim
            local_k = weight[k_start : k_start + head_dim, :]

            v_offset = q_total_dim + kv_total_dim
            v_start = v_offset + kv_idx * head_dim
            local_v = weight[v_start : v_start + head_dim, :]

            new_weight = torch.cat([local_q, local_k, local_v], dim=0)
            mod.register_parameter("weight", nn.Parameter(new_weight))

            if mod.bias is not None:
                bias = shard_tensor(mesh, [Replicate()], src_data_rank, mod.bias)
                local_q_bias = bias[q_start:q_end]
                local_k_bias = bias[k_offset + kv_idx * head_dim : k_offset + kv_idx * head_dim + head_dim]
                local_v_bias = bias[v_offset + kv_idx * head_dim : v_offset + kv_idx * head_dim + head_dim]
                new_bias = torch.cat([local_q_bias, local_k_bias, local_v_bias], dim=0)
                mod.register_parameter("bias", nn.Parameter(new_bias))

            mod.register_state_dict_post_hook(partial(stat_dict_rename_hook, ("weight", "bias"), mesh))

        prepare_input_fn = partial(
            MojoRegisterableParallelStyle.prepare_input_fn,
            [Replicate()],
            self.input_layouts,
        )
        prepare_output_fn = partial(
            MojoRegisterableParallelStyle.prepare_output_fn,
            [Shard(-1)],
            self.output_layouts,
            self.use_local_output,
        )

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(qkv_partition_fn, self.src_data_rank),
            prepare_input_fn,
            prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )
