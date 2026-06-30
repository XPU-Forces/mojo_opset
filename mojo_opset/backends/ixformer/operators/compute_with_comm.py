import torch
import torch.distributed as dist

import ixformer.distributed as ixfd
from ixformer import functions as ixf_f

from mojo_opset.core.operators.compute_with_comm import (
    MojoAllGatherQuantGemm,
    MojoQuantGemmReduceScatter,
)


def _cast_weight_scale_to_fp32(module, incompatible_keys):
    module.weight_scale = torch.nn.Parameter(
        module.weight_scale.detach().to(torch.float32),
        requires_grad=module.weight_scale.requires_grad,
    )


class IxformerAllGatherQuantGemm(MojoAllGatherQuantGemm):
    supported_platforms_list = ["ilu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.gather_dim != 0:
            raise NotImplementedError(
                f"IxformerAllGatherQuantGemm only supports gather_dim=0, got {self.gather_dim}"
            )
        if self.output_dtype == torch.float32:
            raise NotImplementedError(
                "IxformerAllGatherQuantGemm does not support float32 output dtype"
            )
        self.register_load_state_dict_post_hook(_cast_weight_scale_to_fp32)

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input K {input.shape[-1]} must match in_features {self.in_features}."
            )

        if dist.is_available() and dist.is_initialized():
            pg = self.process_group
            world_size = dist.get_world_size(group=pg)
            seq_local = input.shape[0]
            seq_full = seq_local * world_size

            input_full = torch.empty(
                (seq_full, input.shape[1]), dtype=input.dtype, device=input.device,
            )
            scale_full = torch.empty(
                (seq_full, *input_scale.shape[1:]),
                dtype=input_scale.dtype, device=input_scale.device,
            )
            ixfd.all_gather_into_tensor(
                input_full, input.contiguous(), group=pg, async_op=True,
            )
            ixfd.all_gather_into_tensor(
                scale_full, input_scale.contiguous(), group=pg, async_op=True,
            )
            input, input_scale = input_full, scale_full

        input_scale = input_scale.reshape(-1)
        if input_scale.numel() != input.shape[0]:
            raise ValueError(
                f"input_scale must contain one scale per input row, got {input_scale.numel()} and {input.shape[0]}"
            )
        return ixf_f.w8a8(
            input, self.weight, input_scale, self.weight_scale,
            format="TN" if self.trans_weight else "NN",
            out_dtype=self.output_dtype,
        )


class IxformerQuantGemmReduceScatter(MojoQuantGemmReduceScatter):
    supported_platforms_list = ["ilu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scatter_dim != 0:
            raise NotImplementedError(
                f"IxformerQuantGemmReduceScatter only supports scatter_dim=0, got {self.scatter_dim}"
            )
        if self.output_dtype == torch.float32:
            raise NotImplementedError(
                "IxformerQuantGemmReduceScatter does not support float32 output dtype"
            )
        self.register_load_state_dict_post_hook(_cast_weight_scale_to_fp32)

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input K {input.shape[-1]} must match in_features {self.in_features}."
            )

        input_scale = input_scale.reshape(-1)
        if input_scale.numel() != input.shape[0]:
            raise ValueError(
                f"input_scale must contain one scale per input row, got {input_scale.numel()} and {input.shape[0]}"
            )
        partial = ixf_f.w8a8(
            input, self.weight, input_scale, self.weight_scale,
            format="TN" if self.trans_weight else "NN",
            out_dtype=self.output_dtype,
        )

        if dist.is_available() and dist.is_initialized():
            pg = self.process_group
            world_size = dist.get_world_size(group=pg)
            seq_full = partial.shape[0]
            if seq_full % world_size != 0:
                raise ValueError(
                    f"seq_full {seq_full} must be divisible by world_size {world_size}"
                )
            out = torch.empty(
                (seq_full // world_size, partial.shape[1]),
                dtype=partial.dtype, device=partial.device,
            )
            ixfd.reduce_scatter_tensor(
                out, partial.contiguous(), group=pg, async_op=True,
            )
            return out
        return partial
