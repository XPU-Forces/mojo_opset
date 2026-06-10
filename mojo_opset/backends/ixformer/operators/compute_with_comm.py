import torch
import torch.distributed as dist

import ixformer.distributed as ixfd
from ixformer import functions as ixf_f
from ixformer.distributed import symmetric_memory as symm

from mojo_opset.core.operators.compute_with_comm import (
    MojoAllGatherQuantGemm,
    MojoQuantGemmReduceScatter,
)


def _cast_weight_scale_to_fp32(module, incompatible_keys):
    if module.trans_weight:
        module.weight = torch.nn.Parameter(module.weight.transpose(0, 1).contiguous(), requires_grad=module.weight.requires_grad)

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
        self._symm_group_name = "mojo_quant_gemm_reduce_scatter"
        self.register_load_state_dict_post_hook(_cast_weight_scale_to_fp32)
        self._workspace = None
        self._workspace_bytes = 0

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(group=self.process_group)
            if not symm.is_initialized(group_name=self._symm_group_name):
                symm.init_process_group(group=self.process_group, group_name=self._symm_group_name)
            self._workspace_bytes = ixf_f.quant_matmul_reducescatter_workspace_bytes(
                8192, self.out_features, self.in_features, self.world_size
            )
            self._workspace = symm.empty_p2p(
                (self._workspace_bytes,),
                dtype=torch.int8,
                device=torch.device("cuda"),
                group_name=self._symm_group_name,
            )
        

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
            seq_full = input.shape[0]
            if seq_full > 8192:
                raise ValueError(
                    f"seq_full {seq_full} must be less than or equal to 8192"
                )
            if seq_full % world_size != 0:
                raise ValueError(
                    f"seq_full {seq_full} must be divisible by world_size {world_size}"
                )
            if (seq_full // world_size) % 256 != 0:
                raise ValueError(
                    "IxformerQuantGemmReduceScatter requires seq_full / world_size divisible by BM=256"
                )
            if self.output_dtype != torch.bfloat16:
                raise NotImplementedError(
                    "IxformerQuantGemmReduceScatter fused ixformer path only supports bfloat16 output dtype"
                )
            if input_scale.numel() != seq_full:
                raise ValueError(
                    f"input_scale must contain one scale per input row, got {input_scale.numel()} and {seq_full}"
                )

            return ixf_f.quant_matmul_reducescatter(
                input,
                self.weight,
                input_scale,
                self.weight_scale,
                workspace=self._workspace,
                group=self.process_group,
                group_name=self._symm_group_name,
            )

        partial = ixf_f.w8a8(
            input, self.weight, input_scale, self.weight_scale,
            format="NN",
            out_dtype=self.output_dtype,
        )
        return partial
