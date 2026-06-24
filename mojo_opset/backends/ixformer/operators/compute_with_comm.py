import torch
import torch.distributed as dist

from ixformer import functions as ixf_f
# from ixformer.distributed import symmetric_memory as symm
try:
    from ixformer.distributed import symmetric_memory as symm
except ImportError:
    symm = None 

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


def _destroy_symm_workspace(module):
    if getattr(module, "_workspace", None) is None:
        return

    try:
        group = getattr(module, "process_group", None)
        group_name = getattr(group, "group_name", None)
        module._workspace = None
        symm.destroy(group_name=group_name)
    except Exception:
        pass


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

        self._workspace = None
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(group=self.process_group)
            self._enable_symm_mem(self.process_group)
            workspace_bytes = ixf_f.allgather_quant_matmul_workspace_bytes(
                4096, self.in_features, self.world_size
            )
            device = torch.device("cuda", torch.cuda.current_device())
            self._workspace = symm.empty(
                workspace_bytes,
                dtype=torch.int8,
                device=device,
            )
            self._workspace.zero_()
            symm.rendezvous(self._workspace, self.process_group)
            dist.barrier(group=self.process_group)

    def _enable_symm_mem(self, pg):
        if symm.is_nvshmem_available():
            symm.set_backend("NVSHMEM")
        symm.enable_symm_mem_for_group(pg.group_name)

    def __del__(self):
        _destroy_symm_workspace(self)

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input K {input.shape[-1]} must match in_features {self.in_features}."
            )

        if dist.is_available() and dist.is_initialized():
            seq_local = input.shape[0]
            if seq_local % 256 != 0:
                raise NotImplementedError(
                    "IxformerAllGatherQuantGemm requires local sequence length divisible by BM=256"
                )

            return ixf_f.allgather_quant_matmul(
                input,
                self.weight,
                input_scale.reshape(-1),
                self.weight_scale.to(torch.float32),
                workspace=self._workspace,
                group=self.process_group,
                format="NN",
            )

        return ixf_f.w8a8(
            input, self.weight, input_scale, self.weight_scale,
            format="NN",
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
        self._workspace = None

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(group=self.process_group)
            self._enable_symm_mem(self.process_group)
            workspace_bytes = ixf_f.quant_matmul_reducescatter_workspace_bytes(
                8192, self.out_features, self.in_features, self.world_size
            )
            device = torch.device("cuda", torch.cuda.current_device())
            self._workspace = symm.empty(
                workspace_bytes,
                dtype=torch.int8,
                device=device,
            )
            self._workspace.zero_()
            symm.rendezvous(self._workspace, self.process_group)
            dist.barrier(group=self.process_group)

    def _enable_symm_mem(self, pg):
        if symm.is_nvshmem_available():
            symm.set_backend("NVSHMEM")
        symm.enable_symm_mem_for_group(pg.group_name)

    def __del__(self):
        _destroy_symm_workspace(self)

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input K {input.shape[-1]} must match in_features {self.in_features}."
            )

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size(group=self.process_group)
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
                raise NotImplementedError(
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
                format="NN"
            )

        partial = ixf_f.w8a8(
            input, self.weight, input_scale, self.weight_scale,
            format="NN",
            out_dtype=self.output_dtype,
        )
        return partial
