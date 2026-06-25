import torch
import torch.distributed as dist

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


def _restore_weight_layout_state_dict_post_hook(module, state_dict, prefix, local_metadata):
    """Dual of ``_cast_weight_scale_to_fp32``.

    The runtime weight is post-transpose ``[in, out]`` (when ``trans_weight``
    is True). ``state_dict()`` would normally emit that runtime layout, which
    cannot be re-loaded via ``load_state_dict()`` because the freshly-built
    module's Parameter is still ``[out, in]`` and the load post-hook would
    transpose a second time. Restore the pre-transpose ``[out, in]`` layout
    here so the load post-hook reproduces the runtime state correctly. The
    ``weight_scale`` fp32 cast in the load hook is dtype-idempotent (fp32 ->
    fp32 round-trips losslessly), no inverse needed.
    """
    if module.trans_weight:
        key = prefix + "weight"
        if key in state_dict:
            state_dict[key] = state_dict[key].transpose(0, 1).contiguous()


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
        self.register_state_dict_post_hook(_restore_weight_layout_state_dict_post_hook)

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
            BM = 256
            pad = (-seq_local) % BM  # 0 if already aligned, else rows to append

            if pad > 0:
                # Pad input on the seq dim and matching positions in scale.
                # Each rank pads its own local segment; after all-gather the
                # global output is sliced back to the unpadded layout below.
                input_padded = torch.zeros(
                    seq_local + pad, self.in_features,
                    dtype=input.dtype, device=input.device,
                )
                input_padded[:seq_local].copy_(input)

                scale_flat = input_scale.reshape(-1)
                if scale_flat.numel() == seq_local:
                    scale_padded = torch.zeros(
                        seq_local + pad,
                        dtype=scale_flat.dtype, device=scale_flat.device,
                    )
                    scale_padded[:seq_local].copy_(scale_flat)
                else:
                    raise ValueError(
                        f"input_scale must contain one scale per input row to pad, "
                        f"got {scale_flat.numel()} for seq_local {seq_local}"
                    )
            else:
                input_padded = input
                scale_padded = input_scale.reshape(-1)

            out = ixf_f.allgather_quant_matmul(
                input_padded,
                self.weight,
                scale_padded,
                self.weight_scale.to(torch.float32),
                workspace=self._workspace,
                group=self.process_group,
                format="NN",
            )

            if pad > 0:
                # out shape: (world_size * (seq_local + pad), out_features) with
                # per-rank padding rows at [k*seq_padded + seq_local : (k+1)*seq_padded].
                # Reshape to (world_size, seq_padded, out_features) and slice
                # off the padding to recover (world_size * seq_local, out_features).
                seq_padded = seq_local + pad
                out_view = out.view(self.world_size, seq_padded, -1)
                out = out_view[:, :seq_local, :].contiguous().view(
                    self.world_size * seq_local, -1
                )
            return out

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
        self.register_state_dict_post_hook(_restore_weight_layout_state_dict_post_hook)
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
            if self.output_dtype != torch.bfloat16:
                raise NotImplementedError(
                    "IxformerQuantGemmReduceScatter fused ixformer path only supports bfloat16 output dtype"
                )
            if input_scale.numel() != seq_full:
                raise ValueError(
                    f"input_scale must contain one scale per input row, got {input_scale.numel()} and {seq_full}"
                )

            seq_per_rank = seq_full // world_size
            BM = 256
            pad_per_rank = (-seq_per_rank) % BM  # rows to append per rank

            if pad_per_rank > 0:
                # Reduce-scatter splits the output by contiguous row blocks
                # of length seq_per_rank. To keep each rank's slice aligned
                # to BM, padding must be inserted between rank segments
                # (not just appended at the tail). Reshape (world_size,
                # seq_per_rank, ...), zero-pad along the per-rank axis, then
                # flatten back.
                seq_per_rank_padded = seq_per_rank + pad_per_rank
                input_r = input.view(world_size, seq_per_rank, self.in_features)
                input_padded = torch.zeros(
                    world_size, seq_per_rank_padded, self.in_features,
                    dtype=input.dtype, device=input.device,
                )
                input_padded[:, :seq_per_rank, :].copy_(input_r)
                input_p = input_padded.view(
                    world_size * seq_per_rank_padded, self.in_features
                )

                scale_r = input_scale.reshape(world_size, seq_per_rank)
                scale_padded = torch.zeros(
                    world_size, seq_per_rank_padded,
                    dtype=input_scale.dtype, device=input_scale.device,
                )
                scale_padded[:, :seq_per_rank].copy_(scale_r)
                scale_p = scale_padded.view(world_size * seq_per_rank_padded)
            else:
                input_p = input
                scale_p = input_scale

            out = ixf_f.quant_matmul_reducescatter(
                input_p,
                self.weight,
                scale_p,
                self.weight_scale,
                workspace=self._workspace,
                group=self.process_group,
                format="NN"
            )

            if pad_per_rank > 0:
                # Per-rank output is (seq_per_rank_padded, out_features) with
                # padding rows at the tail. Slice back to (seq_per_rank, out).
                out = out[:seq_per_rank, :].contiguous()
            return out

        partial = ixf_f.w8a8(
            input, self.weight, input_scale, self.weight_scale,
            format="NN",
            out_dtype=self.output_dtype,
        )
        return partial
