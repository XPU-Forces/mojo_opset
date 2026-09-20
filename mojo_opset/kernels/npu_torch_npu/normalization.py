"""torch_npu normalization leaves migrated from the original provider."""

import torch
import torch_npu


@torch.library.custom_op("mojo_npu_torch_npu::rms_norm_infer", mutates_args=())
def rms_norm_infer_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch_npu.npu_rms_norm(x.contiguous(), weight.contiguous(), epsilon=eps)[0].contiguous()


@rms_norm_infer_fwd.register_fake
def _rms_norm_infer_fake(x, weight, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def group_rms_norm_infer_fwd(input_groups, weight, eps):
    # Keep every npu_rms_norm launch as its own leaf in a captured graph.
    return [
        rms_norm_infer_fwd(x, x.new_ones(x.shape[-1]) if weight is None else weight[i], eps)
        for i, x in enumerate(input_groups)
    ]


@torch.library.custom_op("mojo_npu_torch_npu::residual_add_rms_norm_infer", mutates_args=())
def residual_add_rms_norm_infer_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, _, summed = torch_npu.npu_add_rms_norm(x.contiguous(), residual.contiguous(), weight.contiguous(), eps)
    return output.contiguous(), summed.contiguous()


@residual_add_rms_norm_infer_fwd.register_fake
def _residual_add_rms_norm_infer_fake(x, residual, weight, eps):
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(x, memory_format=torch.contiguous_format),
    )
