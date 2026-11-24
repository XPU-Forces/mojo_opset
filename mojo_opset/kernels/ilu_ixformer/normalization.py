"""Ixformer inference leaves; no cross-provider fallback."""

from typing import Optional

import torch
from ixformer import functions as ixf_f


@torch.library.custom_op("mojo_ilu_ixformer::layer_norm_infer", mutates_args=())
def layer_norm_infer_fwd(
    x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float,
) -> torch.Tensor:
    if weight is None or bias is None:
        raise NotImplementedError("Ixformer LayerNorm requires weight and bias")
    output, _ = ixf_f.residual_layer_norm(
        x.contiguous(), weight.contiguous(), bias.contiguous(),
        residual=None, residual_bias=None, eps=eps
    )
    return output.contiguous()


@layer_norm_infer_fwd.register_fake
def _layer_norm_infer_fake(x, weight, bias, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_ilu_ixformer::rms_norm_infer", mutates_args=())
def rms_norm_infer_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output, _ = ixf_f.residual_rms_norm(
        x.contiguous(), weight.contiguous(), eps=eps, residual=None, residual_bias=None
    )
    return output.contiguous()


@rms_norm_infer_fwd.register_fake
def _rms_norm_infer_fake(x, weight, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_ilu_ixformer::group_rms_norm_pair_infer", mutates_args=())
def _group_rms_norm_pair_infer(
    q: torch.Tensor, k: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_q, output_k = ixf_f.rms_norm_qk(q, k, weight_q, weight_k, eps=eps, norm_type="general")
    return (output_q.clone(memory_format=torch.contiguous_format),
            output_k.clone(memory_format=torch.contiguous_format))


@_group_rms_norm_pair_infer.register_fake
def _group_rms_norm_pair_infer_fake(q, k, weight_q, weight_k, eps):
    return (torch.empty_like(q, memory_format=torch.contiguous_format),
            torch.empty_like(k, memory_format=torch.contiguous_format))


def group_rms_norm_infer_fwd(input_groups, weight, eps):
    if weight is None:
        weight = input_groups[0].new_ones((len(input_groups), input_groups[0].shape[-1]))
    # Exact original ixformer eligibility: adjacent Q/K views with supported D.
    outputs = []
    for start in range(0, len(input_groups), 2):
        q = input_groups[start]
        k = input_groups[start + 1] if start + 1 < len(input_groups) else None
        can_pair = (len(input_groups) % 2 == 0 and k is not None
                    and q.ndim == k.ndim and q.shape[-1] in (64, 128, 192, 256)
                    and q.data_ptr() + q.size(1) * q.stride(1) * q.element_size() == k.data_ptr())
        if can_pair:
            outputs.extend(_group_rms_norm_pair_infer(q, k, weight[start], weight[start + 1], eps))
        else:
            outputs.append(rms_norm_infer_fwd(q, weight[start], eps))
            if k is not None:
                outputs.append(rms_norm_infer_fwd(k, weight[start + 1], eps))
    return outputs


@torch.library.custom_op("mojo_ilu_ixformer::residual_add_layer_norm_infer", mutates_args=())
def residual_add_layer_norm_infer_fwd(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, summed = ixf_f.residual_layer_norm(
        x.contiguous(), weight.contiguous(), bias.contiguous(), residual=residual.contiguous(),
        residual_bias=None, eps=eps
    )
    return output.contiguous(), summed.contiguous()


@residual_add_layer_norm_infer_fwd.register_fake
def _residual_add_layer_norm_infer_fake(x, residual, weight, bias, eps):
    return (torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty_like(x, memory_format=torch.contiguous_format))


@torch.library.custom_op("mojo_ilu_ixformer::residual_add_rms_norm_infer", mutates_args=())
def residual_add_rms_norm_infer_fwd(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, summed = ixf_f.residual_rms_norm(
        x.contiguous(), weight.contiguous(), eps=eps, residual=residual.contiguous(),
        residual_alpha=1.0, residual_bias=None, is_post=False
    )
    return output.contiguous(), summed.contiguous()


@residual_add_rms_norm_infer_fwd.register_fake
def _residual_add_rms_norm_infer_fake(x, residual, weight, eps):
    return (torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty_like(x, memory_format=torch.contiguous_format))
