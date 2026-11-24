"""Standalone grouped GEMM with native forward, dgrad and wgrad."""

import torch

from ._extension import launcher


@torch.library.custom_op("mojo_npu_native_a2::group_gemm", mutates_args=())
def group_gemm_fwd(x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor, trans_weight: bool) -> torch.Tensor:
    return launcher("group_gemm")(
        x.contiguous(), weight.contiguous(), counts.to(device=x.device, dtype=torch.int32).contiguous(), trans_weight
    )


@group_gemm_fwd.register_fake
def _fake_fwd(x, weight, counts, trans_weight):
    return x.new_empty((x.shape[0], weight.shape[1 if trans_weight else 2]))


@torch.library.custom_op("mojo_npu_native_a2::group_gemm_wgrad", mutates_args=())
def _wgrad(dy: torch.Tensor, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return launcher("group_gemm_wgrad")(
        dy.contiguous(), x.contiguous(), counts.to(device=x.device, dtype=torch.int32).contiguous()
    )


@_wgrad.register_fake
def _fake_wgrad(dy, x, counts):
    return x.new_empty((counts.numel(), dy.shape[1], x.shape[1]))


def group_gemm_bwd(dy, x, weight, counts, trans_weight, need_input, need_weight):
    dy = dy.contiguous()
    dx = group_gemm_fwd(dy, weight, counts, not trans_weight) if need_input else None
    dw = _wgrad(dy, x, counts) if need_weight else None
    if dw is not None and not trans_weight:
        dw = dw.transpose(1, 2).contiguous()
    return dx, dw
