"""Grouped GEMM via ixformer ``moe_w16a16_group_gemm``."""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from ixformer import functions as ixf_f
from mojo_opset.core import MojoGroupGemm


def _ixf_align_size(fmt: str) -> int:
    """Per-expert row count alignment for ``moe_w16a16_group_gemm``.

    ``fmt`` is ``TN`` (transposed weight, ``(E, n, k)``) or ``NN`` (``(E, k, n)``). Both have
    ``N`` at index 1, so checking ``fmt[1]`` would always yield no padding. The TN path needs
    M aligned for the ixformer/cuinfer kernel; ``NN`` does not.
    """
    if len(fmt) < 2:
        return 1
    return 64 if fmt[0] == "T" else 1


def _pad_tokens_per_expert(tokens: torch.Tensor, fmt: str) -> torch.Tensor:
    align = _ixf_align_size(fmt)
    if align == 1:
        return tokens
    return (tokens + align - 1) // align * align


class IxformerGroupGemm(MojoGroupGemm):
    """Maps :class:`MojoGroupGemm` to ixformer ``moe_w16a16_group_gemm`` (W16A16 MoE-style grouped matmul)."""

    supported_platforms_list = ["ilu"]

    def forward(self, input: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"IxformerGroupGemm only supports fp16/bf16 input, got {input.dtype}."
            )

        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"

        weight = self.weight

        num_groups = int(group_list.numel())
        assert weight.size(0) == num_groups, "weight group count must match group_list length"

        m, _ = input.shape
        tokens = group_list.to(dtype=torch.int32, device="cpu").flatten()
        assert int(tokens.sum().item()) == m, "sum(group_list) must equal number of input rows"

        fmt = "TN" if self.trans_weight else "NN"
        align = _ixf_align_size(fmt)
        if align != 1 and bool((tokens % align).any().item()):
            raise NotImplementedError(
                f"IxformerGroupGemm requires per-expert token counts aligned to {align} for fmt={fmt}."
            )
        padded = tokens

        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        if self.trans_weight:
            # Mojo (G, N, K) == ixformer TN (E, n, k)
            weight_ix = weight
        else:
            # Mojo (G, K, N) == ixformer NN (E, k, n)
            weight_ix = weight

        try:
            out_full = ixf_f.moe_w16a16_group_gemm(
                input,
                weight_ix,
                input.dtype,
                padded,
                None,
                None,
                fmt,
            )
        except RuntimeError as e:
            raise NotImplementedError(
                "IxformerGroupGemm hit an unsupported case for ixformer backend."
            ) from e

        return out_full
