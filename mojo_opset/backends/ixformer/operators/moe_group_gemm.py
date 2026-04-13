"""Grouped GEMM via ixformer ``moe_w16a16_group_gemm``."""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from mojo_opset.backends.ixformer.utils import _get_ixf_and_check_device
from mojo_opset.core import MojoGroupGemm


def _ixf_align_size(fmt: str) -> int:
    return 1 if fmt[1] == "N" else 64


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
            return super().forward(input, group_list)

        ixf_f = _get_ixf_and_check_device(input, self.__class__.__name__)

        assert input.dim() == 2, "input must be 2D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        num_groups = int(group_list.numel())
        assert weight.size(0) == num_groups, "weight group count must match group_list length"

        m, k = input.shape
        tokens = group_list.to(dtype=torch.int32, device="cpu").flatten()
        assert int(tokens.sum().item()) == m, "sum(group_list) must equal number of input rows"

        fmt = "TN" if self.trans_weight else "NN"
        padded = _pad_tokens_per_expert(tokens, fmt)
        pad_m = int(padded.sum().item())

        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        if pad_m != m:
            input_pad = torch.zeros((pad_m, k), dtype=input.dtype, device=input.device)
            in_off = 0
            pad_off = 0
            for i in range(num_groups):
                ti = int(tokens[i].item())
                pi = int(padded[i].item())
                if ti:
                    input_pad[pad_off : pad_off + ti] = input[in_off : in_off + ti]
                in_off += ti
                pad_off += pi
            input_ix = input_pad
        else:
            input_ix = input

        if self.trans_weight:
            # Mojo (G, N, K) == ixformer TN (E, n, k)
            weight_ix = weight
        else:
            # Mojo (G, K, N) == ixformer NN (E, k, n)
            weight_ix = weight

        try:
            out_full = ixf_f.moe_w16a16_group_gemm(
                input_ix,
                weight_ix,
                input.dtype,
                padded,
                None,
                None,
                fmt,
            )
        except RuntimeError:  # pragma: no cover - cuinfer layout / alignment constraints
            return super().forward(input, group_list)

        if pad_m == m:
            return out_full

        n_out = out_full.shape[1]
        out = input.new_empty((m, n_out))
        pad_off = 0
        out_off = 0
        for i in range(num_groups):
            ti = int(tokens[i].item())
            pi = int(padded[i].item())
            if ti:
                out[out_off : out_off + ti] = out_full[pad_off : pad_off + ti]
            pad_off += pi
            out_off += ti
        return out
