# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for Ascend NPU by the mojo_opset contributors.
import torch


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    # Keep the FP32 einsum, post product and final add separate; the Triton
    # kernels retain the same math so near-midpoint BF16 rounding matches
    # reference semantics.
    mixed_residual = torch.einsum(
        "tij,tih->tjh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)
