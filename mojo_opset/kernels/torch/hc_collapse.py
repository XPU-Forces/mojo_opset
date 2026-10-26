# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for Ascend NPU by the mojo_opset contributors.
import torch


def hc_collapse_fwd(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    # Keep the FP32 multiply and sum separate; the Triton kernels retain the
    # same math so near-midpoint BF16 rounding matches reference semantics.
    return (pre_mix.unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)
