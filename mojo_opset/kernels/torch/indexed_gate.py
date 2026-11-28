# SPDX-License-Identifier: Apache-2.0
"""Independent PyTorch implementation of indexed residual gate."""

import torch


def indexed_gate_infer_fwd(x, gate, other, indices):
    if x.numel() == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    return (x.float() + gate.index_select(0, indices).float() * other.float()).to(x.dtype).contiguous()
