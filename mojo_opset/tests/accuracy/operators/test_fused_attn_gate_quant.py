"""Tests for MojoFusedAttnGateQuant: fused attention gate + dynamic quantization.

Compares the platform-dispatched backend against the torch reference backend.

Run:
    python -m pytest mojo_opset/tests/accuracy/operators/test_fused_attn_gate_quant.py -v
"""

import pytest
import torch
import torch.nn as nn

from mojo_opset.experimental import MojoFusedAttnGateQuant
from mojo_opset.tests.utils import bypass_not_implemented

torch.manual_seed(42)

DTYPES = [torch.bfloat16, torch.float16]

SHAPES = [
    # (seq_len, hidden_size, num_heads_full, num_heads_swa, head_dim)
    (1, 4096, 32, 8, 128),
    (7, 4096, 32, 8, 128),
    (13, 4096, 32, 8, 128),
    (128, 4096, 32, 8, 128),
    (255, 4096, 32, 8, 128),
    (1000, 4096, 32, 8, 128),
    (4096, 4096, 32, 8, 128),
    (8192, 4096, 32, 8, 128),
    (1, 2048, 16, 4, 128),
    (37, 2048, 16, 4, 128),
    (4096, 2048, 16, 4, 128),
    (1, 6144, 48, 16, 128),
    (63, 6144, 48, 16, 128),
    (1, 3072, 32, 8, 96),
    (11, 3072, 32, 8, 96),
    (128, 3072, 32, 8, 96),
    (999, 3072, 32, 8, 96),
    (4096, 3072, 32, 8, 96),
    (1, 1536, 16, 4, 96),
    (53, 1536, 16, 4, 96),
    (8191, 1536, 16, 4, 96),
]


@pytest.mark.parametrize("seq_len, hidden_size, num_heads_full, num_heads_swa, head_dim", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [False, True])
@bypass_not_implemented
def test_fused_attn_gate_quant(seq_len, hidden_size, num_heads_full, num_heads_swa, head_dim, dtype, bias):
    """forward_diff_with: op vs torch reference backend."""
    op = MojoFusedAttnGateQuant(
        hidden_size=hidden_size,
        num_heads_full=num_heads_full,
        num_heads_swa=num_heads_swa,
        head_dim=head_dim,
        bias=bias,
        dtype=dtype,
    )
    op_ref = MojoFusedAttnGateQuant._registry.get("torch")(
        hidden_size=hidden_size,
        num_heads_full=num_heads_full,
        num_heads_swa=num_heads_swa,
        head_dim=head_dim,
        bias=bias,
        dtype=dtype,
    )

    for name, p in op_ref.named_parameters():
        if name != "o_quantize.inv_smooth_scale":
            nn.init.normal_(p, std=0.02)
    
    output_size = (num_heads_full + num_heads_swa) * head_dim
    smooth_scale = torch.rand(output_size, dtype=torch.float32) + 0.1
    inv_smooth_scale = 1.0 / smooth_scale
    state_dict = op_ref.state_dict()
    state_dict["o_quantize.inv_smooth_scale"] = inv_smooth_scale
    op_ref.load_state_dict(state_dict)
    op.load_state_dict(state_dict)
    assert op.attn_gate._cached_weight is not None

    hidden_states = torch.randn(seq_len, hidden_size, dtype=dtype)
    full_attn_output = torch.randn(seq_len, num_heads_full, head_dim, dtype=dtype)
    swa_attn_output = torch.randn(seq_len, num_heads_swa, head_dim, dtype=dtype)

    op.forward_diff_with(op_ref, hidden_states, full_attn_output, swa_attn_output, atol=(1, 1e-3), rtol=(0, 1e-3))
