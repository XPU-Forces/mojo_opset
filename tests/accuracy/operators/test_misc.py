import random

import pytest
import torch

from mojo_opset import MojoQuantInt8
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform, bypass_not_implemented

dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@pytest.mark.parametrize(
    "batch, seq_len, head_dim, dtype,",
    [
        (
            2,
            1024,
            64,
            dtype,
        )
        for dtype in ["bfloat16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_quant_int8(batch, seq_len, head_dim, dtype):
    device = get_platform()

    dtype = dtype_str_map[dtype]

    input_tensor = torch.randn(batch, seq_len, head_dim, dtype=dtype, device=device)
    # *  Only consider scale_tensor as 1.
    scale_tensor = torch.ones(head_dim, dtype=dtype, device=device)

    quant_func = MojoQuantInt8()

    quant_func_ref = quant_func._registry.get("torch")()
    quant_func.forward_diff_with(
        quant_func_ref, input_tensor, scale_tensor, mixed_tol=False, ptol=0.999
    )
