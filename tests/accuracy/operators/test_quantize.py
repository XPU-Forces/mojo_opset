import random

import pytest
import torch

from mojo_opset import MojoQuantInt8
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform, bypass_not_implemented

dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@pytest.mark.parametrize(
    "batch, seq_len, head_dim, dtype",
    [
        (
            batch,
            seq_len,
            head_dim,
            dtype,
        )
        for batch in [1, 4]
        for seq_len in [256, 334, 512, 777, 973, 1024, 2048, 2233, 8192, 16384]
        for head_dim in [64, 128]
        for dtype in ["bfloat16", "float16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_quant_int8_3d(batch, seq_len, head_dim, dtype):
    device = get_platform()

    dtype = dtype_str_map[dtype]

    input_tensor = torch.randn(batch, seq_len, head_dim, dtype=dtype, device=device)
    # *  Only consider scale_tensor as 1.
    scale_tensor = torch.ones(head_dim, dtype=dtype, device=device)

    quant_func = MojoQuantInt8()

    quant_func_ref = quant_func._registry.get("torch")()
    quant_func.forward_diff_with(quant_func_ref, input_tensor, scale_tensor, mixed_tol=False, ptol=0.999)


@pytest.mark.parametrize(
    "batch, seq_len, num_head, head_dim, dtype",
    [
        (
            batch,
            seq_len,
            num_head,
            head_dim,
            dtype,
        )
        for batch in [4]
        for seq_len in [256, 512, 777, 1024, 2048, 8192, 16384]
        for num_head in [8]
        for head_dim in [128]
        for dtype in ["bfloat16", "float16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_quant_int8_4d(batch, seq_len, num_head, head_dim, dtype):
    device = get_platform()

    dtype = dtype_str_map[dtype]

    input_tensor = torch.randn(batch, seq_len, num_head, head_dim, dtype=dtype, device=device)
    # *  Only consider scale_tensor as 1.
    scale_tensor = torch.ones(head_dim, dtype=dtype, device=device)

    quant_func = MojoQuantInt8()

    quant_func_ref = quant_func._registry.get("torch")()
    quant_func.forward_diff_with(quant_func_ref, input_tensor, scale_tensor, mixed_tol=False, ptol=0.999)
