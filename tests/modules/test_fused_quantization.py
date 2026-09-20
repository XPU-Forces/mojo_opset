from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as M
from tests._checks import assert_close

SWIGLU_CASES = [(12, 64, [4, 3, 5]), (20, 128, [6, 4, 7, 3]), (24, 256, [5, 8, 4, 7]), (30, 512, [6, 3, 8, 5, 8])]


def check_grouped(actual, expected, *, swiglu=False):
    q_atol, scale_tol = (0, 1e-4) if swiglu else (1, 2e-3)
    assert_close(actual[0], expected[0], torch.float32, name="quantized", rtol=0, atol=q_atol)
    assert_close(actual[1], expected[1], torch.float32, name="scale", rtol=scale_tol, atol=scale_tol)


def swiglu_case(case, device, impl, *, activate_left=False, with_bias=False):
    tokens, width, counts = case
    x = torch.randint(-1024, 1024, (tokens, width * 2), dtype=torch.int32, device=device)
    token_count = torch.tensor(counts, dtype=torch.int64, device=device)
    activation_scale = torch.rand(tokens, device=device)
    weight_scale = torch.rand(len(counts), width * 2, device=device)
    quant_scale = torch.rand(len(counts), width, device=device)
    bias = torch.randn(len(counts), width * 2, device=device) if with_bias else None
    params = dict(weight_scale=weight_scale, quant_scale=quant_scale)
    options = dict(activate_left=activate_left)
    call = M.DequantSwiGLUQuant(len(counts), width, **options, implementation=impl, device=device)
    call.load_state_dict(params)

    def reference(x, activation_scale, bias, quant_offset, token_count):
        return F.dequant_swiglu_quant(
            x,
            **params,
            activation_scale=activation_scale,
            bias=bias,
            quant_offset=quant_offset,
            token_count=token_count,
            **options,
            implementation="torch_reference",
        )

    return (
        partial(call, bias=bias, quant_offset=None, token_count=token_count),
        partial(reference, bias=bias, quant_offset=None, token_count=token_count),
        (x, activation_scale),
    )


@pytest.mark.api("modules.DequantSwiGLUQuant", ops=["dequant_swiglu_quant"])
@pytest.mark.accuracy
@pytest.mark.parametrize("case", SWIGLU_CASES)
def test_swiglu(accuracy_backend, case):
    impl, _, device = accuracy_backend
    call, reference, inputs = swiglu_case(case, device, impl)
    check_grouped(call(*inputs), reference(*inputs), swiglu=True)
