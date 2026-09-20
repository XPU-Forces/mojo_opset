from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules as M
from tests._checks import assert_close

MOE_CASES = [
    (8, 128, [8]),
    (12, 256, [4, 3, 5]),
    (18, 512, [6, 6, 4, 2]),
    (21, 1024, [2, 5, 1, 7, 6]),
    (32, 2048, [8, 7, 5, 6, 4, 2]),
]


def check_grouped(actual, expected, *, swiglu=False):
    q_atol, scale_tol = (0, 1e-4) if swiglu else (1, 2e-3)
    assert_close(actual[0], expected[0], torch.float32, name="quantized", rtol=0, atol=q_atol)
    assert_close(actual[1], expected[1], torch.float32, name="scale", rtol=scale_tol, atol=scale_tol)


def moe_case(case, dtype, device, impl):
    tokens, width, counts = case
    x = torch.randn(tokens, width, dtype=dtype, device=device)
    token_count = torch.tensor(counts, dtype=torch.int32, device=device)
    inv_smooth_scale = (torch.rand(len(counts), width, device=device) + 0.1).reciprocal()
    options = dict(inv_smooth_scale=inv_smooth_scale)
    call = M.MoEDynamicQuant(len(counts), width, implementation=impl, device=device)
    call.load_state_dict(options)
    reference = partial(F.moe_dynamic_quant, **options, implementation="torch_reference")
    return (call, reference, (x, token_count))


@pytest.mark.api("modules.MoEDynamicQuant", ops=["moe_dynamic_quant"])
@pytest.mark.accuracy
@pytest.mark.parametrize("case", MOE_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cpu_counts", [False, True], ids=["device_counts", "cpu_counts"])
def test_moe(accuracy_backend, case, dtype, cpu_counts):
    impl, _, device = accuracy_backend
    call, reference, inputs = moe_case(case, dtype, device, impl)
    if cpu_counts:
        inputs = (inputs[0], inputs[1].cpu())
    check_grouped(call(*inputs), reference(*inputs))
