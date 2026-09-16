import pytest
import torch

from mojo_opset import functions
from mojo_opset import modules
from tests._checks import assert_close


@pytest.mark.api("modules.CausalConv1dUpdateStateInfer")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_update(accuracy_backend, dtype):
    implementation, _, device = accuracy_backend
    x = torch.randn(2, 64, 3, device=device, dtype=dtype)
    state = torch.randn(2, 64, 4, device=device, dtype=dtype)
    reference_state = state.clone()
    weight = torch.randn(64, 4, device=device, dtype=dtype)
    actual = modules.CausalConv1dUpdateStateInfer(implementation=implementation)(x, state, weight)
    expected = functions.causal_conv1d_update_state_infer(x, reference_state, weight, implementation="torch_reference")
    assert_close(actual, expected, dtype)
    assert_close(state, reference_state, rtol=0, atol=0)
