import pytest
import torch

from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoCausalConv1dUpdateState


@pytest.mark.parametrize(
    "B, T, D, W, act",
    [
        (1, 12291, 8192, 4, "swish"),
        (1, 5000, 2048, 4, "swish"),
        (2, 64, 128, 3, "swish"),
        (2, 128, 128, 4, "swish"),
        (2, 64, 128, 3, None),
        (3, 1446, 256, 4, None),
        (1, 32, 32, 4, None),
    ],
)
@bypass_not_implemented
def test_causal_conv1d_update_state(B, T, D, W, act):
    hidden_states = torch.randn(B, D, T, dtype=torch.float16)
    conv_state = torch.randn(B, D, W, dtype=torch.float16)
    weight = torch.randn(D, W, dtype=torch.float16)
    bias = None
    causal_conv1d = MojoCausalConv1dUpdateState()
    causal_conv1d_ref = MojoCausalConv1dUpdateState._registry.get("torch")()
    conv_state_ref = conv_state.clone()
    out = causal_conv1d(hidden_states, conv_state, weight, bias, act)
    out_ref = causal_conv1d_ref(hidden_states, conv_state_ref, weight, bias, act)

    assert_close(out, out_ref)
    assert_close(conv_state, conv_state_ref)
