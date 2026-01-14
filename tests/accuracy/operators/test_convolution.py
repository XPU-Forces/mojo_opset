from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented
from tests.utils import assert_close

from mojo_opset import MojoCausalConv1dUpdateState


@pytest.mark.parametrize(
    "hidden_states, conv_state, weight, bias, activation",
    [
        (
            torch.randn(B, D, T, dtype=torch.float16),
            torch.randn(B, D, W, dtype=torch.float16),
            torch.randn(D, W, dtype=torch.float16),
            None,
            act
        ) for (B, T, D, W, act) in [
            (1, 12291, 8192, 4, "swish"),
            (1, 5000, 2048, 4, "swish"),

            (2, 64, 128, 3, "swish"),
            (2, 128, 128, 4, "swish"),
            (2, 64, 128, 3, None),
            (3, 1446, 256, 4, None),

            (1, 32, 32, 4, None),
        ]
    ]
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_causal_conv1d_update_state(hidden_states, conv_state, weight, bias, activation):
    causal_conv1d = MojoCausalConv1dUpdateState()
    causal_conv1d_ref = MojoCausalConv1dUpdateState._registry.get("torch")()
    conv_state_ref = conv_state.clone()
    out = causal_conv1d(hidden_states, conv_state, weight, bias, activation)
    out_ref = causal_conv1d_ref(hidden_states, conv_state_ref, weight, bias, activation)
    print(out.shape, out_ref.shape)
    assert_close(out, out_ref)
    assert_close(conv_state, conv_state_ref)
    
    