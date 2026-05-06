import pytest
import torch

from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoCausalConv1dUpdateState
from mojo_opset import MojoConv1d


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


@pytest.mark.parametrize(
    "B, C_in, C_out, T, K, stride, padding, dilation, groups, use_bias, dtype",
    [
        (2, 8, 8, 32, 3, 1, "same", 1, None, True, torch.float32),
        (2, 16, 24, 17, 5, 1, 2, 1, 1, False, torch.float32),
        (1, 32, 32, 64, 7, 1, "same", 1, None, True, torch.float16),
        (3, 12, 18, 19, 4, 1, 3, 1, 3, True, torch.bfloat16),
    ],
)
@bypass_not_implemented
def test_conv1d(B, C_in, C_out, T, K, stride, padding, dilation, groups, use_bias, dtype):
    resolved_groups = C_in if groups is None else groups
    hidden_states = torch.randn(B, C_in, T, dtype=dtype)
    weight = torch.randn(C_out, C_in // resolved_groups, K, dtype=dtype)
    bias = torch.randn(C_out, dtype=dtype) if use_bias else None

    op = MojoConv1d(stride=stride, padding=padding, dilation=dilation, groups=groups)
    op_ref = MojoConv1d._registry.get("torch")(stride=stride, padding=padding, dilation=dilation, groups=groups)

    out = op(hidden_states, weight, bias)
    out_ref = op_ref(hidden_states, weight, bias)

    assert_close(out, out_ref)


def test_conv1d_matches_torch_conv1d():
    hidden_states = torch.randn(2, 4, 11, dtype=torch.float32)
    weight = torch.randn(4, 1, 5, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)

    op = MojoConv1d._registry.get("torch")(stride=1, padding="same", dilation=1, groups=None)
    out = op(hidden_states, weight, bias)
    ref = torch.nn.functional.conv1d(
        hidden_states,
        weight,
        bias,
        stride=1,
        padding="same",
        dilation=1,
        groups=hidden_states.shape[1],
    )
    assert_close(out, ref)
