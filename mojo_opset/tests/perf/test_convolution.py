
import pytest
import torch

from mojo_opset import MojoCausalConv1dUpdateState
from mojo_opset import MojoConv1d
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


@pytest.mark.parametrize(
    "hidden_states, conv_state, weight, bias, activation",
    [
        (
            torch.randn(B, D, T, dtype=torch.float16),
            torch.randn(B, D, W, dtype=torch.float16),
            torch.randn(D, W, dtype=torch.float16),
            None,
            act,
        )
        for (B, T, D, W, act) in [
            (1, 12291, 8192, 4, "swish"),
            (1, 5000, 2048, 4, "swish"),
            (2, 64, 128, 3, "swish"),
            (2, 128, 128, 4, "swish"),
            (2, 64, 128, 3, None),
            (3, 1446, 256, 4, None),
            (1, 32, 32, 4, None),
        ]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_causal_conv1d_update_state(hidden_states, conv_state, weight, bias, activation):
    causal_conv1d = MojoCausalConv1dUpdateState()
    causal_conv1d_ref = MojoCausalConv1dUpdateState._registry.get("torch")()
    conv_state_ref = conv_state.clone()
    perf(lambda: causal_conv1d(hidden_states, conv_state, weight, bias, activation))  # noqa: F821
    perf(lambda: causal_conv1d_ref(hidden_states, conv_state_ref, weight, bias, activation))  # noqa: F821


@pytest.mark.parametrize(
    "hidden_states, weight, bias, stride, padding, dilation, groups",
    [
        (
            torch.randn(B, C_in, T, dtype=dtype),
            torch.randn(C_out, C_in // resolved_groups, K, dtype=dtype),
            torch.randn(C_out, dtype=dtype) if use_bias else None,
            stride,
            padding,
            dilation,
            groups,
        )
        for (B, C_in, C_out, T, K, stride, padding, dilation, groups, use_bias, dtype) in [
            (2, 256, 256, 512, 5, 1, "same", 1, None, True, torch.float16),
            (2, 512, 768, 1024, 7, 1, 3, 1, 1, True, torch.bfloat16),
            (1, 128, 128, 2048, 31, 1, "same", 1, None, False, torch.bfloat16),
        ]
        for resolved_groups in [C_in if groups is None else groups]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_conv1d(hidden_states, weight, bias, stride, padding, dilation, groups):
    op = MojoConv1d(stride=stride, padding=padding, dilation=dilation, groups=groups)
    op_ref = MojoConv1d._registry.get("torch")(stride=stride, padding=padding, dilation=dilation, groups=groups)
    perf(lambda: op(hidden_states, weight, bias))  # noqa: F821
    perf(lambda: op_ref(hidden_states, weight, bias))  # noqa: F821
