import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoCausalConv1dFunction


@pytest.mark.parametrize(
    ("B", "T", "D", "W", "activation", "has_bias", "has_residual", "dtype"),
    [
        pytest.param(
            *test,
            id="B{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test),
        )
        for test in [
            (2, 64, 1024, 3, "swish", True, True, torch.float16),
            (2, 128, 8192, 4, "swish", False, True, torch.float16),
            (2, 64, 8192, 3, "swish", True, False, torch.float16),
            (2, 128, 4096, 4, "swish", False, False, torch.float16),
            (2, 64, 8192, 3, None, True, False, torch.float16),
            (3, 1446, 8192, 4, None, False, False, torch.float16),
        ]
    ],
)
@bypass_not_implemented
def test_conv(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    x = torch.rand(B, T, D).to(dtype).requires_grad_(True)
    weight = torch.rand(D, W).to(dtype).requires_grad_(True)
    bias = torch.rand(D).to(dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.rand(B, T, D).to(dtype)

    ctx = MockFunctionCtx()
    y, _ = MojoCausalConv1dFunction.forward(ctx, x, weight, bias, residual, None, False, activation)

    ctx_ref = MockFunctionCtx()
    y_ref, _ = MojoCausalConv1dFunction._registry.get("torch").forward(
        ctx_ref, x, weight, bias, residual, None, False, activation
    )
    assert_close(y, y_ref)
    dx, dw, db, dr, d_init, _, _, _ = MojoCausalConv1dFunction.backward(ctx, dy)
    dx_ref, dw_ref, db_ref, dr_ref, d_init_ref, _, _, _ = MojoCausalConv1dFunction._registry.get("torch").backward(
        ctx_ref, dy
    )
    assert_close(dx, dx_ref)
    assert_close(dw, dw_ref)
    if has_bias:
        assert_close(db, db_ref)
    if has_residual:
        assert_close(dr, dr_ref)


@pytest.mark.parametrize(
    ("N", "T", "D", "W", "activation", "has_bias", "has_residual", "dtype"),
    [
        pytest.param(*test, id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test))
        for test in [
            (4, 500, 1024, 3, "silu", True, False, torch.float16),
            (3, 1024, 256, 4, "silu", False, False, torch.float16),
            (4, 500, 1024, 3, None, True, False, torch.float16),
            (4, 1024, 1024, 4, None, False, False, torch.float16),
            (5, 8192, 8192, 4, None, False, False, torch.float32),
            (3, 7666, 8192, 4, None, False, False, torch.float32),
            (5, 12291, 8192, 4, None, False, False, torch.bfloat16),
            (5, 12291, 8192, 4, "silu", False, False, torch.bfloat16),
            (6, 11357, 8192, 4, None, False, False, torch.bfloat16),
            (9, 10287, 8192, 4, "silu", False, False, torch.bfloat16),
        ]
    ],
)
@bypass_not_implemented
def test_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(41)

    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.long),
            torch.arange(16, T)[torch.randperm(T - 16)[: N - 1]],
            torch.tensor([T], dtype=torch.long),
        ],
        0,
    ).sort()[0]

    x = torch.rand(1, T, D).to(dtype).requires_grad_(True)
    weight = torch.rand(D, W).to(dtype).requires_grad_(True)
    bias = torch.rand(D).to(dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.rand(1, T, D).to(dtype)

    ctx = MockFunctionCtx()
    y, _ = MojoCausalConv1dFunction.forward(ctx, x, weight, bias, residual, None, False, activation, cu_seqlens)

    ctx_ref = MockFunctionCtx()
    y_ref, _ = MojoCausalConv1dFunction._registry.get("torch").forward(
        ctx_ref, x, weight, bias, residual, None, False, activation, cu_seqlens
    )
    assert_close(y, y_ref)
    dx, dw, db, dr, d_init, _, _, _ = MojoCausalConv1dFunction.backward(ctx, dy)
    dx_ref, dw_ref, db_ref, dr_ref, d_init_ref, _, _, _ = MojoCausalConv1dFunction._registry.get("torch").backward(
        ctx_ref, dy
    )
    assert_close(dx, dx_ref)
    assert_close(dw, dw_ref)
    if has_bias:
        assert_close(db, db_ref)
    if has_residual:
        assert_close(dr, dr_ref)
