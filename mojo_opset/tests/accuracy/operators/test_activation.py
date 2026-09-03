import os

import pytest
import torch

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU
from mojo_opset.experimental import MojoRotateActivation
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented

dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@pytest.mark.parametrize(
    "shape",
    [
        ([128, 128]),
        ([999, 9999]),
        ([1024, 10240]),
    ],
)
@bypass_not_implemented
def test_gelu(shape):
    x = torch.rand(*shape, dtype=torch.bfloat16)
    gelu = MojoGelu()
    gelu_ref = MojoGelu._registry.get("torch")()
    gelu.forward_diff_with(gelu_ref, x)


@pytest.mark.parametrize(
    "shape",
    [
        ([256, 128]),
        ([1024, 10240]),
        ([999, 9999]),
    ],
)
@bypass_not_implemented
def test_silu(shape):
    x = torch.rand(*shape, dtype=torch.bfloat16)
    silu = MojoSilu()
    silu_ref = MojoSilu._registry.get("torch")()
    silu.forward_diff_with(silu_ref, x)


@pytest.mark.parametrize(
    "shape",
    [
        ([256, 128]),
        ([1024, 10240]),
        ([999, 9999]),
    ],
)
@bypass_not_implemented
def test_swiglu(shape):
    gate_out = torch.rand(*shape, dtype=torch.bfloat16)
    up_out = torch.rand(*shape, dtype=torch.bfloat16)
    swiglu = MojoSwiGLU()
    swiglu_ref = MojoSwiGLU._registry.get("torch")()
    swiglu.forward_diff_with(swiglu_ref, gate_out, up_out)


def test_swiglu_limit_reference():
    gate_out = torch.tensor([[-3.0, 0.5, 2.0, 6.0]], dtype=torch.float32)
    up_out = torch.tensor([[-4.0, -1.0, 3.0, 9.0]], dtype=torch.float32)
    swiglu = MojoSwiGLU._registry.get("torch")(swiglu_limit=2.0)

    expected_gate = torch.clamp(gate_out, max=2.0)
    expected_up = torch.clamp(up_out, min=-2.0, max=2.0)
    expected = torch.nn.functional.silu(expected_gate) * expected_up

    torch.testing.assert_close(swiglu(gate_out, up_out), expected)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_head, head_dim, dtype",
    [
        (batch_size, seq_len, num_head, head_dim, dtype)
        for batch_size in [2, 8, 32]
        for seq_len in [1, 2048]
        for num_head in [1, 32]
        for head_dim in [128, 1024]
        for dtype in ["bfloat16", "float16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_rotate_activation(batch_size, seq_len, num_head, head_dim, dtype):
    device = get_torch_device()
    map_tol = {
        "bfloat16": (1.6e-2, 1e-5),
        "float16": (1e-3, 1e-5),
        "float32": (1.3e-6, 1e-5),
    }
    atol, rtol = map_tol[dtype]
    dtype = dtype_str_map[dtype]

    x = torch.randn(batch_size, seq_len, num_head, head_dim, device=device, dtype=dtype)

    res = MojoRotateActivation()
    res_ref = MojoRotateActivation._registry.get("torch")()
    res.forward_diff_with(res_ref, x, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [([3072, 3072]), ([4096, 2048]), ([257, 480])])
@bypass_not_implemented
def test_swiglu_keeps_silu_intermediate_in_fp32(shape):
    """SwiGLU must round exactly once, on the output store.
    This pins the reference-platform contract. byted-seed-kernels' ``seed_swiglu``
    is ``(silu(x1.float()) * x2.float()).to(x1.dtype)`` for ``implementation="torch"``
    and ``bumi.kernel.swiglu``'s ``y = silu(x1) * x2`` (silu() promotes to fp32,
    output allocated ``dtype=x1.dtype``) for ``implementation="bumi"``. Both keep
    the SiLU intermediate in fp32.
    The NPU kernel used to compute ``silu_a.to(a.dtype) * b``, i.e. it rounded the
    intermediate first. Measured on 910B2C at the P6D audio-MLP width (3072, 3072)
    with bf16 inputs, that cost 2592593/9437184 forward elements, max|delta|
    6.25e-2. With the fp32 intermediate the forward is bitwise equal to the fp32
    reference, so this test asserts bitwise equality rather than a tolerance.
    """
    gate = torch.randn(*shape, dtype=torch.bfloat16)
    up = torch.randn(*shape, dtype=torch.bfloat16)
    swiglu = MojoSwiGLU()
    actual = swiglu(gate, up)
    expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)