import pytest
import torch
import os

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU
from mojo_opset import MojoIndexerRotateActivation

dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


@pytest.mark.parametrize(
    "x",
    [
        (torch.rand(128, 128, dtype=torch.bfloat16)),
        (torch.rand(999, 9999, dtype=torch.bfloat16)),
        (torch.rand(1024, 10240, dtype=torch.bfloat16)),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_gelu(x):
    gelu = MojoGelu()
    gelu_ref = MojoGelu._registry.get("torch")()
    gelu.forward_diff_with(gelu_ref, x)


@pytest.mark.parametrize(
    "x",
    [
        (torch.rand(128, 128, dtype=torch.bfloat16)),
        (torch.rand(999, 9999, dtype=torch.bfloat16)),
        (torch.rand(1024, 10240, dtype=torch.bfloat16)),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_silu(x):
    silu = MojoSilu()
    silu_ref = MojoSilu._registry.get("torch")()
    silu.forward_diff_with(silu_ref, x)


@pytest.mark.parametrize(
    "gate_out, up_out",
    [
        (
            torch.rand(size=(256, 128), dtype=torch.bfloat16),
            torch.rand(size=(256, 128), dtype=torch.bfloat16),
        ),
        (
            torch.rand(size=(1024, 10240), dtype=torch.bfloat16),
            torch.rand(size=(1024, 10240), dtype=torch.bfloat16),
        ),
        (
            torch.rand(size=(999, 9999), dtype=torch.bfloat16),
            torch.rand(size=(999, 9999), dtype=torch.bfloat16),
        ),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_swiglu(gate_out, up_out):
    swiglu = MojoSwiGLU()
    swiglu_ref = MojoSwiGLU._registry.get("torch")()
    swiglu.forward_diff_with(swiglu_ref, gate_out, up_out)


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
def test_indexer_rotate_activation(batch_size, seq_len, num_head, head_dim, dtype):
    device = get_platform()
    map_tol = {
        "bfloat16": (1.6e-2, 1e-5),
        "float16": (1e-3, 1e-5),
        "float32": (1.3e-6, 1e-5),
    }
    if device == 'npu':
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
    atol, rtol = map_tol[dtype]
    dtype = dtype_str_map[dtype]

    # create input tensor
    x = torch.randn(batch_size, seq_len, num_head, head_dim, device=device, dtype=dtype)

    res = MojoIndexerRotateActivation()
    res_ref = MojoIndexerRotateActivation._registry.get("torch")()
    res.forward_diff_with(res_ref, x, atol=atol, rtol=rtol)


if __name__ == "__main__":

    # pytest.main(["-s", "-v", "tests/accuracy/operators/test_activation.py::test_indexer_rotate_activation[32-2048-32-1024-float32]"])
    pytest.main(["-s", "-v", "tests/accuracy/operators/test_activation.py::test_indexer_rotate_activation"])
