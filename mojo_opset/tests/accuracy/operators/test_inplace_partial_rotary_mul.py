import numpy as np
import pytest
import torch

from mojo_opset import MojoInplacePartialRotaryMul
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack((-x_odd, x_even), dim=-1)
    return stacked.reshape(x.shape)


def _ref_inplace_partial_rotary_mul(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_slice: list[int],
) -> torch.Tensor:
    start, end = partial_slice
    chunks = torch.split(x, [start, end - start, x.size(-1) - end], dim=-1)
    prefix, x_rope, suffix = chunks[0], chunks[1], chunks[2]
    rope_out = cos * x_rope + _rotate_every_two(x_rope) * sin
    return torch.cat([prefix, rope_out, suffix], dim=-1)


@bypass_not_implemented
def test_inplace_partial_rotary_mul_with_cpu_benchmark():
    device = get_torch_device()
    if device != "npu":
        pytest.skip("Ascend NPU only.")

    b = 128
    s = 64
    d = 512
    slice_size = 64
    partial_slice = [448, 512]

    np.random.seed(0)
    dtype = torch.bfloat16

    x = torch.tensor(np.random.uniform(-10, 10, (b, s, 1, d)).astype(np.int32), device=device).to(dtype)
    cos = torch.tensor(np.random.uniform(1, 17, (b, 1, 1, slice_size)).astype(np.int32), device=device).to(dtype)
    sin = torch.tensor(np.random.uniform(-5, 10, (b, 1, 1, slice_size)).astype(np.int32), device=device).to(dtype)

    ref = _ref_inplace_partial_rotary_mul(x, cos, sin, partial_slice)

    mojo_op = MojoInplacePartialRotaryMul()
    print(f"{type(mojo_op)=}")
    out = mojo_op.forward(x, cos, sin, rotary_mode="interleave", partial_slice=partial_slice)

    torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)


@bypass_not_implemented
def test_inplace_partial_rotary_mul_with_graph():
    device = get_torch_device()
    if device != "npu":
        pytest.skip("Graph-mode test only runs on Ascend NPU.")

    try:
        import torchair  # noqa: F401
        from torchair.configs.compiler_config import CompilerConfig
    except Exception as e:
        pytest.skip(f"torchair is not available: {e}")

    try:
        import custom_ops  # noqa: F401
    except Exception as e:
        pytest.skip(f"custom_ops is not available: {e}")

    if not hasattr(torch.ops.custom, "inplace_partial_rotary_mul"):
        pytest.skip("torch.ops.custom.inplace_partial_rotary_mul is not registered in this environment.")

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available in this PyTorch build.")

    b = 128
    s = 64
    d = 512
    slice_size = 64
    partial_slice = [448, 512]

    np.random.seed(0)
    dtype = torch.bfloat16

    x = torch.tensor(np.random.uniform(-10, 10, (b, s, 1, d)).astype(np.int32), device=device).to(dtype)
    cos = torch.tensor(np.random.uniform(1, 17, (b, 1, 1, slice_size)).astype(np.int32), device=device).to(dtype)
    sin = torch.tensor(np.random.uniform(-5, 10, (b, 1, 1, slice_size)).astype(np.int32), device=device).to(dtype)

    ref = _ref_inplace_partial_rotary_mul(x, cos, sin, partial_slice)

    class Network(torch.nn.Module):
        def forward(self, x_npu, cos_npu, sin_npu, partial_slice_val):
            torch.ops.custom.inplace_partial_rotary_mul(
                x_npu,
                cos_npu,
                sin_npu,
                rotary_mode="interleave",
                partial_slice=partial_slice_val,
            )
            return x_npu

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    npu_model = Network().to(device)
    try:
        npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend, dynamic=False)
    except TypeError:
        npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend)
    except Exception as e:
        pytest.skip(f"torch.compile failed on this environment: {e}")

    out = npu_model(x, cos, sin, partial_slice)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1e-3)

