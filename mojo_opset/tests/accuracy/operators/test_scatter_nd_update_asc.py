import numpy as np
import pytest
import torch

from mojo_opset import MojoScatterNdUpdateAsc
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _scatter_nd_update_asc_cpu(var: np.ndarray, indices: np.ndarray, update: np.ndarray) -> np.ndarray:
    var = var.copy()
    u = indices.shape[0]
    for i in range(u):
        j = int(indices[i][0])
        if j >= 0:
            var[j, :] = update[i, :]
    return var


def _make_case(a: int, b: int, c: int, var_dtype: str, indices_dtype: str):
    np_dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float16,  # upstream uses fp16 to simulate bf16 on numpy side
        "int8": np.int8,
        "int32": np.int32,
        "int64": np.int64,
    }
    torch_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.float16,  # match upstream test mapping
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    np_var_dtype = np_dtype_map[var_dtype]
    torch_var_dtype = torch_dtype_map[var_dtype]
    np_indices_dtype = np_dtype_map[indices_dtype]
    torch_indices_dtype = torch_dtype_map[indices_dtype]

    var = np.random.uniform(-10, 10, (a, b)).astype(np_var_dtype)
    indices = np.random.choice(a, size=c, replace=False).reshape(c, 1).astype(np_indices_dtype)
    update = np.random.uniform(20, 40, (c, b)).astype(np_var_dtype)

    var_t = torch.tensor(var, dtype=torch_var_dtype)
    indices_t = torch.tensor(indices, dtype=torch_indices_dtype)
    update_t = torch.tensor(update, dtype=torch_var_dtype)
    cpu_out = _scatter_nd_update_asc_cpu(var, indices, update).astype(np_var_dtype)
    return var_t, indices_t, update_t, cpu_out


@pytest.mark.parametrize(
    "a,b,c,var_dtype,indices_dtype",
    [
        (5088, 512, 128, "bfloat16", "int64"),
        (16512, 512, 128, "bfloat16", "int32"),
        (8656, 1, 128, "float16", "int32"),
        (8656, 128, 128, "int8", "int32"),
        (8656, 512, 128, "bfloat16", "int32"),
        (2304, 1, 2047, "float16", "int32"),
        (2304, 128, 2047, "int8", "int32"),
        (2304, 512, 2047, "bfloat16", "int32"),
        (256, 512, 63, "bfloat16", "int32"),
        (8448, 512, 8192, "bfloat16", "int64"),
    ],
)
@bypass_not_implemented
def test_scatter_nd_update_asc_single(a, b, c, var_dtype, indices_dtype):
    device = get_torch_device()
    if device != "npu":
        pytest.skip("Ascend NPU only.")

    np.random.seed(0)
    var, indices, update, cpu_out = _make_case(a, b, c, var_dtype, indices_dtype)
    var_npu = var.to(device)
    indices_npu = indices.to(device)
    update_npu = update.to(device)

    mojo_op = MojoScatterNdUpdateAsc()
    print(f"{type(mojo_op)=}")
    out = mojo_op.forward(var_npu, indices_npu, update_npu)
    assert out is var_npu

    npu_out = var_npu.cpu().numpy().astype(cpu_out.dtype)
    assert np.array_equal(npu_out, cpu_out)


@pytest.mark.parametrize(
    "a,b,c,var_dtype,indices_dtype",
    [
        (5088, 512, 128, "bfloat16", "int64"),
        (16512, 512, 128, "bfloat16", "int32"),
        (8656, 1, 128, "float16", "int32"),
        (8656, 128, 128, "int8", "int32"),
        (8656, 512, 128, "bfloat16", "int32"),
        (2304, 1, 2047, "float16", "int32"),
        (2304, 128, 2047, "int8", "int32"),
        (2304, 512, 2047, "bfloat16", "int32"),
        (256, 512, 63, "bfloat16", "int32"),
        (8448, 512, 8192, "bfloat16", "int64"),
    ],
)
@bypass_not_implemented
def test_scatter_nd_update_asc_graph(a, b, c, var_dtype, indices_dtype):
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

    if not hasattr(torch.ops.custom, "scatter_nd_update_asc"):
        pytest.skip("torch.ops.custom.scatter_nd_update_asc is not registered in this environment.")

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available in this PyTorch build.")

    # Match upstream: avoid recompilation-limit failures when parametrizing many shapes.
    torch._dynamo.config.cache_size_limit = 64

    np.random.seed(0)
    var, indices, update, cpu_out = _make_case(a, b, c, var_dtype, indices_dtype)
    var_npu = var.to(device)
    indices_npu = indices.to(device)
    update_npu = update.to(device)

    class Network(torch.nn.Module):
        def forward(self, var_in, indices_in, update_in):
            torch.ops.custom.scatter_nd_update_asc(var_in, indices_in, update_in)
            return var_in

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    npu_model = Network().to(device)
    # Make sure we don't reuse a compiled graph across wildly different shapes.
    torch._dynamo.reset()
    try:
        npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend, dynamic=False)
    except TypeError:
        npu_model = torch.compile(npu_model, fullgraph=True, backend=npu_backend)
    except Exception as e:
        pytest.skip(f"torch.compile failed on this environment: {e}")

    out = npu_model(var_npu, indices_npu, update_npu)
    assert out is var_npu

    npu_out = var_npu.cpu().numpy().astype(cpu_out.dtype)
    assert np.array_equal(npu_out, cpu_out)
