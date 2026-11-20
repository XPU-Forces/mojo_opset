"""Exercise the integration boundary without loading full model weights."""

from types import SimpleNamespace

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.integrations import vllm_vision_pos_embed_interp as integration
from mojo_opset.kernels.torch.vision_pos_embed_interp import vision_pos_embed_interp_fwd as native
from mojo_opset.utils import target as target_utils
from tests._compile import compile_fullgraph


@pytest.mark.api("functions.vision_pos_embed_interp_batch")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_capability_dispatch_and_native_parity(accuracy_backend, dtype, monkeypatch):
    _, _, device = accuracy_backend
    weight = torch.randn(48 * 48, 1152, device=device, dtype=dtype) * 0.25
    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    called = []

    def counted_native(*args):
        called.append(1)
        return native(*args)

    # Community CI uses an older stack than the vLLM integration requires.
    monkeypatch.setattr(integration, "_software_supported", lambda: False)
    assert not integration.can_use_npu_triton(weight, True)
    before = integration.interpolate_vllm_grid(weight, grids, 48, 2, dtype, has_triton=True, native=counted_native)
    assert len(called) == len(grids)

    monkeypatch.setattr(integration, "_software_supported", lambda: True)
    assert not integration.can_use_npu_triton(weight, False)
    assert integration.can_use_npu_triton(weight, True)
    called.clear()
    after = integration.interpolate_vllm_grid(weight, grids, 48, 2, dtype, has_triton=True, native=counted_native)
    assert not called
    torch.testing.assert_close(
        after, before, atol=5e-5 if dtype == torch.float32 else 1e-2, rtol=1e-5 if dtype == torch.float32 else 1e-2
    )
    separate = F.vision_pos_embed_interp_batch(weight, grids, 48, 2, dtype, implementation="triton")
    torch.testing.assert_close(after, separate, atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp_batch")
def test_kernel_errors_are_not_hidden(accuracy_backend, monkeypatch):
    _, _, device = accuracy_backend
    weight = torch.randn(14 * 14, 33, device=device)
    monkeypatch.setattr(integration, "can_use_npu_triton", lambda *args: True)

    def fail(*args, **kwargs):
        raise RuntimeError("kernel diagnostic")

    monkeypatch.setattr(integration, "vision_pos_embed_interp_batch", fail)
    with pytest.raises(RuntimeError, match="kernel diagnostic"):
        integration.interpolate_vllm_grid(weight, [(1, 8, 16)], 14, 2, weight.dtype, has_triton=True, native=native)


@pytest.mark.api("functions.vision_pos_embed_interp_batch")
def test_custom_op_fake_and_fullgraph(accuracy_backend):
    _, target, device = accuracy_backend
    weight = torch.randn(14 * 14, 65, device=device, dtype=torch.bfloat16) * 0.25
    grids = [(1, 2, 6), (3, 4, 8)]
    preload("vision_pos_embed_interp_batch", implementation="triton")
    namespace = "mojo_npu_triton_a5" if target.startswith("npu.a5") else "mojo_npu_triton_a2"
    op = getattr(getattr(torch.ops, namespace), "vision_pos_embed_interp_batch_fwd").default
    flat = [v for grid in grids for v in grid]
    torch.library.opcheck(op, (weight, flat, 14, 2, weight.dtype), test_utils=("test_schema", "test_faketensor"))

    def run(x):
        return F.vision_pos_embed_interp_batch(x, grids, 14, 2, x.dtype, implementation="triton")

    expected = run(weight)
    try:
        actual = compile_fullgraph(run)(weight)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        torch._dynamo.reset()


@pytest.mark.reference
@pytest.mark.parametrize(
    "name,expected",
    [
        ("Ascend910_9382", "npu.a2.910_9382"),
        ("Ascend910_9391", "npu.a2.910_9391"),
        ("Ascend910B3", "npu.a2.910b3"),
        ("Ascend950PR", "npu.a5.950pr"),
    ],
)
def test_detect_supported_soc(monkeypatch, name, expected):
    monkeypatch.delenv("MOJO_ASCEND_ARCH", raising=False)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda: name), raising=False)
    assert str(target_utils._detect_npu_target()) == expected
