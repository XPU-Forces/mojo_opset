from __future__ import annotations

import pytest
import torch

from mojo_opset import functions as F

# (shape, axis) coverage: tail, stride-2 non-tail, wide-axis lanes and the
# generic transpose fallback.  Large cases enter the adaptive wide branches.
LAYOUT_CASES = [
    pytest.param((2, 64), -1, id="tail"),
    pytest.param((2, 512, 2), 1, id="stride2-fused"),
    pytest.param((2, 576, 2), 1, id="stride2"),
    pytest.param((256, 8), 0, id="width8"),
    pytest.param((256, 16), 0, id="width16"),
    pytest.param((64, 32), 0, id="width32"),
    pytest.param((64, 64), 0, id="width64"),
    pytest.param((64, 128), 0, id="width128"),
    pytest.param((2, 48, 3), 1, id="generic-axis"),
]

LARGE_LAYOUT_CASES = [
    pytest.param((128, 128, 32), id="width32-tile128"),
    pytest.param((64, 256, 32), id="width32-tile256"),
    pytest.param((32, 512, 32), id="width32-tile512"),
    pytest.param((128, 128, 64), id="width64-tile128"),
    pytest.param((64, 256, 64), id="width64-tile256"),
    pytest.param((128, 128, 128), id="width128-tile128"),
]


def _random_input(shape, axis, src):
    generator = torch.Generator().manual_seed(20260924)
    cpu = (torch.randn(shape, generator=generator) * 7).to(src)
    logical = cpu.movedim(axis, -1).contiguous().reshape(-1, shape[axis])
    logical[0, :32] = 0
    logical[0, 1:32:2] = -0.0
    if logical.numel() >= 40:
        logical.reshape(-1)[32:40] = torch.tensor(
            [float("inf"), -float("inf"), float("nan"), 0.5, -0.5, 1.125, -1.125, 0.0]
        ).to(src)
    return logical.reshape(cpu.movedim(axis, -1).shape).movedim(-1, axis).contiguous()


def _assert_bitwise(actual, expected):
    assert type(actual) is tuple and len(actual) == 2
    y, scale = actual
    ref_y, ref_scale = expected
    assert y.shape == ref_y.shape and y.dtype == ref_y.dtype
    assert scale.shape == ref_scale.shape and scale.dtype == ref_scale.dtype
    assert torch.equal(y.view(torch.uint8).cpu(), ref_y.view(torch.uint8).cpu())
    assert torch.equal(scale.view(torch.uint8).cpu(), ref_scale.view(torch.uint8).cpu())


def _run_case(backend, shape, axis, src, dst_code=24, algorithm=1, bound=0.0, *, cpu=None):
    implementation, device = backend
    if cpu is None:
        cpu = _random_input(shape, axis, src)
    x = cpu.to(device)
    kwargs = dict(
        axis=axis, dst_type=dst_code, scale_alg=algorithm, max_low_bound=bound
    )
    actual = F.dynamic_mx_quant_sr(x, implementation=implementation, **kwargs)
    expected = F.dynamic_mx_quant_sr(cpu, implementation="torch_reference", **kwargs)
    _assert_bitwise(actual, expected)
    return actual


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("shape,axis", LAYOUT_CASES)
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_layouts(dynamic_mx_quant_sr_backend, shape, axis, src, dst_code, algorithm):
    _run_case(dynamic_mx_quant_sr_backend, shape, axis, src, dst_code, algorithm)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("shape", LARGE_LAYOUT_CASES)
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_adaptive_tiles(dynamic_mx_quant_sr_backend, shape, src, dst_code, algorithm):
    _run_case(dynamic_mx_quant_sr_backend, shape, -2, src, dst_code, algorithm)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_max_low_bound(dynamic_mx_quant_sr_backend, src, dst_code):
    _run_case(dynamic_mx_quant_sr_backend, (4, 256), -1, src, dst_code, 1, bound=32.0)
    _run_case(dynamic_mx_quant_sr_backend, (4, 80), -1, src, dst_code, 1, bound=32.0)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"axis": True}, TypeError, "axis"),
        ({"axis": 2}, ValueError, "axis"),
        ({"round_mode": "rint"}, NotImplementedError, "round_mode"),
        ({"dst_type": 35}, NotImplementedError, "dst_type"),
        ({"block_size": 0}, ValueError, "block_size"),
        ({"block_size": 64}, NotImplementedError, "block_size"),
        ({"scale_alg": 2}, NotImplementedError, "scale_alg"),
        ({"dst_type_max": 448.0}, NotImplementedError, "dst_type_max"),
        ({"max_low_bound": -1.0}, ValueError, "max_low_bound"),
        ({"max_low_bound": float("nan")}, ValueError, "max_low_bound"),
        ({"max_low_bound": 1e100}, ValueError, "max_low_bound"),
        ({"scale_alg": 0, "max_low_bound": 1.0}, ValueError, "max_low_bound"),
    ],
)
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rejects_attributes(kwargs, error, match):
    with pytest.raises(error, match=match):
        F.dynamic_mx_quant_sr(torch.ones((2, 64)), **kwargs)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("value,axis,error,match", [
    (None, -1, TypeError, "input"),
    (torch.empty(()), -1, ValueError, "nonempty"),
    (torch.empty((0, 64)), -1, ValueError, "nonempty"),
    (torch.empty((2, 17)), -1, ValueError, "multiple of 16"),
    (torch.empty((2, 16)), -1, ValueError, "at least 32"),
    (torch.empty((2, 17, 3)), 1, ValueError, "multiple of 16"),
    (torch.empty((2, 64)).t(), 0, ValueError, "contiguous"),
])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rejects_invalid_tensors(value, axis, error, match):
    with pytest.raises(error, match=match):
        F.dynamic_mx_quant_sr(value, axis=axis)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rejects_dtype(dtype):
    with pytest.raises(TypeError, match="float32.*bfloat16"):
        F.dynamic_mx_quant_sr(torch.ones((2, 64), dtype=dtype))
