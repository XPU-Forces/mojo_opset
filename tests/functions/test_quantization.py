import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

STATIC_QUANT_CASES = [
    (shape, (shape[-1],)) for shape in [(1, 128), (2, 256), (8, 512), (32, 1024), (64, 4096), (57, 7338), (128, 8192)]
] + [((2, 4, 128), (4, 128)), ((3, 8, 64), (8, 64)), ((5, 2, 16, 32), (16, 32))]


DEQUANT_CASES = [
    (shape, (shape[-1],)) for shape in [(1, 128), (4, 128), (16, 512), (32, 1024), (96, 4096), (128, 8192)]
] + STATIC_QUANT_CASES[-3:]


DYNAMIC_QUANT_SHAPES = [
    (1, 128),
    (8, 128),
    (17, 320),
    (24, 512),
    (48, 1536),
    (64, 2048),
    (3, 129),
    (7, 257),
    (15, 511),
    (31, 1023),
    (63, 2047),
    (96, 4097),
    (128, 6144),
    (257, 8192),
]


def make_static_quant_case(shape, scale_shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    axes = tuple(range(x.ndim - len(scale_shape)))
    scale = (x.float().abs().amax(dim=axes) / 127).clamp(min=1e-10)
    return x, scale


@pytest.mark.api("functions.static_quant")
@pytest.mark.parametrize("shape,scale_shape", STATIC_QUANT_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_static(accuracy_backend, shape, scale_shape, dtype):
    impl, _, device = accuracy_backend
    x, scale = make_static_quant_case(shape, scale_shape, dtype, device)
    actual, returned_scale = F.static_quant(x, scale, implementation=impl)
    expected, _ = F.static_quant(x, scale, implementation="torch_reference")
    assert returned_scale is scale
    assert_close(actual, expected, rtol=0, atol=1)


@pytest.mark.api("functions.dequant")
@pytest.mark.parametrize("shape,scale_shape", DEQUANT_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_dequant(accuracy_backend, shape, scale_shape, dtype):
    impl, _, device = accuracy_backend
    x, scale = make_static_quant_case(shape, scale_shape, dtype, device)
    quantized, _ = F.static_quant(x, scale, implementation="torch_reference")
    actual = F.dequant(quantized, scale, output_dtype=dtype, implementation=impl)
    expected = F.dequant(quantized, scale, output_dtype=dtype, implementation="torch_reference")
    assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.api("functions.dynamic_quant")
@pytest.mark.parametrize(
    "shape,dtype,zero",
    [
        pytest.param(shape, dtype, False, id=f"{dtype}-{shape}")
        for dtype in (torch.float16, torch.bfloat16)
        for shape in DYNAMIC_QUANT_SHAPES
    ]
    + [pytest.param(shape, torch.bfloat16, True, id=f"zero-{shape}") for shape in ((128,), (0, 128), (2, 128))],
)
@pytest.mark.accuracy
def test_dynamic(accuracy_backend, shape, dtype, zero):
    impl, _, device = accuracy_backend
    x = torch.zeros(shape, dtype=dtype, device=device) if zero else torch.randn(shape, dtype=dtype, device=device)
    smooth = None if zero else (torch.rand(shape[-1], device=device) + 0.1).reciprocal()
    actual = F.dynamic_quant(x, smooth, implementation=impl)
    expected = F.dynamic_quant(x, smooth, implementation="torch_reference")
    assert_close(actual[0], expected[0], rtol=0, atol=0 if zero else 1)
    assert_close(actual[1], expected[1], rtol=0 if zero else 2e-3, atol=0 if zero else 2e-3)
