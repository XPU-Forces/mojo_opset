import pytest
import torch

from mojo_opset import functions
from mojo_opset import modules
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


QUANT_CASES = [
    (name, shape, scale_shape, dtype, False)
    for name, cases in (("StaticQuant", STATIC_QUANT_CASES), ("Dequant", DEQUANT_CASES))
    for shape, scale_shape in cases
    for dtype in (torch.float16, torch.bfloat16)
]
QUANT_CASES += [
    ("DynamicQuant", shape, (shape[-1],), dtype, False)
    for shape in DYNAMIC_QUANT_SHAPES
    for dtype in (torch.float16, torch.bfloat16)
]
QUANT_CASES += [("DynamicQuant", shape, None, torch.bfloat16, True) for shape in ((128,), (0, 128), (2, 128))]
QUANT_CASES = [
    pytest.param(
        case,
        id=(
            f"{case[0]}-shape{case[1]}-scale{case[2]}-{str(case[3]).removeprefix('torch.')}"
            f"-{'zeros' if case[4] else 'random'}"
        ),
    )
    for case in QUANT_CASES
]


def _make_quant_case(case, backend):
    name, shape, scale_shape, dtype, zero = case
    implementation, _, device = backend
    if name == "Dequant":
        x, scale = make_static_quant_case(shape, scale_shape, dtype, device)
        q, _ = functions.static_quant(x, scale, implementation="torch_reference")
        inputs = (q, scale)
        options = dict(output_dtype=dtype)
    elif name == "StaticQuant":
        x, scale = make_static_quant_case(shape, scale_shape, dtype, device)
        inputs, options = (x,), dict(input_size=scale_shape, device=device)
    else:
        inputs = ((torch.zeros if zero else torch.randn)(shape, device=device, dtype=dtype),)
        options = dict(input_size=None if zero else shape[-1], device=device)
    actual = getattr(modules, name)(**options, implementation=implementation)
    reference = getattr(modules, name)(**options, implementation="torch_reference")
    with torch.no_grad():
        if name == "StaticQuant":
            actual.scale.copy_(scale)
        elif name == "DynamicQuant" and not zero:
            actual.inv_smooth_scale.copy_((torch.rand(shape[-1], device=device) + 0.1).reciprocal())
        reference.load_state_dict(actual.state_dict())
    return actual, reference, inputs


@pytest.mark.api("modules.StaticQuant", "modules.DynamicQuant", "modules.Dequant")
@pytest.mark.parametrize("name", ["StaticQuant", "DynamicQuant", "Dequant"])
@pytest.mark.accuracy
def test_quant(accuracy_backend, name):
    impl, _, device = accuracy_backend
    kwargs = {} if name == "Dequant" else dict(input_size=128, device=device)
    actual = getattr(modules, name)(**kwargs, implementation=impl)
    reference = getattr(modules, name)(**kwargs, implementation="torch_reference")
    reference.load_state_dict(actual.state_dict())
    inputs = (torch.randn(3, 128, device=device, dtype=torch.bfloat16),)
    if name == "Dequant":
        inputs = (torch.randint(-127, 127, (3, 128), dtype=torch.int8, device=device), torch.rand(128, device=device))
    a, e = actual(*inputs), reference(*inputs)
    if name == "Dequant":
        assert_close(a, e, rtol=0, atol=0)
    else:
        assert_close(a[0], e[0], rtol=0, atol=1)
        assert_close(a[1], e[1], rtol=2e-3, atol=2e-3)


@pytest.mark.api("modules.StaticQuant", "modules.DynamicQuant", "modules.Dequant")
@pytest.mark.parametrize("case", QUANT_CASES)
@pytest.mark.accuracy
def test_quant_shapes(accuracy_backend, case):
    actual, reference, inputs = _make_quant_case(case, accuracy_backend)
    output, expected = actual(*inputs), reference(*inputs)
    if case[0] == "Dequant":
        assert_close(output, expected, rtol=0, atol=0)
    else:
        assert_close(output[0], expected[0], rtol=0, atol=0 if case[4] else 1)
        assert_close(output[1], expected[1], rtol=0 if case[4] else 2e-3, atol=0 if case[4] else 2e-3)
