import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._compile import compile_fullgraph

BF16 = torch.bfloat16
FP16 = torch.float16
PERF_CASES = [
    (1, 1024, 4096, BF16),
    (1, 128, 4096, BF16),
    (1, 1, 4096, BF16),
    (1, 7, 4096, BF16),
    (1, 17, 7168, BF16),
    (2, 64, 4096, BF16),
    (1, 128, 7168, BF16),
    (1, 512, 4096, BF16),
    (1, 1024, 7168, BF16),
    (1, 1, 4096, FP16),
    (1, 7, 4096, FP16),
    (1, 128, 7168, FP16),
    (1, 8192, 4096, BF16),
]
CASE_ARGS = ("batch", "seq", "hidden", "dtype")
pytestmark = pytest.mark.api("functions.mhc_coefficients")


def make_inputs(batch, seq, hidden, dtype, seed=3901, *, normal=False):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((batch, seq, 4, hidden), generator=generator).to(dtype)
    phi = torch.empty((24, 4 * hidden), dtype=torch.float32)
    if normal:
        phi.normal_(0, (4 * hidden) ** -0.5, generator=generator)
    else:
        phi.uniform_(-0.01, 0.01, generator=generator)
    alpha = torch.tensor([0.16, 0.24, 0.12], dtype=torch.float32)
    bias = torch.randn((24,), generator=generator, dtype=torch.float32) * 0.25
    return x, phi, alpha, bias


def assert_outputs(actual, expected):
    for result, reference in zip(actual, expected, strict=True):
        assert result.is_contiguous()
        result = result.cpu()
        tolerance = {FP16: 2**-10, BF16: 2**-7, torch.float32: 2**-13}[reference.dtype]
        torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)


@pytest.fixture
def provider(accuracy_backend):
    implementation, target, device = accuracy_backend
    if not target.startswith(("npu.a2", "npu.a5")) or implementation not in (None, "triton"):
        pytest.skip("mHC accelerators are Ascend A2 and A5 Triton")
    return implementation, device


@pytest.mark.accuracy
@pytest.mark.parametrize(CASE_ARGS, PERF_CASES + [(1, 255, 4096, BF16), (1, 256, 4096, BF16), (1, 257, 4096, BF16)])
def test_mhc_accuracy(provider, batch, seq, hidden, dtype):
    implementation, device = provider
    inputs = make_inputs(batch, seq, hidden, dtype, normal=seq in (17, 255, 256, 257, 1024))
    reference = F.mhc_coefficients(*inputs, implementation="torch")
    actual = F.mhc_coefficients(*(tensor.to(device) for tensor in inputs), implementation=implementation)
    assert_outputs(actual, reference)


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [BF16, FP16])
def test_reference_zero_projection(dtype):
    x, phi, alpha, bias = make_inputs(2, 3, 4096, dtype)
    hin, post, residual = F.mhc_coefficients(x, phi.zero_(), alpha, bias.zero_(), implementation="torch")
    expected_hin = (x.float() * (0.5 + 1e-6)).sum(dim=2).to(dtype)
    torch.testing.assert_close(hin, expected_hin, atol=0, rtol=0)
    torch.testing.assert_close(post, torch.ones_like(post), atol=0, rtol=0)
    torch.testing.assert_close(residual, torch.full_like(residual, 0.25), atol=1e-6, rtol=0)


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [BF16, FP16])
def test_empty_reference(dtype):
    result = F.mhc_coefficients(*make_inputs(2, 0, 4096, dtype), implementation="torch")
    assert [tuple(tensor.shape) for tensor in result] == [(2, 0, 4096), (2, 0, 4), (2, 0, 16)]
    assert [tensor.dtype for tensor in result] == [dtype, torch.float32, torch.float32]


@pytest.mark.reference
def test_invalid_inputs():
    x, phi, alpha, bias = make_inputs(1, 2, 4096, BF16)
    invalid_cases = (
        ((x[:, :, :3], phi, alpha, bias), "shape"),
        ((x.float(), phi, alpha, bias), "BF16 or FP16"),
        ((x, phi[:, :-1], alpha, bias), "shapes"),
        ((x, phi, alpha.half(), bias), "FP32"),
        ((x, phi, alpha, bias.to("meta")), "same device"),
        ((x, phi, alpha, torch.empty((24, 2))[:, 0]), "contiguous"),
    )
    for args, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            F.mhc_coefficients(*args, implementation="torch")
    with pytest.raises(RuntimeError, match="autograd"):
        F.mhc_coefficients(x, phi.requires_grad_(), alpha, bias, implementation="torch")


@pytest.mark.accuracy
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_mhc_fullgraph_output_lifetime(provider, backend):
    implementation, device = provider
    preload("mhc_coefficients", implementation=implementation)
    cpu_inputs = [make_inputs(1, 7, 4096, BF16, seed) for seed in (10, 11)]
    device_inputs = [tuple(tensor.to(device) for tensor in inputs) for inputs in cpu_inputs]

    def fn(x, phi, alpha, bias):
        return F.mhc_coefficients(x, phi, alpha, bias, implementation=implementation)

    compiled = compile_fullgraph(fn, backend=backend)
    results = [compiled(*inputs) for inputs in device_inputs]
    for actual, inputs in zip(results, cpu_inputs, strict=True):
        assert_outputs(actual, F.mhc_coefficients(*inputs, implementation="torch"))
