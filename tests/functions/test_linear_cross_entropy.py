import pytest
import torch

from mojo_opset import functions
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph


@pytest.mark.api("functions.linear_cross_entropy")
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.accuracy
def test_master(accuracy_backend, reduction):
    implementation, _, device = accuracy_backend
    x = torch.randn(2048, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4096, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(4096, (2048,), device=device)
    actual, zloss = functions.linear_cross_entropy(
        x, weight, labels, reduction=reduction, implementation=implementation
    )
    expected, _ = functions.linear_cross_entropy(
        x, weight, labels, reduction=reduction, implementation="torch_reference"
    )
    assert zloss is None
    assert_close(actual, expected, torch.float32, rtol=6e-3, atol=6e-3)
    upstream = torch.rand_like(actual)
    if reduction != "mean":
        upstream /= x.shape[0]
    a_grads = torch.autograd.grad(actual, (x, weight), upstream)
    e_grads = torch.autograd.grad(expected, (x, weight), upstream)
    for a, e in zip(a_grads, e_grads):
        # Original master BF16 standard, including its aggregate-error check.
        assert_close(a, e, a.dtype, rtol=0.05, atol=0.1)
        assert (a - e).abs().mean() < 0.1 or ((a - e) / (e + 0.01)).abs().mean() < 0.01


@pytest.mark.api("functions.linear_cross_entropy")
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("option", ["bias", "class_weight", "smoothing", "zloss", "softcap", "accum"])
@pytest.mark.accuracy
def test_options(accuracy_backend, reduction, option):
    implementation, _, device = accuracy_backend
    x = torch.randn(17, 64, device=device, requires_grad=True)
    weight = torch.randn(128, 64, device=device, requires_grad=True)
    labels = torch.randint(128, (17,), device=device)
    labels[0] = -100
    bias = torch.randn(128, device=device, requires_grad=True) if option == "bias" else None
    options = {
        "class_weight": dict(ce_weight=torch.rand(128, device=device) + 0.1),
        "smoothing": dict(label_smoothing=0.1),
        "zloss": dict(lse_square_scale=0.01, return_z_loss=True),
        "softcap": dict(softcap=10.0),
        "accum": dict(accum_dtype=torch.float32),
    }.get(option, {})
    loss, diagnostic = functions.linear_cross_entropy(
        x,
        weight,
        labels,
        bias,
        reduction=reduction,
        implementation=implementation,
        **options,
    )
    # Independently model the original optimized formula. The preserved old
    # reference ignores softcap and uses mean-only z-loss, so is not its oracle.
    logits = x @ weight.T
    if bias is not None:
        logits = logits + bias
    if option == "softcap" and implementation != "torch_reference":
        logits = 10.0 * (logits / 10.0).tanh()
    expected = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=options.get("ce_weight"),
        reduction=reduction,
        label_smoothing=options.get("label_smoothing", 0.0),
    )
    if option == "zloss":
        z = (0.01 * logits.logsumexp(-1).square()).masked_fill(labels == -100, 0)
        if implementation == "torch_reference" or reduction == "mean":
            z = z.sum() / (labels != -100).sum()
        elif reduction == "sum":
            z = z.sum()
        expected = expected + z
        assert_close(diagnostic, z, torch.float32, rtol=6e-3, atol=6e-3)
    else:
        assert diagnostic is None
    assert_close(loss, expected, torch.float32, rtol=6e-3, atol=6e-3)
    upstream = torch.randn_like(loss)
    inputs = (x, weight) if bias is None else (x, weight, bias)
    for actual, reference in zip(
        torch.autograd.grad(loss, inputs, upstream, retain_graph=True), torch.autograd.grad(expected, inputs, upstream)
    ):
        assert_close(actual, reference, torch.float32, rtol=6e-3, atol=6e-3)
    # A second backward must not mutate and rescale the saved gradient buffers.
    for a, e in zip(
        torch.autograd.grad(loss, inputs, upstream * 2, retain_graph=True),
        torch.autograd.grad(loss, inputs, upstream, retain_graph=True),
    ):
        assert_close(a, e * 2, rtol=1.3e-6, atol=1e-5)


@pytest.mark.api("functions.linear_cross_entropy")
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.bitwise
def test_linear_ce_bitwise(accuracy_backend, reduction):
    implementation, _, device = accuracy_backend
    x = torch.randn(17, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, 64, device=device, dtype=x.dtype, requires_grad=True)
    labels = torch.randint(128, (17,), device=device)

    def run(x, w):
        return functions.linear_cross_entropy(x, w, labels, reduction=reduction, implementation=implementation)

    assert_repeatable(run, (x, w))


@pytest.mark.accuracy
@pytest.mark.api("functions.linear_cross_entropy")
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_compile(accuracy_backend, reduction):
    implementation, _, device = accuracy_backend
    preload("linear_cross_entropy", implementation=implementation)
    x = torch.randn(17, 64, device=device, requires_grad=True)
    w = torch.randn(128, 64, device=device, requires_grad=True)
    labels = torch.randint(128, (17,), device=device)

    def run(x, w):
        return functions.linear_cross_entropy(x, w, labels, reduction=reduction, implementation=implementation)[0]

    try:
        compiled = compile_fullgraph(run)
        actual, expected = compiled(x, w), run(x, w)
        if implementation == "triton" and reduction != "none":
            assert all(t is None for t in expected.grad_fn.saved_tensors[:5])
        torch.testing.assert_close(actual, expected)
        for a, e in zip(torch.autograd.grad(actual.sum(), (x, w)), torch.autograd.grad(expected.sum(), (x, w))):
            torch.testing.assert_close(a, e)
    finally:
        torch._dynamo.reset()
