import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_accuracy
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph


@pytest.mark.api("functions.linear_cross_entropy")
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.accuracy
def test_ce_reduction(accuracy_backend, reduction):
    device = accuracy_backend[2]
    labels = torch.randint(128, (17,), device=device)
    labels[0] = -100

    def run(x, weight, **selection):
        return F.linear_cross_entropy(x, weight, labels, reduction=reduction, **selection)

    assert_accuracy(
        run,
        tuple(torch.randn(shape, device=device, dtype=torch.bfloat16) for shape in ((17, 64), (128, 64))),
        implementation=accuracy_backend[0],
        grad_tolerances={0: (2e-2, 2e-2), 1: (2e-2, 2e-2)},
    )


@pytest.mark.api("functions.linear_cross_entropy_and_zloss")
@pytest.mark.parametrize("calc_acc", [False, True])
@pytest.mark.accuracy
def test_ce_zloss(accuracy_backend, calc_acc):
    device = accuracy_backend[2]
    labels = torch.randint(128, (17,), device=device)
    labels[0] = -100

    def run(x, weight, **selection):
        return F.linear_cross_entropy_and_zloss(x, weight, labels, 0.01, calc_acc=calc_acc, **selection)

    assert_accuracy(
        run,
        tuple(torch.randn(shape, device=device, dtype=torch.bfloat16) for shape in ((17, 64), (128, 64))),
        implementation=accuracy_backend[0],
    )


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(op, marks=pytest.mark.api("functions." + op))
        for op in ["linear_cross_entropy", "linear_cross_entropy_and_zloss"]
    ],
)
@pytest.mark.bitwise
def test_loss_bitwise(accuracy_backend, op):
    implementation, target, device = accuracy_backend
    inputs = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True) for shape in ((17, 64), (128, 64))
    )
    labels = torch.randint(128, (17,), device=device)

    def run(x, weight):
        kwargs = {"z_loss_weight": 0.01} if op.endswith("_and_zloss") else {}
        return getattr(F, op)(x, weight, labels, implementation=implementation, **kwargs)

    assert_repeatable(run, inputs)


# Preserve the Ext master chunk/vocabulary boundaries as well as the option matrix.
CE_CASES = (
    [(17, 16, 64, acc, zloss, ignored, True) for acc in (False, True) for zloss in (0.0, 1e-4) for ignored in (-100, 0)]
    + [(tokens, 32, 67, False, 1e-4, -100, False) for tokens in (0, 2047, 2048, 2049, 4097)]
    + [(9, 32, 4103, True, 1e-4, -100, True)]
)


def _ce_inputs(case, device):
    tokens, hidden, vocab, *_ = case
    torch.manual_seed(44 if vocab == 4103 else 42 if hidden == 16 else 43)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    weight = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device)
    labels = torch.randint(vocab, (tokens,), dtype=torch.long, device=device)
    if vocab != 4103:
        labels[:: 5 if hidden == 16 else 7] = case[5]
    return x, weight, labels


def _ce_call(x, weight, labels, case, implementation):
    return F.linear_cross_entropy_and_zloss(
        x,
        weight,
        labels,
        case[4],
        calc_acc=case[3],
        ignore_index=case[5],
        align_precision=case[6],
        reduction="none",
        implementation=implementation,
    )


@pytest.mark.api("functions.linear_cross_entropy_and_zloss")
@pytest.mark.parametrize("case", CE_CASES)
@pytest.mark.accuracy
def test_ce(accuracy_backend, case):
    impl, _, device = accuracy_backend
    x, weight, labels = _ce_inputs(case, device)
    actual_inputs = [t.detach().clone().requires_grad_(True) for t in (x, weight)]
    reference_inputs = [t.detach().clone().requires_grad_(True) for t in (x, weight)]
    actual = _ce_call(*actual_inputs, labels, case, impl)
    expected = _ce_call(*reference_inputs, labels, case, "torch_reference")
    grad = torch.randn_like(actual[0])
    actual[0].backward(grad)
    expected[0].backward(grad)
    for index, (a, e) in enumerate(zip(actual, expected)):
        tol = 0.0 if case[3] and index == 1 else 2e-2
        assert_close(a.float(), e.float(), rtol=tol, atol=tol)
    assert_close(actual_inputs[0].grad.float(), reference_inputs[0].grad.float(), rtol=2e-2, atol=2e-2)
    assert_close(
        actual_inputs[1].grad.float(),
        reference_inputs[1].grad.float(),
        rtol=2e-2,
        atol=5e-2 if case[0] >= 2047 else 2e-2,
    )


@pytest.mark.api("functions.linear_cross_entropy_and_zloss")
@pytest.mark.parametrize("case", CE_CASES)
@pytest.mark.bitwise
def test_ce_bitwise(accuracy_backend, case):
    impl, _, device = accuracy_backend
    x, weight, labels = _ce_inputs(case, device)
    assert_repeatable(lambda a, b: _ce_call(a, b, labels, case, impl), (x.requires_grad_(), weight.requires_grad_()))


@pytest.mark.accuracy
@pytest.mark.parametrize("reduction", ["mean", "none"])
@pytest.mark.parametrize("op", [
    pytest.param("linear_cross_entropy", marks=pytest.mark.api("functions.linear_cross_entropy")),
    pytest.param("linear_cross_entropy_and_zloss", marks=pytest.mark.api("functions.linear_cross_entropy_and_zloss")),
])
def test_compile(accuracy_backend, reduction, op):
    implementation, _, device = accuracy_backend
    preload(op, implementation=implementation)
    x = torch.randn(17, 64, device=device, requires_grad=True)
    w = torch.randn(128, 64, device=device, requires_grad=True)
    labels = torch.randint(128, (17,), device=device)
    options = {"z_loss_weight": 0.01} if op.endswith("_and_zloss") else {}

    def run(x, w):
        result = getattr(F, op)(x, w, labels, reduction=reduction, implementation=implementation, **options)
        return result[0] if options else result

    try:
        compiled = compile_fullgraph(run)
        actual, expected = compiled(x, w), run(x, w)
        torch.testing.assert_close(actual, expected)
        for a, e in zip(torch.autograd.grad(actual.sum(), (x, w)), torch.autograd.grad(expected.sum(), (x, w))):
            torch.testing.assert_close(a, e)
    finally:
        torch._dynamo.reset()
