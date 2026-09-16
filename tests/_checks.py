"""Reference accuracy and independent forward/backward repeatability checks."""

import torch

from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map

# Dtype means the computation/input precision, not simply the output dtype:
# BF16 operators can expose FP32 outputs or accumulate FP32 parameter gradients.
TOLERANCES = {
    torch.float64: (1e-7, 1e-9),
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


def comparison_dtype(tensors):
    dtypes = {t.dtype for t in tensors if t.is_floating_point()}
    for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        if dtype in dtypes:
            return dtype
    raise ValueError("accuracy comparison requires floating-point inputs")


def clone_with_grad(tensors):
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


def assert_close(actual, expected, dtype=None, *, name="tensor", max_print=5, rtol=None, atol=None):
    if isinstance(actual, (tuple, list, dict)) or isinstance(expected, (tuple, list, dict)):
        actual_leaves, actual_spec = tree_flatten(actual)
        expected_leaves, expected_spec = tree_flatten(expected)
        assert actual_spec == expected_spec, f"{name}: result structures differ"
        for index, (a, e) in enumerate(zip(actual_leaves, expected_leaves)):
            assert_close(a, e, dtype, name=f"{name}[{index}]", max_print=max_print, rtol=rtol, atol=atol)
        return
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        return
    dtype = actual.dtype if dtype is None else dtype
    default_rtol, default_atol = TOLERANCES.get(dtype, (0, 0))
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    except AssertionError as error:
        # PyTorch remains the sole pass/fail decision. Diagnose only numerical
        # failures; metadata errors should retain their original explanation.
        details = [f"{name} (rtol={rtol:g}, atol={atol:g}): {error}"]
        if (
            max_print > 0
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and actual.device == expected.device
        ):
            a, e = actual.detach().reshape(-1), expected.detach().reshape(-1)
            mismatches = ~torch.isclose(a, e, rtol=rtol, atol=atol, equal_nan=False)
            indices = mismatches.nonzero().flatten()[:max_print].cpu().tolist()
            for flat_index in indices:
                index = []
                remainder = flat_index
                for size in reversed(actual.shape):
                    remainder, coordinate = divmod(remainder, size)
                    index.append(coordinate)
                index = tuple(reversed(index))
                av, ev = a[flat_index].item(), e[flat_index].item()
                details.append(
                    f"{name}{index}: actual={av}, reference={ev}, "
                    f"abs_error={abs(av - ev):.8g}, allowed={atol + rtol * abs(ev):.8g}"
                )
        raise AssertionError("\n".join(details)) from None


def assert_mojo_close(actual, expected, *, name="tensor", max_print=5):
    """Original Mojo training criterion: per-element max tolerance plus its mean-error check."""
    assert actual.shape == expected.shape, f"{name}: shapes differ: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{name}: dtypes differ: {actual.dtype} != {expected.dtype}"
    assert actual.device == expected.device, f"{name}: devices differ: {actual.device} != {expected.device}"
    max_rtol, max_atol, mean_rtol, mean_atol = {
        torch.float32: (6e-3, 6e-3, 1e-4, 1e-4),
        torch.float16: (2e-2, 2e-2, 2e-2, 2e-2),
        torch.bfloat16: (5e-2, 1e-1, 1e-2, 1e-2),
    }[actual.dtype]
    assert_close(
        actual.float(),
        expected.float(),
        torch.float32,
        rtol=max_rtol,
        atol=max_atol,
        name=name,
        max_print=max_print,
    )
    # Master evaluates these means in the original dtype and uses max_atol in
    # the absolute branch; retain that exact criterion, not a new global default.
    difference = expected - actual
    mean_abs = difference.abs().mean()
    mean_rel = (difference / (expected + mean_atol)).abs().mean()
    assert bool(mean_abs < max_atol or mean_rel < mean_rtol), (
        f"{name}: mean_abs_error={mean_abs.item():.8g} (limit={max_atol:g}), "
        f"mean_relative_error={mean_rel.item():.8g} (limit={mean_rtol:g})"
    )


def assert_accuracy(function, inputs, *, implementation, gradient_scale=1.0, grad_tolerances=None, **kwargs):
    """Compare public training forward/backward with reference using one upstream.

    Tests with a specialized oracle can call ``assert_close`` directly; both
    paths use the same dtype tolerances and failure diagnostics.
    """
    dtype = comparison_dtype(inputs)
    grad_tolerances = {} if grad_tolerances is None else grad_tolerances
    if not set(grad_tolerances) <= set(range(len(inputs))):
        raise ValueError("grad_tolerances keys must be input indices")
    actual_inputs = clone_with_grad(inputs)
    reference_inputs = clone_with_grad(inputs)
    actual = function(*actual_inputs, implementation=implementation, **kwargs)
    expected = function(*reference_inputs, implementation="torch_reference", **kwargs)
    actual = (actual,) if isinstance(actual, torch.Tensor) else actual
    expected = (expected,) if isinstance(expected, torch.Tensor) else expected
    assert len(actual) == len(expected)
    outputs, references, upstream = [], [], []
    for index, (a, e) in enumerate(zip(actual, expected)):
        if e is None:
            assert a is None
            continue
        assert_close(a, e, dtype, name=f"output[{index}]")
        if e.requires_grad:
            outputs.append(a)
            references.append(e)
            upstream.append(torch.randn_like(e) * gradient_scale)
    actual_grads = torch.autograd.grad(outputs, actual_inputs, upstream)
    expected_grads = torch.autograd.grad(references, reference_inputs, upstream)
    for index, (a, e) in enumerate(zip(actual_grads, expected_grads)):
        rtol, atol = grad_tolerances.get(index, (None, None))
        assert_close(a, e, dtype, name=f"grad[{index}]", rtol=rtol, atol=atol)


def _snapshot(value):
    # Freeze each result before a later call can reuse/mutate its storage.
    return tree_map(lambda x: x.detach().to("cpu", copy=True) if isinstance(x, torch.Tensor) else x, value)


def assert_bitwise_equal(actual, expected, *, phase="output"):
    actual_leaves, actual_spec = tree_flatten(actual)
    expected_leaves, expected_spec = tree_flatten(expected)
    assert actual_spec == expected_spec, f"{phase}: result structures differ"
    for index, (a, e) in enumerate(zip(actual_leaves, expected_leaves)):
        label = f"{phase}[{index}]"
        if isinstance(a, torch.Tensor) and isinstance(e, torch.Tensor):
            assert a.shape == e.shape and a.dtype == e.dtype, f"{label}: shape/dtype differ"
            # Byte views distinguish signed zero and compare NaN payloads exactly;
            # torch.equal(float tensors) and assert_close(rtol=atol=0) do not.
            a_bytes = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            e_bytes = e.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            different = int((a_bytes != e_bytes).sum().item())
            assert different == 0, f"{label}: {different}/{a_bytes.numel()} bytes differ between runs"
        else:
            assert a is None and e is None, f"{label}: expected matching tensors or None"


def assert_repeatable(function, inputs, *, parameters=(), backward=True):
    """Run twice with the same inputs/parameters/upstream; never change RNG or determinism policy."""
    targets = tuple(t for t in (*inputs, *parameters) if t.requires_grad)
    upstream = None
    first_outputs = first_grads = None
    for iteration in range(2):
        output = function(*inputs)
        frozen_output = _snapshot(output)
        if iteration == 1:
            assert_bitwise_equal(frozen_output, first_outputs, phase="forward")
        grads = ()
        if backward:
            leaves, _ = tree_flatten(output)
            outputs = tuple(t for t in leaves if isinstance(t, torch.Tensor) and t.requires_grad)
            if not outputs or not targets:
                raise ValueError("Backward bitwise checks require differentiable outputs and gradient targets")
            if upstream is None:
                upstream = tuple(torch.randn_like(t) for t in outputs)
            # Avoid accumulating .grad across passes or reusing the first graph.
            grads = torch.autograd.grad(outputs, targets, grad_outputs=upstream)
        frozen_grads = _snapshot(grads)
        if iteration == 0:
            first_outputs, first_grads = frozen_output, frozen_grads
        else:
            assert_bitwise_equal(frozen_grads, first_grads, phase="backward")
