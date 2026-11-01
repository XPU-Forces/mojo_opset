from importlib import import_module

import pytest
import torch

from mojo_opset import Target
from mojo_opset.functions import _dispatch
from mojo_opset.utils import target as target_utils


def _bindings(package, target):
    provider = import_module(package)
    return {**provider.OPS, **getattr(provider, "OVERRIDES", {}).get(target.key, {})}


def check_implementation(op_ids, implementation, target):
    """Check provider declarations only; missing dependencies and broken wrappers must fail at runtime."""
    target = Target.parse(target)
    known_ops = import_module("mojo_opset.kernels.torch").OPS
    assert op_ids and set(op_ids) <= known_ops.keys(), f"Missing or unknown test op IDs: {op_ids}"
    for op_id in sorted(op_ids):
        selected = implementation
        if selected is None:
            try:
                selected = _dispatch.resolve_implementation(op_id, target)
            except LookupError:
                available = any(
                    op_id in _bindings(package, target)
                    for (arch, _), package in _dispatch._PROVIDERS.items()
                    if arch == target.arch_key
                )
                if available:
                    raise  # An implemented op missing from YAML is a config error.
                pytest.skip(f"No provider for {op_id} on {target}")
        if selected != "torch":
            package = _dispatch._provider_package(target, selected)
            if op_id not in _bindings(package, target):
                if implementation is None:
                    raise LookupError(f"Configured {selected} has no provider for {op_id} on {target}")
                pytest.skip(f"{selected} has no provider for {op_id} on {target}")


def check_swa_case(backend, *, dtype=torch.bfloat16, dim=128, interleave=False, output_f32=False, windows=(128, 16)):
    """Select a SWA provider and exclude only its documented unsupported parameters."""
    implementation, target, _ = backend
    selected = implementation or _dispatch.resolve_implementation("swa", Target.parse(target))
    if selected == "torch_npu":
        if dtype != torch.bfloat16 or dim != 128:
            pytest.skip("torch_npu SWA requires BF16 head_dim=128")
        if interleave:
            pytest.skip("torch_npu SWA requires non-interleaved GQA")
        if output_f32:
            pytest.skip("torch_npu SWA has no FP32 output cache")
        local, prefix = windows
        if local is None or local <= 0 or prefix is None or prefix < 0:
            pytest.skip("torch_npu SWA requires explicit positive local and nonnegative global windows")
    return selected


def check_moe_scatter_case(backend, *, weighted=False, dropped_routes=False):
    """A5 native only implements unweighted, complete route permutations."""
    implementation, target, _ = backend
    target = Target.parse(target)
    selected = implementation or _dispatch.resolve_implementation("moe_scatter", target)
    if target.arch_key == "npu.a5" and selected == "native" and (weighted or dropped_routes):
        pytest.skip("A5 native moe_scatter requires unweighted, complete route permutations")
    return selected


@pytest.fixture(autouse=True)
def operator_test(request, monkeypatch):
    torch.manual_seed(42)
    reference = request.node.get_closest_marker("reference")
    if reference and "accuracy_backend" not in request.fixturenames:
        return  # Pure reference tests must also run without an accelerator.
    implementation, target, device = request.getfixturevalue("accuracy_backend")
    if not reference:
        markers = list(request.node.iter_markers("api"))
        assert markers, f"{request.node.nodeid}: missing api mark"
        for marker in markers:
            op_ids = set(marker.kwargs.get("ops", (api.removeprefix("functions.") for api in marker.args)))
            check_implementation(op_ids, implementation, target)
    monkeypatch.setattr(target_utils, "_AUTO_TARGET", Target.parse(target))
    # Apply the same FP32 math policy to candidate and reference, then restore it.
    # These flags do not change input dtypes or production/performance defaults.
    if device == "npu":
        # torch_npu 2.10.0.post7 (the 950 build) exposes these as descriptors over a
        # process-wide C++ flag rather than as instance attributes, so they cannot be
        # deleted: monkeypatch's undo raises AttributeError and turns every test in the
        # session into a teardown ERROR. Set and restore by hand instead, and only where
        # the flag exists.
        for module in (torch.npu.conv, torch.npu.matmul):
            if hasattr(module, "allow_hf32"):
                previous = module.allow_hf32
                module.allow_hf32 = False
                request.addfinalizer(lambda m=module, p=previous: setattr(m, "allow_hf32", p))
    elif device == "cuda":
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
