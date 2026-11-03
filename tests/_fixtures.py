from importlib import import_module

import pytest
import torch

from mojo_opset import Target
from mojo_opset.functions import _dispatch
from mojo_opset.utils import target as target_utils
from mojo_opset.utils._exports import additions


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
            # Composite APIs have no single backend. Their designated CI job
            # supplies all dependencies and exercises the configured mixture.
            selected = None if marker.kwargs.get("default_backend_job") == implementation else implementation
            check_implementation(op_ids, selected, target)
            if device == "npu" and "flash_attention" in op_ids:
                fa_impl = selected or _dispatch.resolve_implementation("flash_attention", Target.parse(target))
                if fa_impl == "native" and torch.npu.get_device_name().startswith("Ascend910B3"):
                    pytest.skip("native flash_attention backward assumes 48 Vector cores; 910B3 adaptation is pending")
        private = additions("tests")
        if private is not None:
            private.check_case(request, implementation, target)
    monkeypatch.setattr(target_utils, "_AUTO_TARGET", Target.parse(target))
    # Apply the same FP32 math policy to candidate and reference, then restore it.
    # These flags do not change input dtypes or production/performance defaults.
    if device == "npu":
        monkeypatch.setattr(torch.npu.conv, "allow_hf32", False)
        monkeypatch.setattr(torch.npu.matmul, "allow_hf32", False)
    elif device == "cuda":
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
