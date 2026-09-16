import os

import pytest
import torch

from mojo_opset import Target
from mojo_opset.utils import target as target_utils
from mojo_opset.utils.target import detect_target


def pytest_addoption(parser):
    group = parser.getgroup("mojo accuracy")
    group.addoption(
        "--check",
        choices=("accuracy", "bitwise", "all"),
        default="all",
        help="Operator checks to run; framework/performance tests are unaffected.",
    )
    group.addoption(
        "--mojo-implementation",
        default=os.getenv("MOJO_ACCURACY_IMPLEMENTATION"),
        help="Optimized implementation to compare with torch_reference (for example, triton).",
    )
    group.addoption(
        "--mojo-target",
        default=None,
        help="Exact hardware target to test (for example, npu.a2).",
    )
    perf = parser.getgroup("mojo performance")
    perf.addoption("--mojo-perf", action="store_true", help="Enable the opt-in performance suite.")
    perf.addoption("--perf-output", default="artifacts/perf.json", help="Machine-readable performance report.")
    perf.addoption("--perf-cache", choices=("warm", "cold", "rotate"), default="rotate")
    perf.addoption("--perf-timers", default="profiler,e2e", help="Comma-separated profiler,event,e2e.")
    perf.addoption("--perf-batch", type=int, default=1, help="Calls per event sample; cold requires 1.")
    perf.addoption("--perf-instances", type=int, default=2, help="Independent input instances for rotate policy.")
    perf.addoption("--perf-reduction", choices=("span", "sum"), default="span")
    perf.addoption("--perf-kernel", action="append", default=[], help="Profiler kernel selector; repeatable.")
    perf.addoption("--perf-kernel-match", choices=("exact", "contains", "regex"), default="contains")
    perf.addoption("--perf-flush-mb", type=int, default=None, help="Cold eviction buffer MiB; default: 2 x L2.")
    perf.addoption("--perf-warmup", type=int, default=10)
    perf.addoption("--perf-repeats", type=int, default=20)
    perf.addoption("--perf-device", type=int, default=0)


def pytest_collection_modifyitems(config, items):
    check = config.getoption("--check")
    if check == "all":
        return
    other = "bitwise" if check == "accuracy" else "accuracy"
    deselected = [item for item in items if item.get_closest_marker(other)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        excluded = set(deselected)
        items[:] = [item for item in items if item not in excluded]


@pytest.fixture(scope="session")
def accuracy_backend(pytestconfig):
    explicit_target = pytestconfig.getoption("--mojo-target")
    detected = detect_target(explicit_target)
    target = detected.key

    hardware = detected.platform
    devices = {"npu": "npu", "ilu": "cuda", "mlu": "mlu"}
    if hardware not in devices:
        if explicit_target:
            pytest.fail(f"Unsupported accelerator target: {target}")
        pytest.skip("accuracy tests require an accelerator target")

    device = devices[hardware]
    if device == "cuda" and not torch.cuda.is_available():
        if explicit_target:
            pytest.fail("the requested accelerator is unavailable")
        pytest.skip("the requested accelerator is unavailable")
    if device != "cuda" and not hasattr(torch, device):
        if explicit_target:
            pytest.fail(f"PyTorch has no {device!r} device backend")
        pytest.skip(f"PyTorch has no {device!r} device backend")
    if device != "cuda" and not getattr(torch, device).is_available():
        pytest.fail(f"Requested device {device!r} is unavailable")

    implementation = pytestconfig.getoption("--mojo-implementation")
    return implementation, target, device


@pytest.fixture
def accuracy_seed(accuracy_backend, monkeypatch):
    # Select the session's requested target at the test boundary, not per op.
    monkeypatch.setattr(target_utils, "_AUTO_TARGET", Target.parse(accuracy_backend[1]))
    torch.manual_seed(42)


@pytest.fixture
def varlen_backend(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation == "torch_reference":
        return implementation, device
    if not target.startswith("npu.a2") or implementation not in (None, "native"):
        pytest.skip("Varlen FA currently has an A2 native provider only")
    return implementation, device


@pytest.fixture
def native_swa_backend(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation == "torch_reference":
        return implementation, device
    if target != "npu.a5.950pr" or implementation not in (None, "native"):
        pytest.skip("Native SWA inference currently has a 950PR native provider only")
    return implementation, device
