import os

from pathlib import Path

import pytest
import torch

from mojo_opset.utils.target import detect_target


@pytest.fixture(autouse=True)
def deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def pytest_configure(config):
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        rank = os.environ["LOCAL_RANK"]
        for option in ("xmlpath", "perf_output"):
            if value := getattr(config.option, option, None):
                path = Path(value)
                setattr(config.option, option, str(path.with_name(f"{path.stem}-rank{rank}{path.suffix}")))


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
        default=None,
        help="Implementation to test (for example, triton); defaults to target configuration.",
    )
    group.addoption(
        "--mojo-target",
        default=None,
        help="Exact hardware target to test (for example, npu.a2).",
    )
    perf = parser.getgroup("mojo performance")
    perf.addoption(
        "--perf-output", default="tests/performance/results/current.json", help="Machine-readable performance report."
    )
    perf.addoption("--perf-timers", default="profiler,e2e", help="Comma-separated profiler,event,e2e.")
    perf.addoption("--perf-instances", type=int, default=2, help="Independent input instances for rotate policy.")
    perf.addoption("--perf-warmup", type=int, default=10)
    perf.addoption("--perf-repeats", type=int, default=20)
    perf.addoption("--perf-device", type=int, default=0)


def pytest_ignore_collect(collection_path, config):
    performance = Path(__file__).parent / "performance"
    if collection_path != performance:
        return None
    # Explicit paths (including files / node IDs) opt in; ordinary pytest excludes benchmarks.
    return not any(
        (Path(config.invocation_params.dir) / str(arg).split("::", 1)[0]).resolve().is_relative_to(performance)
        for arg in config.args
    )


def pytest_collection_modifyitems(config, items):
    check = config.getoption("--check")
    if check == "all":
        return
    operator_roots = (Path(__file__).parent / "functions", Path(__file__).parent / "modules")
    deselected = [
        item
        for item in items
        if any(item.path.is_relative_to(root) for root in operator_roots) and not item.get_closest_marker(check)
    ]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        excluded = set(deselected)
        items[:] = [item for item in items if item not in excluded]


@pytest.fixture(scope="session")
def accuracy_backend(pytestconfig):
    explicit_target = pytestconfig.getoption("--mojo-target")
    if int(os.getenv("WORLD_SIZE", "1")) > 1 and explicit_target and explicit_target.startswith("npu."):
        import torch_npu  # noqa: F401

        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
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


@pytest.fixture(scope="session")
def ep_group(accuracy_backend):
    implementation, target, _ = accuracy_backend
    if not target.startswith("npu.a5") or implementation not in (None, "native"):
        pytest.skip("EP tests require A5 native")
    from mojo_opset import Target
    from mojo_opset.functions._dispatch import resolve_implementation

    if (implementation or resolve_implementation("moe_dispatch", Target.parse(target))) != "native":
        pytest.skip("EP tests require the selected native provider")
    if int(os.getenv("WORLD_SIZE", "1")) not in (2, 4):
        pytest.skip("launch EP tests with torchrun --nproc_per_node=2 or 4")
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    yield dist.group.WORLD
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def ep_context(ep_group, request):
    from mojo_opset.comm import EPContext

    hidden, experts, topk, capacity = getattr(request, "param", (128, 16, 2, 33))
    with EPContext.create(ep_group, layer=int(os.getenv("MOJO_EP_TEST_LAYER", "0")), hidden=hidden,
                          num_experts=experts, num_topk=topk, capacity=capacity) as context:
        yield context


@pytest.fixture
def ep_inputs(ep_context):
    hidden, experts, topk, _ = ep_context.moe_config
    rank = ep_context.group.rank()

    def make(tokens, case="uniform"):
        rng = torch.Generator(device="cpu").manual_seed(2468 + rank)
        x = (torch.randn(tokens, hidden, generator=rng) / 8).bfloat16()
        x[:, 0] = rank
        x[:, 1] = torch.arange(tokens) % 128
        x[:, 2] = torch.arange(tokens) // 128
        routes = torch.rand(tokens, experts, generator=rng).topk(topk, dim=1).indices
        if case == "hotspot":
            routes[:] = torch.arange(topk)
        elif case in ("local", "remote"):
            owner = (rank + (case == "remote")) % ep_context.num_ranks
            routes[:] = owner * (experts // ep_context.num_ranks) + torch.arange(topk)
        elif case == "masked":
            routes[:, 1::2] = -1
        elif case == "all_masked":
            routes.fill_(-1)
        return x, routes, torch.rand(tokens, topk, generator=rng)

    return make


@pytest.fixture
def ep_check(ep_group):
    import torch.distributed as dist

    def check(assertion):
        # Propagate assertion failures before any rank enters the next collective.
        error = None
        try:
            assertion()
        except AssertionError as exc:
            error = str(exc)
        errors = [None] * ep_group.size()
        dist.all_gather_object(errors, error, group=ep_group)
        assert not any(errors), errors

    return check
