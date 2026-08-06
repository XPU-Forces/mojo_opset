from types import SimpleNamespace

import pytest
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry

import mojo_opset.benchmark.xpu_adapter as xpu_adapter

from mojo_opset import MojoGelu
from mojo_opset import MojoSiluFunction
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import profile
from mojo_opset.benchmark import tensor
from mojo_opset.benchmark.api import get_perf_spec
from mojo_opset.benchmark.api import iter_perf_specs
from mojo_opset.benchmark.runner_common import build_provider_map
from mojo_opset.benchmark.runner_common import detect_xpu_backend
from mojo_opset.benchmark.xpu_adapter import GeneratedMojoPerfAdapter
from mojo_opset.core.function import MojoFunction
from mojo_opset.core.operator import MojoOperator


class _DummyTarget:
    @classmethod
    def get_backend_impl(cls, backend):
        return cls

    @classmethod
    def get_registered_backends(cls):
        return ("ttx", "torch")


def test_perf_workload_infers_positional_args():
    workload = PerfWorkload(
        inputs={
            "x": tensor((2, 4), torch.float32),
            "weight": tensor((4,), torch.float32),
            "scale": tensor((2,), torch.float32),
            "mask": tensor((2, 4), torch.bool),
        },
        outputs={"output": tensor((2, 4), torch.float32)},
        state={"weight": "weight"},
        kwargs={"mask": "mask", "alpha": 0.5},
    )

    assert workload.args == ("x", "scale")


def test_perf_workload_keeps_explicit_arg_order():
    workload = PerfWorkload(
        inputs={
            "x": tensor((2, 4), torch.float32),
            "scale": tensor((2,), torch.float32),
        },
        outputs={"output": tensor((2, 4), torch.float32)},
        args=("scale", "x"),
    )

    assert workload.args == ("scale", "x")


def test_perf_case_serializes_torch_dtype():
    case = perf_case("dtype", dtype=torch.bfloat16)

    assert case.params["dtype"] is torch.bfloat16
    assert case.to_task("test_op")["dtype"] == "bfloat16"


def test_get_backend_impl_strict_rejects_fallback():
    with pytest.raises(KeyError, match="backend 'missing' is not registered"):
        MojoGelu.get_backend_impl("missing", strict=True)


def test_mojo_perf_rejects_bare_provider_string():
    with pytest.raises(TypeError, match="for one provider use"):
        mojo_perf(
            name="test_bare_provider",
            target=_DummyTarget,
            cases=(),
            providers="ttx",
        )


@pytest.mark.parametrize("provider", ("typo", "ixformer"))
def test_mojo_perf_inferred_rejects_unknown_profile_provider(provider):
    with pytest.raises(ValueError, match="profiling contains unregistered providers"):
        mojo_perf(
            name="test_unknown_profile_provider",
            target=_DummyTarget,
            cases=(),
            profiling={provider: profile()},
        )


def test_mojo_perf_infers_current_target_backends():
    spec = get_perf_spec("mojo_gelu")
    expected = [
        backend
        for backend in MojoGelu.get_registered_backends()
        if backend != spec.base_backend
    ]

    assert list(spec.providers) == expected


@pytest.mark.parametrize(
    ("platform", "backend"),
    (("npu", "NPU"), ("ilu", "ILU"), ("mlu", "MLU")),
)
def test_detect_xpu_backend_from_platform(monkeypatch, platform, backend):
    monkeypatch.setattr(
        "mojo_opset.benchmark.runner_common.get_platform",
        lambda: platform,
    )

    assert detect_xpu_backend((backend,)) == backend


def test_detect_xpu_backend_rejects_unavailable_backend(monkeypatch):
    monkeypatch.setattr(
        "mojo_opset.benchmark.runner_common.get_platform",
        lambda: "ilu",
    )

    with pytest.raises(RuntimeError, match="requires xpu-perf backend 'ILU'"):
        detect_xpu_backend(("NPU",))


def test_default_providers_include_base_and_follow_current_environment(monkeypatch):
    base_impl = object()
    torch_npu_impl = object()
    ttx_impl = object()
    monkeypatch.setitem(ProviderRegistry.BASE_IMPL_MAPPING, "mojo_gelu", base_impl)
    backend = type(
        "Backend",
        (),
        {
            "op_mapping": {
                "mojo_gelu": {
                    "ttx": ttx_impl,
                    "ixformer": object(),
                    "torch_npu": torch_npu_impl,
                }
            }
        },
    )()

    provider_map = build_provider_map(backend, "mojo_gelu", None)

    assert list(provider_map) == ["base", "ttx", "torch_npu"]
    assert provider_map == {
        "base": base_impl,
        "ttx": ttx_impl,
        "torch_npu": torch_npu_impl,
    }


def test_explicit_providers_can_exclude_base(monkeypatch):
    base_impl = object()
    ttx_impl = object()
    monkeypatch.setitem(ProviderRegistry.BASE_IMPL_MAPPING, "mojo_gelu", base_impl)
    backend = type("Backend", (), {"op_mapping": {"mojo_gelu": {"ttx": ttx_impl}}})()

    assert build_provider_map(backend, "mojo_gelu", None) == {
        "base": base_impl,
        "ttx": ttx_impl,
    }
    assert build_provider_map(backend, "mojo_gelu", ["ttx"]) == {"ttx": ttx_impl}
    assert build_provider_map(backend, "mojo_gelu", ["base", "ttx"]) == {
        "base": base_impl,
        "ttx": ttx_impl,
    }


def test_perf_workload_infers_backward_args_from_forward_args():
    workload = PerfWorkload(
        inputs={
            "x": tensor((2, 4), torch.float32),
            "weight": tensor((4,), torch.float32),
            "dy": tensor((2, 4), torch.float32),
        },
        outputs={"dx": tensor((2, 4), torch.float32)},
        forward_args=("x", "weight", 1e-6),
    )

    assert workload.forward_args == ("x", "weight", 1e-6)
    assert workload.args == ("dy",)


def test_perf_workload_validates_forward_arg_references():
    with pytest.raises(ValueError, match="undefined input tensors"):
        PerfWorkload(
            inputs={"dy": tensor((2, 4), torch.float32)},
            outputs={"dx": tensor((2, 4), torch.float32)},
            forward_args=("missing",),
        )


def test_mojo_perf_rejects_operator_backward():
    with pytest.raises(TypeError, match="only valid for Mojo Functions"):
        mojo_perf(
            name="_test_operator_backward",
            target=MojoGelu,
            cases=(),
            phase="backward",
        )


class _FakeBenchmarkBackend:
    enable_profiling = False

    def __init__(self):
        self.synchronize_calls = 0

    def get_torch_device_name(self):
        return "cpu"

    def device_synchronize(self):
        self.synchronize_calls += 1


def _register_backward_test_spec(name, *, run=None):
    case = perf_case("tiny", tags=("smoke",))

    @mojo_perf(
        name=name,
        target=MojoSiluFunction,
        cases=(case,),
        phase="backward",
        profiling=profile(timing="event"),
    )
    def workload(_case):
        return PerfWorkload(
            inputs={
                "x": tensor((2, 4), torch.float32, creator=torch.randn),
                "dy": tensor((2, 4), torch.float32, creator=torch.randn),
            },
            outputs={"dx": tensor((2, 4), torch.float32)},
            forward_args=("x",),
            run=run,
        )

    return case


def _make_test_adapter(name):
    return type(
        f"Adapter_{name}",
        (GeneratedMojoPerfAdapter,),
        {
            "perf_op_name": name,
            "mojo_backend": "torch",
            "xpu_provider": ProviderRegistry.BASE_PROVIDER,
        },
    )


@pytest.mark.parametrize(
    ("phase", "forward_args", "match"),
    (
        ("backward", None, "requires PerfWorkload.forward_args"),
        ("forward", ("x",), "only valid for Mojo Function backward"),
    ),
)
def test_function_phase_validates_forward_args(phase, forward_args, match):
    name = f"_test_function_{phase}_forward_args_{forward_args is not None}"
    case = perf_case("tiny")

    @mojo_perf(
        name=name,
        target=MojoSiluFunction,
        cases=(case,),
        phase=phase,
        profiling=profile(timing="event"),
    )
    def workload(_case):
        return PerfWorkload(
            inputs={
                "x": tensor((2, 4), torch.float32),
                "dy": tensor((2, 4), torch.float32),
            },
            outputs={"output": tensor((2, 4), torch.float32)},
            forward_args=forward_args,
        )

    with pytest.raises(ValueError, match=match):
        _make_test_adapter(name)(case.to_task(name), _FakeBenchmarkBackend())


def test_function_backward_prepares_distinct_contexts_outside_timing(monkeypatch):
    name = "_test_function_backward_lifecycle"
    case = _register_backward_test_spec(name)
    backend_impl = MojoSiluFunction.get_backend_impl("torch", strict=True)
    original_forward = backend_impl.forward
    original_backward = backend_impl.backward
    calls = []

    def spy_forward(ctx, x):
        calls.append(("forward", id(ctx)))
        return original_forward(ctx, x)

    def spy_backward(ctx, dy):
        calls.append(("backward", id(ctx)))
        return original_backward(ctx, dy)

    monkeypatch.setattr(backend_impl, "forward", staticmethod(spy_forward))
    monkeypatch.setattr(backend_impl, "backward", staticmethod(spy_backward))

    backend = _FakeBenchmarkBackend()
    adapter = _make_test_adapter(name)(case.to_task(name), backend)
    mappings = adapter.create_tensors(2)

    assert [kind for kind, _ in calls] == ["forward", "forward"]
    assert len({ctx_id for _, ctx_id in calls}) == 2
    assert backend.synchronize_calls == 1

    adapter.core_run(mappings[0])
    adapter.core_run(mappings[1])
    adapter.core_run(mappings[0])

    assert [kind for kind, _ in calls] == [
        "forward",
        "forward",
        "backward",
        "backward",
        "backward",
    ]
    assert calls[0][1] == calls[2][1] == calls[4][1]
    assert calls[1][1] == calls[3][1]


def test_function_backward_custom_run_receives_bound_target():
    observed = []

    def run(target, tensors):
        observed.append(target)
        return target(tensors["dy"])

    name = "_test_function_backward_custom_run"
    case = _register_backward_test_spec(name, run=run)
    backend = _FakeBenchmarkBackend()
    adapter = _make_test_adapter(name)(case.to_task(name), backend)
    mapping = adapter.create_tensors(1)[0]

    result = adapter.core_run(mapping)

    assert len(observed) == 1
    assert callable(observed[0])
    assert result.shape == mapping["x"].shape


def test_nested_descriptors_cover_operators_and_function_phases():
    production_specs = [
        spec for spec in iter_perf_specs() if not spec.name.startswith("_test_")
    ]
    operator_specs = [
        spec for spec in production_specs if issubclass(spec.target, MojoOperator)
    ]
    function_specs = [
        spec for spec in production_specs if issubclass(spec.target, MojoFunction)
    ]

    assert len(operator_specs) == 27
    assert len(function_specs) == 12
    assert {spec.phase for spec in operator_specs} == {"forward"}
    assert {spec.phase for spec in function_specs} == {"forward", "backward"}
    assert {spec.name for spec in function_specs} == {
        "mojo_apply_rope_function_backward",
        "mojo_apply_rope_function_forward",
        "mojo_causal_conv1d_function_backward",
        "mojo_causal_conv1d_function_forward",
        "mojo_fused_linear_cross_entropy_function_backward",
        "mojo_fused_linear_cross_entropy_function_forward",
        "mojo_rmsnorm_function_backward",
        "mojo_rmsnorm_function_forward",
        "mojo_silu_function_backward",
        "mojo_silu_function_forward",
        "mojo_swa_function_backward",
        "mojo_swa_function_forward",
    }


def test_generated_vendor_inherits_common_adapter(monkeypatch):
    target = SimpleNamespace(get_backend_impl=lambda backend, strict: object())
    spec = SimpleNamespace(
        name="_test_vendor_inheritance",
        providers={"ttx": SimpleNamespace(backend="ttx")},
        target=target,
    )
    monkeypatch.setattr(xpu_adapter, "iter_perf_specs", lambda: (spec,))
    monkeypatch.setattr(
        ProviderRegistry,
        "register_vendor_impl",
        lambda op_name, provider: lambda cls: cls,
    )

    assert xpu_adapter.register_vendor_specs("ttx") == 1
    generated = getattr(
        xpu_adapter,
        "XpuPerfVendor__test_vendor_inheritance_ttx",
    )
    assert generated.__bases__ == (GeneratedMojoPerfAdapter,)
