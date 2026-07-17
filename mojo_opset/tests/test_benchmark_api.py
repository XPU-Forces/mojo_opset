import pytest
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry

from mojo_opset import MojoGelu
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor
from mojo_opset.benchmark.api import get_perf_spec
from mojo_opset.benchmark.runner_common import build_provider_map
from mojo_opset.benchmark.runner_common import detect_xpu_backend


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
