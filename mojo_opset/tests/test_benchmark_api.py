import pytest
import torch

from mojo_opset import MojoGelu
from mojo_opset.benchmark import PerfWorkload
from mojo_opset.benchmark import mojo_perf
from mojo_opset.benchmark import perf_case
from mojo_opset.benchmark import tensor


class _DummyTarget:
    @classmethod
    def get_backend_impl(cls, backend):
        return cls


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
