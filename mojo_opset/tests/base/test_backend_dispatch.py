import pytest

from mojo_opset.tests.utils import BackendNotImplementedForTest
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import resolve_backend_for_accuracy_test

from mojo_opset import MojoSilu
from mojo_opset import MojoSiluFunction


@pytest.mark.parametrize(
    "MojoOpCls",
    [MojoSilu],
)
@bypass_not_implemented
def test_operator_dispatch(MojoOpCls):
    op_default = MojoOpCls()
    registry = MojoOpCls.get_registry()
    assert registry is MojoOpCls._registry
    assert type(op_default) == MojoOpCls.get_backend_impl()

    TTXOpCls = MojoOpCls.get_backend_impl("ttx")
    assert TTXOpCls is registry.get("ttx")
    if "ttx" in registry._registry:
        op_ttx = TTXOpCls()
        assert type(op_default) == type(op_ttx) == TTXOpCls and TTXOpCls.__name__.startswith("TTX")

    TorchOpCls = MojoOpCls.get_backend_impl("torch")
    assert TorchOpCls is registry.get("torch")
    assert registry.get(" Torch ") is TorchOpCls
    assert MojoOpCls.get_backend_impl(" Torch ") is TorchOpCls
    if "torch_npu" in registry._registry:
        assert MojoOpCls.get_backend_impl("torchnpu") is registry.get("torch_npu")
    op_torch = TorchOpCls()
    assert (
        type(op_torch) == TorchOpCls
        and op_torch.forward.__code__ == MojoOpCls.forward.__code__
        and TorchOpCls.__name__.startswith("Torch")
        and TorchOpCls.forward == MojoOpCls.forward
    )


@pytest.mark.parametrize(
    "MojoFunc",
    [MojoSiluFunction],
)
@bypass_not_implemented
def test_function_dispatch(MojoFunc):
    func_default = MojoFunc
    registry = MojoFunc.get_registry()
    assert registry is MojoFunc._registry

    func_ttx = MojoFunc.get_backend_impl("ttx")
    assert func_ttx is registry.get("ttx")
    if "ttx" in registry._registry:
        assert func_default.forward == func_ttx.forward and func_default.backward == func_ttx.backward

    func_torch = MojoFunc.get_backend_impl("torch")
    assert func_torch is registry.get("torch")
    assert registry.get(" Torch ") is func_torch
    assert MojoFunc.get_backend_impl(" Torch ") is func_torch
    if "torch_npu" in registry._registry:
        assert MojoFunc.get_backend_impl("torchnpu") is registry.get("torch_npu")
    if "ttx" in registry._registry:
        assert func_default.forward != func_torch.forward and func_default.backward != func_torch.backward
    else:
        assert func_default.forward == func_torch.forward and func_default.backward == func_torch.backward


def test_accuracy_helper_raises_when_requested_backend_missing(monkeypatch):
    monkeypatch.setenv("MOJO_BACKEND", "fake_backend")

    with pytest.raises(BackendNotImplementedForTest, match="Silu"):
        resolve_backend_for_accuracy_test(MojoSilu._registry)
