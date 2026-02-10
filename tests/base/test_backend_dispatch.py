import pytest

from tests.utils import bypass_not_implemented

from mojo_opset import MojoSilu
from mojo_opset import MojoSiluFunction


@pytest.mark.parametrize(
    "MojoOpCls",
    [MojoSilu],
)
@bypass_not_implemented
def test_operator_dispatch(MojoOpCls):
    op_default = MojoOpCls()
    TTXOpCls = MojoOpCls._registry.get("ttx")
    op_ttx = TTXOpCls()
    assert type(op_default) == type(op_ttx) == TTXOpCls and TTXOpCls.__name__.startswith("TTX")

    TorchOpCls = MojoOpCls._registry.get("torch")
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
    func_ttx = MojoFunc._registry.get("ttx")
    assert func_default.forward == func_ttx.forward and func_default.backward == func_ttx.backward

    func_torch = MojoFunc._registry.get("torch")
    assert func_default.forward != func_torch.forward and func_default.backward != func_torch.backward
