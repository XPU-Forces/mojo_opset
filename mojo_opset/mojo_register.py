import mojo_opset
from mojo_opset.core.mojo_operator import MojoOperator
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)

_CURRENT_REGISTERED_BACKEND = None
_CURRENT_REGISTERED_OPSET = {}


def MOJO_REGISTER_OPSET(mojo_op: MojoOperator, backend_op: MojoOperator, backend_name: str):
    """
    Register an backend to replace default impl.
    """
    logger.info(f"Registering {backend_op.__name__} as {mojo_op.__name__}")

    """
    Check if the backend is already registered.
    """
    global _CURRENT_REGISTERED_BACKEND
    if _CURRENT_REGISTERED_BACKEND is not None:
        assert backend_name.upper() == _CURRENT_REGISTERED_BACKEND
    else:
        _CURRENT_REGISTERED_BACKEND = backend_name.upper()

    reg_opname = mojo_op.__name__[len("Mojo") :]
    bak_opname = backend_op.__name__[len(backend_name) :]

    global _CURRENT_REGISTERED_OPSET
    assert reg_opname not in _CURRENT_REGISTERED_OPSET
    assert reg_opname == bak_opname

    _CURRENT_REGISTERED_OPSET[reg_opname] = backend_op
    setattr(mojo_opset, mojo_op.__name__, backend_op)

    logger.info(f"Register Done! {getattr(mojo_opset, mojo_op.__name__)}")
