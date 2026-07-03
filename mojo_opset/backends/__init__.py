from mojo_opset.utils.logging import get_logger
from mojo_opset.utils.misc import configure_torch_deterministic
from mojo_opset.utils.misc import get_bool_env
from mojo_opset.utils.platform import get_platform

logger = get_logger(__name__)

platform = get_platform()
is_deterministic = get_bool_env("MOJO_DETERMINISTIC", default=False)

if is_deterministic:
    if platform == "npu":
        import os

        # Default for deterministic NPU matmul; respect explicit user/CANN settings.
        os.environ.setdefault("CLOSE_MATMUL_K_SHIFT", "1")
    configure_torch_deterministic()

_SUPPORT_TTX_PLATFROM = ["npu", "ilu", "mlu"]
_SUPPORT_TORCH_NPU_PLATFROM = ["npu"]
_SUPPORT_UC_PLATFORM = ["npu"]
_SUPPORT_IXFORMER_PLATFORM = ["ilu"]

if platform in _SUPPORT_IXFORMER_PLATFORM:
    try:
        from .ixformer import *
    except ImportError as e:
        logger.warning("Skipping ixformer backend (import failed): %s", e)

if platform in _SUPPORT_TTX_PLATFROM:
    try:
        from .ttx import *
    except Exception as e:
        logger.warning("Skipping ttx backend (import failed): %s", e)

if platform in _SUPPORT_TORCH_NPU_PLATFROM:
    from .torch_npu import *

if platform in _SUPPORT_UC_PLATFORM:
    try:
        from .uc import *
    except ImportError as e:
        logger.warning("Skipping uc backend (import failed): %s", e)
