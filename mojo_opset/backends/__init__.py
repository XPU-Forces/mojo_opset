from mojo_opset.utils.logging import get_logger
from mojo_opset.utils.platform import get_platform

logger = get_logger(__name__)

platform = get_platform()

_SUPPORT_TTX_PLATFROM = ["npu", "ilu", "mlu"]
_SUPPORT_TORCH_NPU_PLATFROM = ["npu"]
_SUPPORT_IXFORMER_PLATFORM = ["ilu"]

if platform in _SUPPORT_IXFORMER_PLATFORM:
    try:
        from .ixformer import *
    except ImportError as e:
        logger.warning("Skipping ixformer backend (import failed): %s", e)

if platform in _SUPPORT_TTX_PLATFROM:
    from .ttx import *

if platform in _SUPPORT_TORCH_NPU_PLATFROM:
    from .torch_npu import *
