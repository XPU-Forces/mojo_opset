from mojo_opset.utils.logging import get_logger
from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.misc import get_bool_env

logger = get_logger(__name__)

platform = get_platform()

_SUPPORT_TTX_PLATFROM = ["npu", "ilu", "mlu"]
_SUPPORT_TORCH_NPU_PLATFROM = ["npu"]
_SUPPORT_ASCENDC_PLATFORM = ["npu"]
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

if platform in _SUPPORT_ASCENDC_PLATFORM:
    from .ascendc import *

if platform == "npu" and get_bool_env("MOJO_DETERMINISTIC", default=False):
    import os
    
    # special setting for npu deterministic matmul
    os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
