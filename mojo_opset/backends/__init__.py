import os

from mojo_opset.utils.platform import get_platform
from mojo_opset.utils.misc import get_bool_env

platform = get_platform()

_SUPPORT_TTX_PLATFROM = ["npu", "ilu", "mlu"]
_SUPPORT_TORCH_NPU_PLATFROM = ["npu"]

if platform in _SUPPORT_TTX_PLATFROM:
    from .ttx import *

if platform in _SUPPORT_TORCH_NPU_PLATFROM:
    from .torch_npu import *

if platform == "npu" and get_bool_env("MOJO_DETERMINISTIC", default=False):
    # special setting for npu deterministic matmul
    os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"