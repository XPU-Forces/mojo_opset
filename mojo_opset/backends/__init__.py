import os

from mojo_opset.utils.platform import get_platform

platform = get_platform()

_SUPPORT_TTX_PLATFROM = ["npu", "ilu"]
_SUPPORT_TORCH_NPU_PLATFROM = ["npu"]

if platform in _SUPPORT_TTX_PLATFROM:
    from .ttx import *

if platform in _SUPPORT_TORCH_NPU_PLATFROM:
    from .torch_npu import *
