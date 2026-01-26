import os
from mojo_opset.utils.platform import get_platform

platform = get_platform()

_SUPPORT_TTX_PLATFROM = ["npu"]

if platform in _SUPPORT_TTX_PLATFROM:
    from .ttx import *
    from .torch_npu import *