from .reference import *

from mojo_opset.utils.platform import get_platform

platform = get_platform()

_SUPPORT_TTX_PLATFROM_ = ["npu"]

if platform in _SUPPORT_TTX_PLATFROM_:
    from .ttx import *
