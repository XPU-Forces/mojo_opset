import os
from mojo_opset.utils.platform import get_platform

platform = get_platform()

_SUPPORT_TTX_PLATFROM = ["npu"]

if (platform in _SUPPORT_TTX_PLATFROM and 
    os.getenv("MOJO_USE_TTX_BACKEND", "0") == "1"):
    from .ttx import *
else:
    from .torch_npu import *
