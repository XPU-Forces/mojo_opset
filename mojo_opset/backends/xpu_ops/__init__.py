import xpu_ops

from .add_norm import XOpsResidualAddNorm
from .gelu_act import XOpsGelu
from .norm import XOpsNorm
from .silu_func import XOpsSiluFunction

xpu_ops.load_xpu_ops()

__all__ = ["XOpsNorm", "XOpsResidualAddNorm", "XOpsGelu", "XOpsSiluFunction"]
