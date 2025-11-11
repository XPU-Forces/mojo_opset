import importlib
from mojo_opset.utils import 

device_type = "npu"  

try:
    import torch_npu
    device_type = "npu"
except ImportError as e:
    pass

try:
    import torch_mlu
    xpu_device_type = "mlu"
except ImportError as e:
    pass

try:
    import ixformer

    xpu_device_type = "ilu"
except ImportError as e:
    pass


try:
    import torch_xmlir
    xpu_device_type = "xpu"
except ImportError as e:
    pass


_PACKAGE_NAME = __name__                   # Current package name for relative imports
try:
    # Relative import: from .{device_type} import *
    module = importlib.import_module(f".{device_type}", package=_PACKAGE_NAME)
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Backend sub-module '{device_type}' not found; check backends/torch/{device_type}.py"
    ) from e


public_items = {k: v for k, v in vars(module).items() if not k.startswith("_")}
globals().update(public_items)

__all__.extend(public_items.keys())