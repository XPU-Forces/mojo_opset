import traceback
import importlib.metadata

from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch_npu"

try:
    import torch
    import torch_npu

    ProviderRegistry.register_provider_info("torch_npu", {
        "torch_npu": importlib.metadata.version("torch_npu")
    })

except Exception:
    traceback.print_exc()
