"""Dynamic xpu-perf bridge for declarative Mojo torch_npu benchmarks."""

import importlib.metadata
import traceback

from xpu_perf.micro_perf.core.op import ProviderRegistry

from mojo_opset.benchmark.xpu_adapter import register_vendor_specs

PROVIDER_NAME = "torch_npu"

try:
    import torch_npu

    try:
        torch_npu_version = importlib.metadata.version("torch_npu")
    except importlib.metadata.PackageNotFoundError:
        torch_npu_version = getattr(torch_npu, "__version__", "unknown")

    ProviderRegistry.register_provider_info("torch_npu", {"torch_npu": torch_npu_version})
except Exception:
    traceback.print_exc()

register_vendor_specs(PROVIDER_NAME)

__all__ = ["PROVIDER_NAME"]
