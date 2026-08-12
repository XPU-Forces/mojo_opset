"""Device-neutral xpu-perf bridge for declarative Mojo benchmarks."""

import importlib
import importlib.metadata

from xpu_perf.micro_perf.core.op import ProviderRegistry

from mojo_opset.benchmark.xpu_adapter import register_available_vendor_specs

# xpu-perf uses this literal to identify the plugin package. Actual provider
# names are registered dynamically below.
PROVIDER_NAME = "mojo_descriptor_vendor_ops"

REGISTERED_PROVIDERS = register_available_vendor_specs()

for provider in REGISTERED_PROVIDERS:
    try:
        version = importlib.metadata.version(provider)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(provider)
            version = getattr(module, "__version__", None)
        except Exception:  # Provider metadata is optional for benchmark reports.
            version = None
    if version is not None:
        ProviderRegistry.register_provider_info(provider, {provider: version})

__all__ = ["PROVIDER_NAME", "REGISTERED_PROVIDERS"]
