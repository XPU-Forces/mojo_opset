"""Dynamic xpu-perf bridge for declarative Mojo TTX benchmarks."""

from mojo_opset.benchmark.xpu_adapter import register_vendor_specs

PROVIDER_NAME = "ttx"

register_vendor_specs(PROVIDER_NAME)

__all__ = ["PROVIDER_NAME"]
