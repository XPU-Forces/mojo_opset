"""Dynamic xpu-perf base bridge for declarative Mojo benchmark specs."""

from mojo_opset.benchmark.xpu_adapter import register_base_specs

PROVIDER_NAME = "mojo_descriptor_base_ops"

register_base_specs()

__all__ = ["PROVIDER_NAME"]
