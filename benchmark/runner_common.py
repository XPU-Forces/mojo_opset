"""Shared helpers for benchmark runners (perf / accuracy)."""
from __future__ import annotations

import pathlib

from xpu_perf.micro_perf.core.op import ProviderRegistry

HERE = pathlib.Path(__file__).parent.absolute()
OP_DEFS = HERE / "op_defs"
VENDOR_OPS = HERE / "vendor_ops" / "NPU" / "ops"
DEFAULT_WORKLOAD = HERE / "workloads" / "mojo_quant_gemm.json"

BASE_PROVIDER = ProviderRegistry.BASE_PROVIDER  # "base"


def build_provider_map(backend, op_name: str, requested: list[str]) -> dict:
    """Map provider name -> op class, pulling ``base`` from BASE_IMPL_MAPPING."""
    available = {}
    if op_name in ProviderRegistry.BASE_IMPL_MAPPING:
        available[BASE_PROVIDER] = ProviderRegistry.BASE_IMPL_MAPPING[op_name]
    available.update(backend.op_mapping.get(op_name, {}))

    ordered = {}
    for name in requested:
        if name in available:
            ordered[name] = available[name]
    for name, cls in available.items():
        ordered.setdefault(name, cls)
    return ordered
