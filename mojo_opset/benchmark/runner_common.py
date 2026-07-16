"""Shared helpers for the benchmark runners."""

from __future__ import annotations

import pathlib
from typing import Any
from typing import Mapping

from xpu_perf.micro_perf.core.op import ProviderRegistry

from mojo_opset.benchmark import build_test_cases
from mojo_opset.benchmark.api import get_perf_spec

HERE = pathlib.Path(__file__).parent.absolute()
DESCRIPTOR_OP_DEFS = HERE / "plugins" / "op_defs"
DESCRIPTOR_VENDOR_OPS = HERE / "plugins" / "vendor_ops" / "NPU" / "ops"
DEFAULT_PRESET = "smoke"

BASE_PROVIDER = ProviderRegistry.BASE_PROVIDER  # "base"


def select_plugin_paths() -> tuple[pathlib.Path, pathlib.Path]:
    return DESCRIPTOR_OP_DEFS, DESCRIPTOR_VENDOR_OPS


def resolve_test_cases(
    preset: str | None = DEFAULT_PRESET,
    op_names: list[str] | None = None,
    timing: str | None = None,
) -> dict[str, list[dict]]:
    return build_test_cases(preset=preset, op_names=op_names, timing=timing)


def case_provider_support(
    op_name: str,
    provider: str,
    case: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Return whether one descriptor case is supported by a provider."""

    if provider == BASE_PROVIDER:
        return True, None

    spec = get_perf_spec(op_name)
    provider_spec = spec.providers.get(provider)
    if provider_spec is None:
        return False, f"provider {provider!r} is not declared by the descriptor"

    params = spec.resolve_case_params(case)
    try:
        supported = provider_spec.supports_case(params)
    except Exception as err:
        case_id = case.get("__case_id__", "<unknown>")
        raise ValueError(
            f"provider capability check failed for {op_name}/{case_id}/{provider}: {err}"
        ) from err

    if supported:
        return True, None
    reason = provider_spec.unsupported_reason or "capability predicate rejected the case"
    return False, reason


def build_provider_map(backend, op_name: str, requested: list[str]) -> dict:
    """Map only requested providers, pulling ``base`` from BASE_IMPL_MAPPING."""

    available = {}
    if op_name in ProviderRegistry.BASE_IMPL_MAPPING:
        available[BASE_PROVIDER] = ProviderRegistry.BASE_IMPL_MAPPING[op_name]
    available.update(backend.op_mapping.get(op_name, {}))

    if requested:
        return {name: available[name] for name in requested if name in available}
    return available
