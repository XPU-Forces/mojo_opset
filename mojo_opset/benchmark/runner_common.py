"""Shared helpers for the benchmark runners."""

from __future__ import annotations

import pathlib
from typing import Any
from typing import Mapping

from xpu_perf.micro_perf.core.op import ProviderRegistry

from mojo_opset.benchmark import build_test_cases
from mojo_opset.benchmark.api import get_perf_spec
from mojo_opset.utils.platform import get_platform

HERE = pathlib.Path(__file__).parent.absolute()
DESCRIPTOR_OP_DEFS = HERE / "plugins" / "op_defs"
DESCRIPTOR_VENDOR_OPS = HERE / "plugins" / "vendor_ops"
DEFAULT_PRESET = "smoke"

BASE_PROVIDER = ProviderRegistry.BASE_PROVIDER  # "base"
PLATFORM_XPU_BACKEND = {
    "npu": "NPU",
    "ilu": "ILU",
    "mlu": "MLU",
}


def select_plugin_paths() -> tuple[pathlib.Path, pathlib.Path]:
    return DESCRIPTOR_OP_DEFS, DESCRIPTOR_VENDOR_OPS


def detect_xpu_backend(available_backends=None) -> str:
    """Map the current Mojo platform to an available xpu-perf backend."""

    platform = get_platform()
    backend = PLATFORM_XPU_BACKEND.get(platform)
    if backend is None:
        raise RuntimeError(
            f"cannot infer an xpu-perf backend from Mojo platform {platform!r}; "
            "run in an accelerator environment or pass --backend explicitly"
        )

    if available_backends is not None and backend not in available_backends:
        raise RuntimeError(
            f"Mojo detected platform {platform!r}, which requires xpu-perf backend "
            f"{backend!r}, but the available backends are {list(available_backends)}"
        )
    return backend


def parse_requested_providers(value: str | None) -> list[str] | None:
    """Keep ``None`` distinct from an explicitly supplied provider list."""

    if value is None:
        return None
    providers = [name.strip() for name in value.split(",") if name.strip()]
    if len(providers) != len(set(providers)):
        raise ValueError(f"provider names must be unique: {providers}")
    return providers


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


def build_provider_map(
    backend,
    op_name: str,
    requested: list[str] | None,
) -> dict:
    """Resolve providers available for one target in this environment.

    Omitting ``--providers`` selects ``base`` followed by the vendor
    providers resolved by the descriptor. Passing ``--providers`` is a
    strict filter and may explicitly exclude ``base``.
    """

    vendor_available = backend.op_mapping.get(op_name, {})
    available = {}
    if op_name in ProviderRegistry.BASE_IMPL_MAPPING:
        available[BASE_PROVIDER] = ProviderRegistry.BASE_IMPL_MAPPING[op_name]

    if requested is None:
        spec = get_perf_spec(op_name)
        available.update(
            {
                name: vendor_available[name]
                for name in spec.providers
                if name in vendor_available
            }
        )
        return available

    available.update(vendor_available)

    return {name: available[name] for name in requested if name in available}
