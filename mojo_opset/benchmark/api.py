"""Small, xpu-perf-independent API used by per-operator perf descriptions."""

from __future__ import annotations

import importlib
import json
import pkgutil
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence

import torch


TensorCreator = Callable[..., torch.Tensor]
TensorFactory = Callable[[str], Mapping[str, torch.Tensor]]
TargetFactory = Callable[[type, str], Callable[..., Any]]
WorkloadRunner = Callable[[Callable[..., Any], Mapping[str, torch.Tensor]], Any]
WorkloadBuilder = Callable[[Mapping[str, Any]], "PerfWorkload"]
ProviderSupport = Callable[[Mapping[str, Any]], bool]
TimingMode = Literal["profiler", "event"]
KernelMatch = Literal["exact", "contains", "regex"]
LatencyReduction = Literal["span", "sum"]
FunctionPhase = Literal["forward", "backward"]


def _serialize_case_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {key: _serialize_case_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize_case_value(item) for item in value]
    return value


@dataclass(frozen=True)
class PerfCase:
    """One benchmark case; torch dtypes are serialized for xpu-perf tasks."""

    id: str
    params: Mapping[str, Any]
    tags: tuple[str, ...] = ()

    def to_task(self, op_name: str) -> dict[str, Any]:
        reserved = {"__case_id__", "__perf_spec__", "__tags__", "__perf_timing__"}
        overlap = reserved.intersection(self.params)
        if overlap:
            raise ValueError(f"case {self.id!r} uses reserved fields: {sorted(overlap)}")

        task = {
            "__case_id__": self.id,
            "__perf_spec__": op_name,
            "__tags__": list(self.tags),
            **{key: _serialize_case_value(value) for key, value in self.params.items()},
        }
        try:
            json.dumps(task)
        except TypeError as err:
            raise TypeError(
                f"benchmark case {op_name}/{self.id} must contain only JSON-serializable values"
            ) from err
        return task


def perf_case(case_id: str, *, tags: Sequence[str] = (), **params: Any) -> PerfCase:
    """Create a case without requiring an operator-specific case class."""

    return PerfCase(id=case_id, params=dict(params), tags=tuple(tags))


@dataclass(frozen=True)
class TensorSpec:
    """Describe a tensor; xpu-perf owns its allocation and cloning."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    creator: TensorCreator | None = None
    device: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"tensor shape must be non-negative, got {self.shape}")
        if self.creator is not None and not callable(self.creator):
            raise TypeError("TensorSpec.creator must be callable")


def tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    creator: TensorCreator | None = None,
    device: str | None = None,
) -> TensorSpec:
    return TensorSpec(shape=tuple(shape), dtype=dtype, creator=creator, device=device)


@dataclass(frozen=True)
class LiteralArg:
    """Wrap a literal string so it is not interpreted as a tensor reference."""

    value: Any


def literal(value: Any) -> LiteralArg:
    """Mark a call argument as a literal value."""

    return LiteralArg(value)


@dataclass(frozen=True)
class PerfWorkload:
    """Provider-independent construction and invocation description.

    String values in ``args`` and ``kwargs`` reference tensors by name; other
    values are passed literally. When ``args`` is omitted, input tensors are
    passed positionally in declaration order, excluding tensors bound through
    ``state`` or referenced by ``kwargs``. Use ``literal("value")`` for a
    literal string. For Function backward, ``forward_args`` are used once
    outside timing to prepare the context; inferred ``args`` exclude those
    tensors. A custom ``run`` receives the measured callable: an Operator
    instance, Function ``apply``, or context-bound Function ``backward``.
    """

    inputs: Mapping[str, TensorSpec]
    outputs: Mapping[str, TensorSpec]
    op_kwargs: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, str] = field(default_factory=dict)
    forward_args: tuple[Any, ...] | None = None
    args: tuple[Any, ...] | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    flops: int = 0
    create_outputs: bool = False
    tensor_factory: TensorFactory | None = None
    target_factory: TargetFactory | None = None
    run: WorkloadRunner | None = None
    read_bytes: int | float | None = None
    write_bytes: int | float | None = None

    def __post_init__(self):
        if self.forward_args is not None:
            object.__setattr__(self, "forward_args", tuple(self.forward_args))

        if self.args is None:
            if self.run is None:
                keyword_tensors = {
                    value for value in self.kwargs.values() if isinstance(value, str)
                }
                forward_tensors = {
                    value
                    for value in (self.forward_args or ())
                    if isinstance(value, str)
                }
                omitted = set(self.state.values()) | keyword_tensors | forward_tensors
                inferred_args = tuple(name for name in self.inputs if name not in omitted)
            else:
                inferred_args = ()
            object.__setattr__(self, "args", inferred_args)
        else:
            object.__setattr__(self, "args", tuple(self.args))

        input_names = set(self.inputs)
        references = {
            value for value in (*self.args, *self.kwargs.values()) if isinstance(value, str)
        }
        references.update(
            value for value in (self.forward_args or ()) if isinstance(value, str)
        )
        references.update(self.state.values())
        missing = references - input_names
        if missing:
            raise ValueError(f"workload references undefined input tensors: {sorted(missing)}")
        if self.flops < 0:
            raise ValueError("PerfWorkload.flops must be non-negative")
        if self.tensor_factory is not None and not callable(self.tensor_factory):
            raise TypeError("PerfWorkload.tensor_factory must be callable")
        if self.target_factory is not None and not callable(self.target_factory):
            raise TypeError("PerfWorkload.target_factory must be callable")
        if self.run is None and not self.args and not self.kwargs:
            raise ValueError("workload must provide args/kwargs or a custom run function")


@dataclass(frozen=True)
class ProfileSpec:
    """How xpu-perf measures one provider.

    Profiler timing is the default. ``span`` measures the wall-time interval
    covered by the selected kernels; ``sum`` adds their individual durations.
    Every requested kernel selector is required to match at least one event.
    """

    timing: TimingMode = "profiler"
    kernels: tuple[str, ...] | None = None
    match: KernelMatch = "exact"
    reduction: LatencyReduction = "span"

    def __post_init__(self):
        if self.timing not in ("profiler", "event"):
            raise ValueError("profile timing must be 'profiler' or 'event'")
        if self.match not in ("exact", "contains", "regex"):
            raise ValueError("kernel match must be 'exact', 'contains', or 'regex'")
        if self.reduction not in ("span", "sum"):
            raise ValueError("profile reduction must be 'span' or 'sum'")

        if self.kernels is not None:
            kernels = tuple(self.kernels)
            if not kernels or any(not kernel for kernel in kernels):
                raise ValueError("profile kernels must contain at least one non-empty selector")
            if len(kernels) != len(set(kernels)):
                raise ValueError("profile kernel selectors must be unique")
            if self.match == "regex":
                for pattern in kernels:
                    re.compile(pattern)
            object.__setattr__(self, "kernels", kernels)

        if self.timing == "event" and (self.kernels is not None or self.reduction != "span"):
            raise ValueError("event timing does not support kernel selection or reduction")


def profile(
    *,
    timing: TimingMode = "profiler",
    kernels: Sequence[str] | None = None,
    match: KernelMatch = "exact",
    reduction: LatencyReduction = "span",
) -> ProfileSpec:
    """Create profiling options for ``mojo_perf``."""

    return ProfileSpec(
        timing=timing,
        kernels=None if kernels is None else tuple(kernels),
        match=match,
        reduction=reduction,
    )


@dataclass(frozen=True)
class PerfProviderSpec:
    """Describe one xpu-perf provider and the cases its backend supports."""

    backend: str
    supports: ProviderSupport | None = None
    unsupported_reason: str | None = None

    def __post_init__(self):
        if not self.backend:
            raise ValueError("provider backend must be non-empty")
        if self.supports is not None and not callable(self.supports):
            raise TypeError("provider supports must be callable")
        if self.supports is None and self.unsupported_reason is not None:
            raise ValueError("unsupported_reason requires a supports predicate")

    def supports_case(self, case: Mapping[str, Any]) -> bool:
        return self.supports is None or bool(self.supports(case))


def perf_provider(
    backend: str,
    *,
    supports: ProviderSupport | None = None,
    unsupported_reason: str | None = None,
) -> PerfProviderSpec:
    """Create a provider declaration with an optional case capability check."""

    return PerfProviderSpec(
        backend=backend,
        supports=supports,
        unsupported_reason=unsupported_reason,
    )


@dataclass(frozen=True)
class PerfTargetSpec:
    name: str
    target: type
    cases: tuple[PerfCase, ...]
    build: WorkloadBuilder
    base_backend: str
    providers: Mapping[str, PerfProviderSpec]
    profiles: Mapping[str, ProfileSpec]
    engine: str
    phase: FunctionPhase

    def resolve_case_params(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        case_id = task.get("__case_id__")
        if case_id is None:
            raise ValueError(f"benchmark task for {self.name!r} is missing __case_id__")
        for case in self.cases:
            if case.id == case_id:
                return case.params
        raise ValueError(
            f"benchmark task references unknown case {self.name}/{case_id}; "
            f"available: {[case.id for case in self.cases]}"
        )


_SPECS: dict[str, PerfTargetSpec] = {}
_DISCOVERED = False


def mojo_perf(
    *,
    name: str,
    cases: Sequence[PerfCase],
    target: type,
    providers: (
        Mapping[str, str | PerfProviderSpec] | Iterable[str | PerfProviderSpec] | None
    ) = None,
    base_backend: str = "torch",
    profiling: ProfileSpec | Mapping[str, ProfileSpec] | None = None,
    engine: str = "ComputeEngine",
    phase: FunctionPhase = "forward",
):
    """Register the complete perf description of a Mojo Operator or Function.

    By default, vendor providers are inferred from the target's backends
    registered for the current platform. Passing ``providers`` explicitly
    replaces inference with an allowlist and optional capability declarations.
    Function descriptors may set ``phase="backward"`` for direct backward
    timing; Operator descriptors only support the default forward phase.
    """

    if (
        not isinstance(target, type)
        or not callable(getattr(target, "get_backend_impl", None))
        or not callable(getattr(target, "get_registered_backends", None))
    ):
        raise TypeError("mojo_perf target must be a Mojo Operator or Function class")
    if phase not in ("forward", "backward"):
        raise ValueError("mojo_perf phase must be 'forward' or 'backward'")
    if phase == "backward":
        from mojo_opset.core.function import MojoFunction

        if not issubclass(target, MojoFunction):
            raise TypeError("mojo_perf phase='backward' is only valid for Mojo Functions")

    infer_providers = providers is None
    if infer_providers:
        raw_provider_map = {
            backend: backend
            for backend in target.get_registered_backends()
            if backend != base_backend
        }
    elif isinstance(providers, Mapping):
        raw_provider_map = dict(providers)
    else:
        if isinstance(providers, (str, PerfProviderSpec)):
            raise TypeError(
                "providers must be a tuple or list; for one provider use "
                "providers=('ttx',)"
            )
        declarations = tuple(providers)
        raw_provider_map = {}
        for declaration in declarations:
            if isinstance(declaration, str):
                provider_name = declaration
            elif isinstance(declaration, PerfProviderSpec):
                provider_name = declaration.backend
            else:
                raise TypeError(
                    "providers must contain backend names or PerfProviderSpec values"
                )
            if provider_name in raw_provider_map:
                raise ValueError(f"duplicate provider declaration {provider_name!r}")
            raw_provider_map[provider_name] = declaration

    provider_map = {}
    for provider_name, declaration in raw_provider_map.items():
        if isinstance(declaration, str):
            provider_map[provider_name] = PerfProviderSpec(backend=declaration)
        elif isinstance(declaration, PerfProviderSpec):
            provider_map[provider_name] = declaration
        else:
            raise TypeError(
                f"provider {provider_name!r} must map to a backend name or PerfProviderSpec"
            )
    if "base" in provider_map:
        raise ValueError("'base' is reserved by xpu-perf; use base_backend='torch'")

    provider_names = {"base", *provider_map}
    if profiling is None:
        profile_map = {provider: ProfileSpec() for provider in provider_names}
    elif isinstance(profiling, ProfileSpec):
        profile_map = {provider: profiling for provider in provider_names}
    else:
        unknown_profiles = set(profiling) - provider_names
        if unknown_profiles:
            raise ValueError(f"profiling contains unregistered providers: {sorted(unknown_profiles)}")
        invalid_profiles = {
            provider: value
            for provider, value in profiling.items()
            if not isinstance(value, ProfileSpec)
        }
        if invalid_profiles:
            raise TypeError(f"profiling values must be ProfileSpec: {sorted(invalid_profiles)}")
        profile_map = {
            provider: profiling.get(provider, ProfileSpec()) for provider in provider_names
        }

    def decorator(builder: WorkloadBuilder):
        case_tuple = tuple(cases)
        case_ids = [case.id for case in case_tuple]
        duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        if duplicate_ids:
            raise ValueError(f"duplicate benchmark case ids for {name!r}: {duplicate_ids}")

        spec = PerfTargetSpec(
            name=name,
            target=target,
            cases=case_tuple,
            build=builder,
            base_backend=base_backend,
            providers=provider_map,
            profiles=profile_map,
            engine=engine,
            phase=phase,
        )
        previous = _SPECS.get(name)
        if previous is not None and previous.build is not builder:
            raise ValueError(f"duplicate mojo_perf registration for {name!r}")
        _SPECS[name] = spec
        return builder

    return decorator


def discover_perf_specs(package_name: str = "mojo_opset.tests.perf_new") -> None:
    """Import all descriptor modules once; import failures are intentionally fatal."""

    global _DISCOVERED
    if _DISCOVERED:
        return

    package = importlib.import_module(package_name)
    module_names = sorted(
        module.name for module in pkgutil.walk_packages(
            package.__path__, prefix=f"{package.__name__}."
        )
    )
    for module_name in module_names:
        importlib.import_module(module_name)
    _DISCOVERED = True


def get_perf_spec(name: str) -> PerfTargetSpec:
    discover_perf_specs()
    try:
        return _SPECS[name]
    except KeyError as err:
        raise KeyError(f"unknown Mojo perf spec {name!r}; available: {sorted(_SPECS)}") from err


def iter_perf_specs() -> tuple[PerfTargetSpec, ...]:
    discover_perf_specs()
    return tuple(_SPECS[name] for name in sorted(_SPECS))


def build_test_cases(
    preset: str | None = "smoke",
    op_names: Iterable[str] | None = None,
    timing: TimingMode | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Compile descriptors into the plain task dictionary consumed by xpu-perf."""

    requested = set(op_names) if op_names is not None else None
    specs = iter_perf_specs()
    available_names = {spec.name for spec in specs}
    if requested is not None and requested - available_names:
        raise ValueError(f"unknown perf ops: {sorted(requested - available_names)}")

    all_tags = {tag for spec in specs for case in spec.cases for tag in case.tags}
    if preset not in (None, "all") and preset not in all_tags:
        raise ValueError(f"unknown preset {preset!r}; available: {sorted(all_tags | {'all'})}")

    result = {}
    for spec in specs:
        if requested is not None and spec.name not in requested:
            continue
        selected = []
        for case in spec.cases:
            if preset not in (None, "all") and preset not in case.tags:
                continue
            task = case.to_task(spec.name)
            if timing is not None:
                if timing not in ("profiler", "event"):
                    raise ValueError("timing override must be 'profiler' or 'event'")
                task["__perf_timing__"] = timing
            selected.append(task)
        if selected:
            result[spec.name] = selected
    return result
