"""Generate xpu-perf base and vendor classes from Mojo perf descriptions."""

from __future__ import annotations

import re
from functools import partial
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

import torch

from xpu_perf.micro_perf.core.op import BasicOp
from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.profiling_result import compute_latency_us
from xpu_perf.micro_perf.core.utils import OpTensorInfo
from xpu_perf.micro_perf.core.utils import calc_tensor_size

from mojo_opset.core.function import MojoFunction
from mojo_opset.core.operator import MojoOperator

from .api import LiteralArg
from .api import PerfWorkload
from .api import ProfileSpec
from .api import TensorSpec
from .api import get_perf_spec
from .api import iter_perf_specs


def _tensor_info(spec: TensorSpec, default_device: str) -> OpTensorInfo:
    kwargs = {
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "device": spec.device or default_device,
    }
    if spec.creator is not None:
        kwargs["creator"] = spec.creator
    return OpTensorInfo(**kwargs)


def _resolve_call_arg(value: Any, tensor_mapping: Mapping[str, torch.Tensor]) -> Any:
    if isinstance(value, LiteralArg):
        return value.value
    if isinstance(value, str):
        return tensor_mapping[value]
    return value


class _BenchmarkFunctionContext:
    """Minimal autograd context used to isolate direct Function.backward timing."""

    def __init__(self):
        self.saved_tensors: tuple[torch.Tensor | None, ...] = ()

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None:
        self.saved_tensors = tensors


class KernelSelectionError(ValueError):
    """A requested profiler kernel selector did not match any event."""


def _event_attr(event: Any, *names: str) -> Any:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return value
    raise AttributeError(f"profiler event {event!r} does not expose any of {names}")


def _event_name(event: Any) -> str:
    return str(_event_attr(event, "name", "kernel_name"))


def _matches(name: str, selector: str, mode: str) -> bool:
    if mode == "exact":
        return name == selector
    if mode == "contains":
        return selector in name
    return re.search(selector, name) is not None


def select_profile_events(events: Sequence[Any], profile_spec: ProfileSpec) -> list[Any]:
    """Select profiler events and require every configured selector to match."""

    events = list(events)
    if profile_spec.kernels is None:
        return events

    matched_indices: set[int] = set()
    missing = []
    event_names = [_event_name(event) for event in events]
    for selector in profile_spec.kernels:
        selector_matches = {
            index
            for index, event_name in enumerate(event_names)
            if _matches(event_name, selector, profile_spec.match)
        }
        if not selector_matches:
            missing.append(selector)
        matched_indices.update(selector_matches)

    if missing:
        available = sorted(set(event_names))
        raise KernelSelectionError(
            f"kernel selectors not found ({profile_spec.match}): {missing}; "
            f"available profiler events: {available}"
        )
    return [event for index, event in enumerate(events) if index in matched_indices]


def compute_profile_latency(events: Sequence[Any], profile_spec: ProfileSpec) -> float:
    """Compute selected profiler latency in microseconds."""

    selected = select_profile_events(events, profile_spec)
    if not selected:
        raise KernelSelectionError("profiler returned no kernel events")

    if profile_spec.reduction == "sum":
        durations = []
        for event in selected:
            event_duration = getattr(event, "duration", None)
            if event_duration is None:
                event_duration = _event_attr(event, "end_time", "end") - _event_attr(
                    event, "start_time", "start"
                )
            durations.append(float(event_duration))
        duration = sum(durations)
    else:
        duration = compute_latency_us(selected)
    return round(duration, 3)


class GeneratedMojoPerfAdapter(BasicOp):
    """Common BasicOp adapter generated for Mojo Operators and Functions."""

    perf_op_name = ""
    mojo_backend = ""
    xpu_provider = ""

    def prepare_args(self):
        self.perf_spec = get_perf_spec(self.perf_op_name)
        self.case_params = self.perf_spec.resolve_case_params(self.args_dict)
        self.workload: PerfWorkload = self.perf_spec.build(self.case_params)

    def vendor_parser(self):
        if self.xpu_provider != ProviderRegistry.BASE_PROVIDER:
            provider_spec = self.perf_spec.providers[self.xpu_provider]
            if not provider_spec.supports_case(self.case_params):
                reason = provider_spec.unsupported_reason or "capability predicate rejected the case"
                raise ValueError(
                    f"{self.perf_op_name}/{self.xpu_provider} does not support "
                    f"case {self.args_dict.get('__case_id__')!r}: {reason}"
                )
        self.backend_target_cls = self.perf_spec.target.get_backend_impl(
            self.mojo_backend, strict=True
        )

    def vendor_impl(self):
        device = self.backend.get_torch_device_name()
        self.input_tensor_info = {
            name: _tensor_info(spec, device) for name, spec in self.workload.inputs.items()
        }
        self.output_tensor_info = {
            name: _tensor_info(spec, device) for name, spec in self.workload.outputs.items()
        }

        self.input_tensor_size = sum(calc_tensor_size(info) for info in self.input_tensor_info.values())
        self.output_tensor_size = sum(calc_tensor_size(info) for info in self.output_tensor_info.values())
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = (
            self.input_tensor_size if self.workload.read_bytes is None else self.workload.read_bytes
        )
        self.write_bytes = (
            self.output_tensor_size if self.workload.write_bytes is None else self.workload.write_bytes
        )
        self.io_bytes = self.read_bytes + self.write_bytes
        self.calc_flops = self.workload.flops

        self._is_function_backward = False
        if issubclass(self.perf_spec.target, MojoOperator):
            if self.perf_spec.phase != "forward":
                raise ValueError("Mojo Operators only support phase='forward'")
            if self.workload.forward_args is not None:
                raise ValueError(
                    "PerfWorkload.forward_args are only valid for Mojo Function backward"
                )
            if self.workload.target_factory is None:
                self._target: Callable[..., Any] = self.backend_target_cls(
                    **dict(self.workload.op_kwargs)
                ).to(device)
            else:
                self._target = self.workload.target_factory(self.backend_target_cls, device)
                if not callable(self._target):
                    raise TypeError("PerfWorkload.target_factory must return a callable target")
                if isinstance(self._target, torch.nn.Module):
                    self._target = self._target.to(device)
            self._is_operator = True
        elif issubclass(self.perf_spec.target, MojoFunction):
            if self.workload.op_kwargs:
                raise ValueError("PerfWorkload.op_kwargs are only valid for Mojo Operators")
            if self.workload.state:
                raise ValueError("PerfWorkload.state is only valid for Mojo Operators")
            if self.workload.target_factory is not None:
                raise ValueError("PerfWorkload.target_factory is only valid for Mojo Operators")
            if self.workload.kwargs:
                raise ValueError(
                    "Mojo Functions accept positional arguments only; "
                    "place tensor references and literals in PerfWorkload.args"
                )
            if self.perf_spec.phase == "backward":
                if self.workload.forward_args is None:
                    raise ValueError(
                        "Mojo Function backward requires PerfWorkload.forward_args"
                    )
                self._target = None
                self._is_function_backward = True
            else:
                if self.workload.forward_args is not None:
                    raise ValueError(
                        "PerfWorkload.forward_args are only valid for Mojo Function backward"
                    )
                self._target = self.backend_target_cls.apply
            self._is_operator = False
        else:
            raise TypeError(
                f"{self.perf_spec.target.__name__} is neither MojoOperator nor MojoFunction"
            )

        self.profile_spec = self._resolve_profile_spec()
        self.require_profiling = self.profile_spec.timing == "profiler"
        if self.require_profiling and not getattr(self.backend, "enable_profiling", False):
            raise RuntimeError(
                "descriptor requires profiler timing, but xpu-perf backend profiling is disabled"
            )
        self._profile_error: Exception | None = None
        if self.require_profiling:
            self.profiler_parser = self._parse_profile_events
        self._device = device
        if self.workload.tensor_factory is None:
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=self.workload.create_outputs,
            )
        else:
            self._create_tensors_func = self._create_workload_tensors
        self._run_func = self.vendor_impl_run

    def _create_workload_tensors(self, instance_num: int):
        first = dict(self.workload.tensor_factory(self._device))
        required = set(self.workload.inputs)
        if self.workload.create_outputs:
            required.update(self.workload.outputs)
        missing = required - set(first)
        if missing:
            raise ValueError(f"tensor_factory omitted required tensors: {sorted(missing)}")
        invalid = sorted(name for name, value in first.items() if not isinstance(value, torch.Tensor))
        if invalid:
            raise TypeError(f"tensor_factory values must be tensors: {invalid}")

        for name, tensor_spec in {**self.workload.inputs, **self.workload.outputs}.items():
            if name not in first:
                continue
            actual = first[name]
            if tuple(actual.shape) != tensor_spec.shape:
                raise ValueError(
                    f"tensor_factory produced {name!r} shape {tuple(actual.shape)}, "
                    f"expected {tensor_spec.shape}"
                )
            if actual.dtype != tensor_spec.dtype:
                raise ValueError(
                    f"tensor_factory produced {name!r} dtype {actual.dtype}, "
                    f"expected {tensor_spec.dtype}"
                )

        mappings = [first]
        for _ in range(instance_num - 1):
            mappings.append({name: value.clone() for name, value in first.items()})
        return mappings

    def create_tensors(self, instance_num: int):
        mappings = self._create_tensors_func(instance_num)
        if self._is_operator and self.workload.state:
            self._bind_state(mappings[0])
        if self._is_function_backward:
            self._function_contexts = {}
            with torch.no_grad():
                for tensor_mapping in mappings:
                    ctx = _BenchmarkFunctionContext()
                    forward_args = [
                        _resolve_call_arg(value, tensor_mapping)
                        for value in self.workload.forward_args
                    ]
                    self.backend_target_cls.forward(ctx, *forward_args)
                    self._function_contexts[id(tensor_mapping)] = ctx
            self.backend.device_synchronize()
        return mappings

    def _bind_state(self, tensor_mapping: Mapping[str, torch.Tensor]) -> None:
        for target_attr, tensor_name in self.workload.state.items():
            tensor = tensor_mapping[tensor_name]
            # Use the direct registry intentionally: get_parameter() raises for a
            # parameter registered as None, while _parameters preserves that state
            # without exception-based control flow.
            if target_attr in self._target._parameters:
                previous = self._target._parameters[target_attr]
                requires_grad = previous.requires_grad if previous is not None else False
                setattr(
                    self._target,
                    target_attr,
                    torch.nn.Parameter(tensor, requires_grad=requires_grad),
                )
            else:
                setattr(self._target, target_attr, tensor)

    def vendor_impl_run(self, tensor_mapping: Mapping[str, torch.Tensor]):
        target = self._target
        if self._is_function_backward:
            try:
                ctx = self._function_contexts[id(tensor_mapping)]
            except KeyError as err:
                raise RuntimeError(
                    "Function backward tensors were not prepared by create_tensors"
                ) from err
            target = partial(self.backend_target_cls.backward, ctx)

        if self.workload.run is not None:
            return self.workload.run(target, tensor_mapping)

        args = [_resolve_call_arg(value, tensor_mapping) for value in self.workload.args]
        kwargs = {
            name: _resolve_call_arg(value, tensor_mapping)
            for name, value in self.workload.kwargs.items()
        }
        return target(*args, **kwargs)

    def _resolve_profile_spec(self) -> ProfileSpec:
        configured = self.perf_spec.profiles[self.xpu_provider]
        override = self.args_dict.get("__perf_timing__")
        if override is None or override == configured.timing:
            return configured
        if override == "event":
            return ProfileSpec(timing="event")
        if override == "profiler":
            return ProfileSpec(timing="profiler")
        raise ValueError(f"unknown timing override {override!r}")

    def _parse_profile_events(self, events: Sequence[Any]) -> float:
        try:
            return compute_profile_latency(events, self.profile_spec)
        except Exception as err:  # xpu-perf catches parser errors inside Backend.perf.
            self._profile_error = err
            return 0.0

    def summary(self, latency_us, kernel_mapping=None):
        if self._profile_error is not None:
            raise RuntimeError(
                f"failed to profile {self.perf_op_name}/{self.xpu_provider}: "
                f"{self._profile_error}"
            ) from self._profile_error
        target = super().summary(latency_us, kernel_mapping or {})
        if target:
            target["timing_mode"] = self.profile_spec.timing
            if self.profile_spec.timing == "profiler":
                target["latency_reduction"] = self.profile_spec.reduction
                if self.profile_spec.kernels is not None:
                    target["profile_kernels"] = list(self.profile_spec.kernels)
                    target["profile_match"] = self.profile_spec.match
        return target


def _class_name(prefix: str, op_name: str, provider: str | None = None) -> str:
    parts = [prefix, op_name]
    if provider is not None:
        parts.append(provider)
    return "_".join(re.sub(r"[^0-9A-Za-z_]", "_", part) for part in parts)


def register_base_specs() -> None:
    """Register every descriptor during xpu-perf's op_defs loading phase."""

    for spec in iter_perf_specs():
        name = _class_name("XpuPerfBase", spec.name)
        generated = type(
            name,
            (GeneratedMojoPerfAdapter,),
            {
                "__module__": __name__,
                "__qualname__": name,
                "perf_op_name": spec.name,
                "mojo_backend": spec.base_backend,
                "xpu_provider": ProviderRegistry.BASE_PROVIDER,
            },
        )
        registered = ProviderRegistry.register_base_impl(spec.name, spec.engine)(generated)
        globals()[name] = registered


def register_vendor_specs(provider: str) -> int:
    """Register descriptors supported by *provider* on the current platform."""

    registered_count = 0
    for spec in iter_perf_specs():
        if provider not in spec.providers:
            continue
        backend = spec.providers[provider].backend
        try:
            spec.target.get_backend_impl(backend, strict=True)
        except KeyError:
            continue

        name = _class_name("XpuPerfVendor", spec.name, provider)
        generated = type(
            name,
            (GeneratedMojoPerfAdapter,),
            {
                "__module__": __name__,
                "__qualname__": name,
                "perf_op_name": spec.name,
                "mojo_backend": backend,
                "xpu_provider": provider,
            },
        )
        registered = ProviderRegistry.register_vendor_impl(spec.name, provider)(generated)
        globals()[name] = registered
        registered_count += 1
    return registered_count


def register_available_vendor_specs() -> tuple[str, ...]:
    """Register descriptor providers whose Mojo backends exist in this process."""

    provider_names = list(dict.fromkeys(
        provider
        for spec in iter_perf_specs()
        for provider in spec.providers
    ))
    registered_providers = []
    for provider in provider_names:
        if register_vendor_specs(provider):
            registered_providers.append(provider)
    return tuple(registered_providers)
