"""Per-call profiler span/sum, batched stream events and synchronized wall latency."""

import json
import math
import re
import statistics
import tempfile
import time

from decimal import Decimal
from pathlib import Path

import torch

SAMPLE_PREFIX = "mojo_perf_sample_"


def summarize(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Timing samples must be finite and positive")
    ordered = sorted(samples)
    median = statistics.median(ordered)
    return {
        "median": median,
        "p10": ordered[int((len(ordered) - 1) * 0.1)],
        "p90": ordered[int((len(ordered) - 1) * 0.9)],
        "mad": statistics.median(abs(value - median) for value in ordered),
        "samples": samples,
    }


def kernel_spans(
    trace, is_kernel, repeats, *, reduction="span", selectors=(), match="contains", details=None, kernel_owners=None
):
    """Group kernels by platform correlation, or synchronized ranges on a shared clock."""
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    ranges = {}
    kernels = []
    origin = next((Decimal(str(e["ts"])) for e in events if e.get("ph") == "X"), Decimal(0))
    for event_index, event in enumerate(events):
        if event.get("ph") != "X":
            continue
        # NPU timestamps may be absolute microseconds encoded as decimal strings.
        start = float(Decimal(str(event["ts"])) - origin)
        duration = float(event.get("dur", 0))
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid trace timestamp/duration")
        end = start + duration
        name = event.get("name", "")
        if name.startswith(SAMPLE_PREFIX):
            if name in ranges:
                raise ValueError(f"Duplicate profiler range: {name}")
            ranges[name] = (start, end, event)
        elif is_kernel(event):
            kernels.append((start, end, name, event_index))
    if len(ranges) != repeats:
        raise RuntimeError(f"Expected {repeats} measurement ranges; profiler returned {len(ranges)}")
    owners = kernel_owners(events, {name: value[2] for name, value in ranges.items()}) if kernel_owners else None
    samples = []
    for index in range(repeats):
        sample_name = f"{SAMPLE_PREFIX}{index}"
        lower, upper, _ = ranges[sample_name]
        selected = [
            (start, end, name)
            for start, end, name, event_index in kernels
            if (owners.get(event_index) == sample_name if owners is not None else lower <= start and end <= upper)
        ]
        if not selected:
            raise RuntimeError(f"No device kernels in sample {index}; check profiler support and the platform adapter")
        if selectors:
            matches = lambda name, selector: (
                name == selector
                if match == "exact"
                else selector in name
                if match == "contains"
                else re.search(selector, name)
            )
            for selector in selectors:
                if not any(matches(name, selector) for _, _, name in selected):
                    raise ValueError(
                        f"Kernel selector {selector!r} missing in sample {index}; "
                        f"available: {sorted({name for _, _, name in selected})}"
                    )
            selected = [event for event in selected if any(matches(event[2], selector) for selector in selectors)]
        origin = min(start for start, _, _ in selected)
        if details is not None:
            details.append(
                [
                    {"name": name, "start_us": start - origin, "duration_us": end - start}
                    for start, end, name in selected
                ]
            )
        samples.append(
            sum(end - start for start, end, _ in selected)
            if reduction == "sum"
            else max(end for _, end, _ in selected) - origin
        )
    return samples


def measure(
    fn,
    adapter,
    *,
    warmup=10,
    repeats=20,
    timers=("profiler", "e2e"),
    batch=1,
    reduction="span",
    selectors=(),
    match="contains",
):
    if warmup < 1 or repeats < 3:
        raise ValueError("warmup must be >= 1 and repeats >= 3")
    if not timers or len(set(timers)) != len(timers) or set(timers) - {"profiler", "event", "e2e"}:
        raise ValueError("Choose distinct profiler, event and/or e2e timers")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if reduction not in ("span", "sum") or match not in ("exact", "contains", "regex"):
        raise ValueError("Invalid profiler reduction or selector match mode")
    if "profiler" not in timers and (selectors or reduction != "span"):
        raise ValueError("Kernel selection/reduction requires profiler timing")
    if match == "regex":
        for selector in selectors:
            re.compile(selector)
    functions = list(fn) if isinstance(fn, (list, tuple)) else []
    if len(functions) < 2 or not all(callable(function) for function in functions):
        raise ValueError("rotate requires at least two independent callables")
    cursor = 0

    def invoke():
        nonlocal cursor
        function = functions[cursor % len(functions)]
        cursor += 1
        return function()

    synchronize = adapter.runtime.synchronize

    for function in functions:
        function()  # Initialize every rotated instance, including saved backward state.
    for _ in range(warmup):
        invoke()
    synchronize()  # Includes lazy imports, JIT and autotuning, all outside measurement.

    metrics = {}
    if "e2e" in timers:
        e2e = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter_ns()
            result = invoke()
            synchronize()
            e2e.append((time.perf_counter_ns() - start) / 1000)
            del result
        metrics["e2e_us"] = summarize(e2e)

    if "event" in timers:
        event_samples = []
        start_event = adapter.runtime.Event(enable_timing=True)
        end_event = adapter.runtime.Event(enable_timing=True)
        # Initialize lazy event resources outside the samples.
        start_event.record()
        end_event.record()
        synchronize()
        for _ in range(repeats):
            synchronize()
            start_event.record()
            for _ in range(batch):
                result = invoke()
                del result
            end_event.record()
            synchronize()
            event_samples.append(start_event.elapsed_time(end_event) * 1000 / batch)
        metrics["event_us"] = summarize(event_samples)

    def profile_samples():
        # Warm every rotated instance after profiler initialization, outside samples.
        for _ in range(max(warmup, len(functions))):
            invoke()
        synchronize()
        for index in range(repeats):
            synchronize()
            with torch.autograd.profiler.record_function(f"{SAMPLE_PREFIX}{index}"):
                result = invoke()
                synchronize()
            del result

    if "profiler" in timers:
        details = []
        with tempfile.TemporaryDirectory(prefix="mojo-perf-") as directory:
            trace_path = adapter.collect_trace(profile_samples, Path(directory))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            samples = kernel_spans(
                trace,
                adapter.is_kernel,
                repeats,
                reduction=reduction,
                selectors=selectors,
                match=match,
                details=details,
                kernel_owners=getattr(adapter, "kernel_owners", None),
            )
        metrics[f"profiler_{reduction}_us"] = {**summarize(samples), "kernels": details}
    return metrics
