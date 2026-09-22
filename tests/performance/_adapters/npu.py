from decimal import Decimal
from pathlib import Path

import torch
import torch_npu

runtime = torch.npu


def collect_trace(run, directory):
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        # Level0 can omit the launch correlation needed by mixed cube/vector kernels.
        experimental_config=profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1),
    ) as profile:
        run()
        profile.step()
    paths = list(Path(directory).rglob("trace_view.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one NPU trace, found {len(paths)}")
    return paths[0]


def is_kernel(event):
    task_type = str(event.get("args", {}).get("Task Type", ""))
    # CANN versions use either KERNEL_* or core names for compute tasks.
    return task_type.startswith("KERNEL") or task_type in {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"}


def kernel_owners(events, ranges):
    """Follow torch_to_npu launch flows; kernels without a flow (GE custom
    operators launched outside the torch_npu dispatch layer, e.g. bisheng JIT
    kernels from cannbotdsl) fall back to device-timestamp attribution against
    the synchronized sample ranges. Host and device timestamps need not align
    exactly for the flow path; the fallback relies on every sample ending with
    a synchronize, so a device kernel start always falls inside its owning
    sample's host range."""

    def point(event):
        if any(key not in event for key in ("pid", "tid", "ts")):
            raise ValueError("NPU flow endpoint is missing pid/tid/ts")
        timestamp = Decimal(str(event["ts"]))
        if not timestamp.is_finite():
            raise ValueError("Invalid NPU flow timestamp")
        return event["pid"], event["tid"], timestamp

    bounds = []
    for name, event in ranges.items():
        pid, tid, start = point(event)
        end = start + Decimal(str(event["dur"]))
        if any(pid == p and tid == t and start < upper and lower < end for _, p, t, lower, upper in bounds):
            raise ValueError("Ambiguous overlapping NPU sample ranges on the same thread")
        bounds.append((name, pid, tid, start, end))

    # Autograd submits backward kernels on worker threads, not the caller's
    # record_function thread. Associate only engine scopes enclosed by a unique
    # sample on the same host process; unrelated threads remain excluded.
    worker_bounds = []
    for event in events:
        if event.get("ph") != "X" or not event.get("name", "").startswith("autograd::engine::evaluate_function:"):
            continue
        pid, tid, start = point(event)
        end = start + Decimal(str(event["dur"]))
        matches = {name for name, p, _, lower, upper in bounds if pid == p and lower <= start and end <= upper}
        if len(matches) > 1:
            raise ValueError("Autograd scope belongs to multiple NPU sample ranges")
        if matches:
            worker_bounds.append((matches.pop(), pid, tid, start, end))

    targets, kernels, flows = {}, set(), {}
    for index, event in enumerate(events):
        if event.get("ph") == "X" and "Task Type" in event.get("args", {}):
            endpoint = point(event)
            if endpoint in targets:
                raise ValueError("Ambiguous duplicate NPU device endpoint")
            targets[endpoint] = index
            if is_kernel(event):
                kernels.add(index)
        if event.get("cat") != "async_npu" or event.get("name") != "torch_to_npu":
            continue
        phase = event.get("ph")
        if "id" not in event or phase not in ("s", "f"):
            raise ValueError("Invalid NPU torch_to_npu flow")
        key = str(event["id"])
        pair = flows.setdefault(key, {})
        if phase in pair:
            raise ValueError(f"Duplicate NPU flow {key!r} {phase!r} endpoint")
        pair[phase] = event

    owners = {}
    for key, pair in flows.items():
        if set(pair) != {"s", "f"}:
            raise ValueError(f"Incomplete NPU flow {key!r}")
        index = targets.get(point(pair["f"]))
        if index is None:
            raise ValueError(f"NPU flow {key!r} has no matching device event")
        if index not in kernels:
            continue  # Memory copies have flows too, but are not kernel timings.
        if index in owners:
            raise ValueError("NPU kernel has multiple flow owners")
        pid, tid, timestamp = point(pair["s"])
        matches = {
            name
            for name, p, t, lower, upper in bounds + worker_bounds
            if pid == p and tid == t and lower <= timestamp < upper
        }
        if len(matches) > 1:
            raise ValueError("NPU kernel belongs to multiple sample ranges")
        owners[index] = matches.pop() if matches else None  # Warmup stays outside samples.
    unowned = kernels - owners.keys()
    if unowned:
        # GE custom operators (e.g. bisheng JIT kernels from cannbotdsl) launch
        # outside the torch_npu dispatch layer, so their device kernels carry no
        # torch_to_npu flow. Attribute them by timestamp against the sample
        # ranges instead: every sample ends with a synchronize, so a kernel's
        # device start always falls inside its owning sample's host range.
        # Pure-aclnn workloads leave `unowned` empty and keep the strict
        # original behavior unchanged.
        for index in sorted(unowned):
            ts = Decimal(str(events[index]["ts"]))
            matches = [name for name, _, _, lower, upper in bounds if lower <= ts < upper]
            owners[index] = matches[0] if len(matches) == 1 else None
    return owners


def l2_bytes(properties):
    return getattr(properties, "L2_cache_size", None)
