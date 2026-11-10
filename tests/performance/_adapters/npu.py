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
        # Timing needs launch correlation, not PMU counters. CANN 9.2 PMU
        # collection can corrupt block-detail export under multi-device profiling.
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
        ),
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
    """Follow Torch or direct CANN launch flows; device clocks need not align."""

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

    # Native launches use torch_npu's task queue. Follow its explicit enqueue
    # correlation to the dequeue scope, rather than comparing device clocks.
    queue_flows, dequeues = {}, {}
    for event in events:
        if event.get("cat") == "async_task_queue":
            phase = event.get("ph")
            if phase not in ("s", "f") or "id" not in event:
                raise ValueError("Invalid NPU task queue flow")
            pair = queue_flows.setdefault((event["pid"], str(event["id"])), {})
            if phase in pair:
                raise ValueError("Duplicate NPU task queue endpoint")
            pair[phase] = event
        if event.get("cat") == "dequeue" and event.get("ph") == "X":
            key = point(event)
            if key in dequeues:
                raise ValueError("Ambiguous NPU dequeue scope")
            dequeues[key] = event
    for (_, correlation), pair in queue_flows.items():
        if set(pair) != {"s", "f"}:
            raise ValueError("Incomplete NPU task queue flow")
        scope = dequeues.get(point(pair["f"]))
        if scope is None or str(scope.get("args", {}).get("correlation_id")) != correlation:
            raise ValueError("NPU task queue has no matching dequeue scope")
        pid, tid, start = point(pair["s"])
        matches = {name for name, p, t, lower, upper in bounds + worker_bounds
                   if pid == p and tid == t and lower <= start < upper}
        if len(matches) > 1:
            raise ValueError("NPU enqueue belongs to multiple samples")
        if matches:
            pid, tid, start = point(scope)
            duration = Decimal(str(scope["dur"]))
            if not duration.is_finite() or duration < 0:
                raise ValueError("Invalid NPU dequeue duration")
            worker_bounds.append((matches.pop(), pid, tid, start, start + duration))

    targets, kernels, flows = {}, set(), {"torch": {}, "cann": {}}
    processes = {}
    for index, event in enumerate(events):
        if event.get("ph") == "M" and event.get("name") == "process_name":
            processes.setdefault(event.get("args", {}).get("name"), set()).add(event.get("pid"))
        if event.get("ph") == "X" and "Task Type" in event.get("args", {}):
            endpoint = point(event)
            if endpoint in targets:
                raise ValueError("Ambiguous duplicate NPU device endpoint")
            targets[endpoint] = index
            if is_kernel(event):
                kernels.add(index)
        if event.get("cat") == "async_npu" and event.get("name") == "torch_to_npu":
            family = "torch"
        elif event.get("cat") == "HostToDevice":
            family = "cann"
        else:
            continue
        phase = event.get("ph")
        if "id" not in event or phase not in ("s", "f"):
            raise ValueError("Invalid NPU launch flow")
        key = str(event["id"])
        pair = flows[family].setdefault(key, {})
        if phase in pair:
            raise ValueError(f"Duplicate NPU flow {key!r} {phase!r} endpoint")
        pair[phase] = event

    # CANN 9.2 exports some native launch flows with only the device end.
    # The AscendCL API still carries connection_id; the exported flow ID is
    # (connection_id << 32) | device_id. Restrict recovery to this exact schema
    # and a unique CANN API, and keep rejecting unresolvable/ambiguous flows.
    native_apis = {}
    for event in events:
        connection = event.get("args", {}).get("connection_id")
        if (event.get("ph") == "X" and event.get("pid") in processes.get("CANN", set())
                and event.get("name", "").startswith("AscendCL@")
                and type(connection) is int and connection > 0):
            native_apis.setdefault(connection, []).append(event)
    for key, pair in flows["cann"].items():
        if set(pair) != {"f"}:
            continue
        if not key.isdecimal():
            continue
        candidates = native_apis.get(int(key) >> 32, [])
        if len(candidates) > 1:
            raise ValueError("Ambiguous CANN API connection_id")
        if candidates:
            pair["s"] = candidates[0]

    owners = {}
    for family, connections in flows.items():
        claimed = set()
        for key, pair in connections.items():
            if set(pair) != {"s", "f"}:
                raise ValueError(f"Incomplete NPU flow {key!r}")
            index = targets.get(point(pair["f"]))
            if index is None:
                raise ValueError(f"NPU flow {key!r} has no matching device event")
            if index not in kernels:
                continue  # Memory copies have flows too, but are not kernel timings.
            if index in claimed:
                raise ValueError("NPU kernel has multiple flow owners")
            claimed.add(index)
            if family == "cann" and index in owners:
                continue  # Torch's flow also follows asynchronous task-queue submissions.
            pid, tid, timestamp = point(pair["s"])
            if family == "cann":
                # Direct AscendC launches may have no torch_to_npu flow. CANN
                # records their host thread/clock under a synthetic process lane.
                # Require an unambiguous single-process trace before translating
                # that lane; never assign kernels by device-time containment.
                python_pids = processes.get("Python", set())
                if (
                    pid not in processes.get("CANN", set())
                    or len(python_pids) != 1
                    or python_pids != {p for _, p, _, _, _ in bounds}
                ):
                    raise ValueError("Ambiguous CANN host process for native NPU launch")
                pid = next(iter(python_pids))
            matches = {
                name
                for name, p, t, lower, upper in bounds + worker_bounds
                if pid == p and tid == t and lower <= timestamp < upper
            }
            if len(matches) > 1:
                raise ValueError("NPU kernel belongs to multiple sample ranges")
            owners[index] = matches.pop() if matches else None  # Warmup stays outside samples.
    if kernels - owners.keys():
        raise ValueError("NPU device kernel is missing a launch flow")
    return owners


def l2_bytes(properties):
    return getattr(properties, "L2_cache_size", None)
