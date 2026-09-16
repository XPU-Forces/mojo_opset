"""Allocator-tracked device memory, measured separately from timing/profiling."""


def measure_memory(workload, adapter, device):
    runtime = adapter.runtime
    names = ("memory_allocated", "max_memory_allocated", "reset_peak_memory_stats")
    missing = [name for name in names if not callable(getattr(runtime, name, None))]
    if missing:
        raise RuntimeError(f"Device memory reporting requires runtime APIs: {', '.join(missing)}")
    functions = list(workload) if isinstance(workload, (list, tuple)) else [workload]
    if not functions or not all(callable(fn) for fn in functions):
        raise ValueError("Provide at least one memory workload")
    runtime.synchronize()
    allocated = int(runtime.memory_allocated(device))
    runtime.reset_peak_memory_stats(device)
    # Timing already warmed every instance. Keep outputs alive through completion.
    # No profiler, eviction buffer, empty_cache or timing events in this pass.
    for fn in functions:
        result = fn()
        runtime.synchronize()
        del result
    peak = int(runtime.max_memory_allocated(device))
    return {
        "baseline_allocated_bytes": allocated,
        "peak_allocated_bytes": peak,
        "incremental_peak_bytes": max(0, peak - allocated),
    }
