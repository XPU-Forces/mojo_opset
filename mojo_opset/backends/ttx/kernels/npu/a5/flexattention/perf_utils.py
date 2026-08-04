import gc
import os
import time
from pathlib import Path

import torch


def sync_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_profile_grads(q, k, v):
    q.grad = None
    k.grad = None
    v.grad = None


def current_memory_mb():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.memory_allocated() / (1024 ** 2)
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def reset_peak_memory_stats():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats()
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def max_memory_allocated_mb():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.max_memory_allocated() / (1024 ** 2)
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def empty_cache():
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def profile_trace_dir(prof_root, mask_name, op_name, phase):
    now = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    usec = int((now - int(now)) * 1000000)
    trace_dir = Path(prof_root) / mask_name / op_name / phase / f"run_{stamp}_{usec:06d}_pid_{os.getpid()}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def clear_l2_cache():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return
    for _ in range(2):
        a = torch.randn(8 * 1024 * 1024, dtype=torch.float32).npu()
        b = torch.randn(8 * 1024 * 1024, dtype=torch.float32).npu()
        _ = a + b
    sync_device()


def profile_npu_op(op_name, mask_name, phase, step_fn, prof_root, warmup=1, active=3):
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("torch_npu profiler is required for NPU profiling") from exc

    trace_dir = profile_trace_dir(prof_root, mask_name, op_name, phase)
    print(f"[profiler:{op_name}:{phase}] begin: {trace_dir}")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    for _ in range(warmup):
        step_fn()
    clear_l2_cache()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=active, repeat=1, skip_first=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(active + 3):
            step_fn()
            clear_l2_cache()
            sync_device()
            prof.step()
    print(f"[profiler:{op_name}:{phase}] end: {trace_dir}")
    return str(trace_dir)


def profile_phase_guard(op_name, phase, fn):
    try:
        return {"trace_dir": fn()}
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"[profiler:{op_name}:{phase}] failed: {reason}")
        empty_cache()
        return {"failed": reason}


def timed_call(fn):
    reset_peak_memory_stats()
    sync_device()
    start = time.perf_counter()
    result = fn()
    sync_device()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms, max_memory_allocated_mb()


def profile_autograd_op(
    op_name,
    mask_name,
    forward_fn,
    q,
    k,
    v,
    return_tensor,
    prof_root,
    enable_profiler=False,
):
    clear_profile_grads(q, k, v)

    def _forward_step():
        out = forward_fn(q, k, v)
        sync_device()
        del out

    if enable_profiler:
        forward_result = profile_phase_guard(
            op_name,
            "forward",
            lambda: profile_npu_op(op_name, mask_name, "forward", _forward_step, prof_root),
        )
    else:
        _, fwd_ms, fwd_mem = timed_call(_forward_step)
        forward_result = {"time_ms": fwd_ms, "peak_mem_mb": fwd_mem}

    clear_profile_grads(q, k, v)
    pre_out = forward_fn(q, k, v)
    sync_device()

    def _backward_step():
        if tuple(pre_out.shape) != tuple(return_tensor.shape):
            raise RuntimeError(
                f"return_tensor shape {tuple(return_tensor.shape)} does not match output shape {tuple(pre_out.shape)}"
            )
        pre_out.backward(return_tensor, retain_graph=True)
        sync_device()
        clear_profile_grads(q, k, v)

    if enable_profiler:
        backward_result = profile_phase_guard(
            op_name,
            "backward",
            lambda: profile_npu_op(op_name, mask_name, "backward", _backward_step, prof_root),
        )
    else:
        _, bwd_ms, bwd_mem = timed_call(_backward_step)
        backward_result = {"time_ms": bwd_ms, "peak_mem_mb": bwd_mem}

    del pre_out
    clear_profile_grads(q, k, v)
    empty_cache()
    return {"forward": forward_result, "backward": backward_result}
