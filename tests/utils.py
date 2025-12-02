import csv
import functools
import os
import subprocess
import sys
import time

from typing import Callable
from typing import Literal

import pytest
import torch
import torch_npu

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


@functools.lru_cache
def get_platform() -> Literal["npu", "mlu", "cpu"]:
    """
    Detect whether the system has NPU or MLU.
    """
    try:
        subprocess.run(["npu-smi", "info"], check=True)
        logger.info("Ascend NPU detected")
        return "npu"
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            subprocess.run(["cnmon"], check=True)
            logger.info("Cambricon MLU detected")
            return "mlu"
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.info("No accelerator detected")
            return "cpu"


def auto_switch_platform(set_perf: bool = False):
    device = get_platform()

    if set_perf:
        if device == "npu":
            perf_fn = perf_npu
        else:
            raise NotImplementedError(f"Performance test is not implemented on {device}")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = [arg.to(device=device) if isinstance(arg, torch.Tensor) else arg for arg in args]
            new_kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            if set_perf:
                module = sys.modules[func.__module__]
                setattr(module, "perf", perf_fn)

            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator


# Skip current test if this case is not implemented on current chosen backend.
def bypass_not_implemented(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            pytest.skip("Not implemented on this backend, skipped.")
            return None

    return wrapper


def is_similar(result, golden, atol=1e-2, rtol=1e-2):
    return result.shape == golden.shape and torch.allclose(result, golden, atol, rtol)


def get_max_diff(result, golden):
    return torch.max(torch.abs(result - golden))


def device_perf_npu(executor, profiling_dir="./npu_profiling", active=5):
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )
    # When using the Triton backend, JIT compilation causes the first execution to include
    # extra compile-time ops; without a warm-up run, the op counts in the profiling results
    # will be inconsistent.
    executor()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=active, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        mat_a = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_b = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_c = torch.matmul(mat_a, mat_b)
        mat_c.cpu()

        for _ in range(10):
            executor()
            prof.step()

        torch.npu.synchronize()

    try:
        all_subdirs = [
            os.path.join(profiling_dir, d)
            for d in os.listdir(profiling_dir)
            if os.path.isdir(os.path.join(profiling_dir, d))
        ]

        if all_subdirs:
            return max(all_subdirs, key=os.path.getmtime)
        else:
            logger.warning("Profiling dir unfound.")
            return None

    except Exception as e:
        logger.warning(f"Failed to get Profiling folder name: {e}")
        return None


def host_perf(func, device, warmup=3, repeat=10):
    if device.lower() == "npu":
        sync = torch.npu.synchronize
    else:
        sync = lambda: None

    for _ in range(warmup):
        func()
        sync()

    start = time.time()
    for _ in range(repeat):
        func()
        sync()
    end = time.time()

    avg_time = (end - start) / repeat * 1000
    return avg_time


def perf_npu(executor, profiling_dir="./npu_profiling", active=5):
    kernel_profiling_path = device_perf_npu(executor, profiling_dir, active)
    csv_file_path = os.path.join(kernel_profiling_path, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")

    if not os.path.exists(csv_file_path):
        logger.warning(f"File not found: {csv_file_path}")
        return None

    total_avg_time_us = 0.0

    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            avg_time = float(row["Total Time(us)"])
            total_avg_time_us += avg_time

    device_latency = total_avg_time_us / active
    host_latency = host_perf(executor, "npu")

    logger.info(
        f"Device latency = {device_latency:.4f} us | "
        f"Host latency = {host_latency:.4f} ms | "
        f"Profile dir = {kernel_profiling_path}"
    )
