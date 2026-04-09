import csv
import datetime
import functools
import inspect
import os
import re
import sys
import time

from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union

import pytest
import torch

try:
    import torch_npu
except Exception:
    torch_npu = None

from mojo_opset.utils.logging import get_logger
from mojo_opset.utils.platform import get_platform

logger = get_logger(__name__)


def assert_close(
    results: Union[torch.Tensor, Tuple[Any, ...]],
    refs: Union[torch.Tensor, Tuple[Any, ...]],
):
    """
    Asserts that the results are close to the reference tensors within specified tolerances.

    Args:
        results (Union[torch.Tensor, Tuple[Any, ...]]): The calculated result tensor(s).
        refs (Union[torch.Tensor, Tuple[Any, ...]]): The reference/golden tensor(s).

    Raises:
        AssertionError: If shapes, dtypes, or values do not match within tolerance.
    """
    assert type(results) is type(refs)
    if isinstance(results, torch.Tensor) and isinstance(refs, torch.Tensor):
        results = tuple([results])
        refs = tuple([refs])

    for result, ref in zip(results, refs):
        if isinstance(result, torch.Tensor) and isinstance(ref, torch.Tensor):
            assert result.shape == ref.shape
            assert result.dtype == ref.dtype
            dtype = result.dtype
            if dtype == torch.bfloat16:
                max_atol = 0.1
                max_rtol = 0.05
                mean_atol = 0.01
                mean_rtol = 0.01
            elif dtype == torch.float16:
                max_atol = 2e-2
                max_rtol = 2e-2
                mean_atol = 2e-2
                mean_rtol = 2e-2
            elif dtype == torch.float32:
                max_atol = 6e-3
                max_rtol = 6e-3
                mean_atol = 1e-4
                mean_rtol = 1e-4
            else:
                logger.warning(f"dtype {dtype} is not supported.")
                assert False

            torch.testing.assert_close(result.to(torch.float32), ref.to(torch.float32), atol=max_atol, rtol=max_rtol)
            assert (
                torch.mean(torch.abs(ref - result)) < max_atol
                or torch.mean(torch.abs((ref - result) / (ref + mean_atol))) < mean_rtol
            )
        else:
            assert result == ref


def assert_deterministic(
    func: Callable[[], Any],
    seed: int = 42,
    num_runs: int = 2,
    err_msg_prefix: str = "",
):
    """
    Asserts that a given function produces deterministic results over multiple runs.

    This function is useful for testing numerical operations in frameworks like
    PyTorch to ensure they are not affected by sources of non-determinism
    (e.g., certain CUDA algorithms).

    Args:
        func (Callable[[], Any]):
            A no-argument function (lambda or regular function) that, when called,
            executes the computation to be tested and returns its results.
            The results can be a single tensor or a tuple of tensors.
        seed (int, optional):
            The random seed to reset PyTorch to before each run. Defaults to 42.
        num_runs (int, optional):
            The number of times to run the function. Must be at least 2. Defaults to 2.
        err_msg_prefix (str, optional):
            A prefix for the assertion error messages to provide more context. Defaults to "".

    Raises:
        AssertionError: If the results of any two runs are not identical.
    """
    assert num_runs >= 2, "Need at least 2 runs to check for determinism."

    torch.manual_seed(seed)
    result1 = func()

    for i in range(1, num_runs):
        torch.manual_seed(seed)
        result_next = func()

        _compare_deterministic_results(result1, result_next, err_msg_prefix, run_index=i + 1)


def _compare_deterministic_results(res1: Any, res2: Any, prefix: str, run_index: int):
    """Helper function to recursively compare two results (tensor or tuple of tensors)."""

    prefix = f"{prefix}: " if prefix else ""

    if isinstance(res1, tuple) and isinstance(res2, tuple):
        assert len(res1) == len(res2), (
            f"{prefix}Result tuples have different lengths between runs (run 1 vs run {run_index})."
        )
        for i, (item1, item2) in enumerate(zip(res1, res2)):
            _compare_deterministic_results(
                item1,
                item2,
                prefix=f"{prefix}Tuple element {i}",
                run_index=run_index,
            )

    elif torch.is_tensor(res1) and torch.is_tensor(res2):
        assert torch.equal(res1, res2), (
            f"{prefix}Tensor result is not deterministic between runs (run 1 vs run {run_index}).\n"
            f"Max difference: {(res1 - res2).abs().max().item()}"
        )

    elif res1 is None and res2 is None:
        pass

    else:
        raise AssertionError(
            f"{prefix}Type mismatch between runs (run 1 vs run {run_index}). Got {type(res1)} and {type(res2)}."
        )


def get_executor_info(executor):
    """
    Inspects a callable executor to retrieve its name and enclosed input arguments.

    Args:
        executor (Callable): The function or closure to inspect.

    Returns:
        Tuple[str, List[str]] or None: A tuple containing the function name and formatted arguments, or None if inspection fails.
    """

    def format_arg(arg):
        if isinstance(arg, torch.Tensor):
            return f"Tensor(shape={tuple(arg.shape)}, dtype={arg.dtype}, device={arg.device})"
        elif isinstance(arg, (int, float, str, bool)):
            return str(arg)
        elif isinstance(arg, (list, tuple)):
            return "[" + ", ".join(format_arg(a) for a in arg) + "]"
        elif arg is None:
            return "None"
        else:
            return f"<{type(arg).__name__}>"

    try:
        sig = inspect.signature(executor)
    except (TypeError, ValueError):
        logger.error("<Unknown callable>")
        return None

    if len(sig.parameters) > 0:
        logger.error("<Inputs unknown: executor takes parameters>")
        return None

    closure = executor.__closure__
    freevars = executor.__code__.co_freevars

    if not closure:
        logger.error("<No inputs (no closure)>")
        return None

    result = []
    for name, cell in zip(freevars, closure):
        value = cell.cell_contents
        result.append(f"{name}: {format_arg(value)}")

    matches = [re.search(r"<(.*?)>", r).group(1) for r in result if re.search(r"<(.*?)>", r) is not None]

    # Currently extracting class names from __closure__ using angle brackets.
    # TODO: Evaluate a more robust approach.
    assert len(matches) == 1
    func_name = matches[0]
    result = [r for r in result if f"<{func_name}>" not in r]

    return func_name, result


def format_executor_info(info_list):
    """
    Formats the executor information list into a human-readable string.

    Args:
        info_list (List[str]): List containing the function name as the first element and arguments as subsequent elements.

    Returns:
        str: A formatted string representation.
    """
    func = info_list[0]
    args = info_list[1:]

    arg_lines = "<br>  " + "<br>  ".join(args) if args else ""

    return f"{func}{arg_lines}"


def auto_switch_platform(set_perf: bool = False):
    """
    Decorator to automatically move tensor arguments to the current platform device and optionally inject performance testing.

    Args:
        set_perf (bool): If True, injects a 'perf' function into the module for performance profiling (NPU only).

    Returns:
        Callable: The decorated function.
    """
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
    """
    Decorator to skip a test if it raises a NotImplementedError.

    Args:
        func (Callable): The test function to wrap.

    Returns:
        Callable: The wrapped function that handles NotImplementedError.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(str(e) or "Not implemented on this backend, skipped.")
            return None

    return wrapper


def device_perf_npu(executor, profiling_dir="./npu_profiling", active=5):
    """
    Profiles the NPU kernel execution time using torch_npu.profiler.

    Args:
        executor (Callable): The function to profile.
        profiling_dir (str): Directory to save profiling results.
        active (int): Number of active steps for profiling.

    Returns:
        Tuple[float, str] or None: Average kernel time in us and path to the profiling logic, or None on failure.
    """
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )

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
        kernel_profiling_path = max(
            [
                os.path.join(profiling_dir, d)
                for d in os.listdir(profiling_dir)
                if os.path.isdir(os.path.join(profiling_dir, d))
            ],
            key=os.path.getmtime,
        )
        csv_file_path = os.path.join(kernel_profiling_path, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")

        if not os.path.exists(csv_file_path):
            logger.error(f"File not found: {csv_file_path}")
            return None

    except Exception as e:
        logger.error(f"Failed to get Profiling folder name: {e}")
        return None

    total_avg_time_us = 0.0

    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            avg_time = float(row["Total Time(us)"])
            total_avg_time_us += avg_time

    return total_avg_time_us / active, kernel_profiling_path


def device_perf_mlu(executor, profiling_dir="./mlu_profiling", active=5):
    """
    Profiles the MLU kernel execution time (Not Implemented).

    Args:
        executor (Callable): The function to profile.
        profiling_dir (str): Directory to save profiling results.
        active (int): Number of active steps.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError


def device_perf_ilu(executor, profiling_dir="./mlu_profiling", active=5):
    """
    Profiles the ILU kernel execution time (Not Implemented).

    Args:
        executor (Callable): The function to profile.
        profiling_dir (str): Directory to save profiling results.
        active (int): Number of active steps.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError


def host_perf(func, device, warmup=3, repeat=10):
    """
    Measures the host-side latency of a function execution.

    Args:
        func (Callable): The function to measure.
        device (str): Device type ('npu', etc.) for synchronization.
        warmup (int): Number of warmup iterations.
        repeat (int): Number of measurement iterations.

    Returns:
        float: Average execution time in milliseconds.
    """
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
    """
    Performs comprehensive NPU performance testing including device and host latency.

    Args:
        executor (Callable): The function to benchmark.
        profiling_dir (str): Directory for NPU profiling data.
        active (int): Number of active profiling steps.

    Returns:
        None: Logs and writes benchmark results to a file.
    """
    device_result = device_perf_npu(executor, profiling_dir, active)
    host_latency = host_perf(executor, "npu")

    if device_result is None:
        device_latency = None
        kernel_profiling_path = "unavailable"
        logger.warning("NPU profiler output is unavailable, falling back to host-latency-only benchmark logging.")
    else:
        device_latency, kernel_profiling_path = device_result

    func_name, para_list = get_executor_info(executor)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_latency_str = f"{device_latency:.4f} us" if device_latency is not None else "N/A"

    logger.info(
        f"[{func_name}] | "
        f"{', '.join(para_list)} | "
        f"Device latency = {device_latency_str} | "
        f"Host latency = {host_latency:.4f} ms | "
        f"Profile dir = {kernel_profiling_path}"
    )

    plain_log_file = (
        f"| {timestamp} | {func_name} | {format_executor_info(para_list)} | "
        f"{device_latency_str} | {host_latency:.4f} ms |"
    )
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf/benchmark.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        if not (os.path.exists(log_path) and os.path.getsize(log_path) > 1):
            f.write("| Timestamp | Op Name | Parameters | Device Latency (us) | Host Latency (ms) |\n")
            f.write("|-----------|---------|------------|---------------------|-------------------|\n")
        f.write(plain_log_file + "\n")


class MockFunctionCtx:
    """
    A mock context object to simulate torch.autograd.Function context methods.
    """

    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *tensors):
        """
        Simulates saving tensors for backward pass.

        Args:
            *tensors: Variable number of tensors to save.
        """
        self.saved_tensors = tensors
