import functools
import subprocess

from typing import Callable
from typing import Literal

import pytest
import torch


@functools.lru_cache
def get_platform() -> Literal["npu", "mlu", "cpu"]:
    """
    Detect whether the system has NPU or MLU.
    """
    try:
        subprocess.run(["npu-smi", "info"], check=True)
        print("Ascend NPU detected")
        return "npu"
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            subprocess.run(["cnmon"], check=True)
            print("Cambricon MLU detected")
            return "mlu"
        except (subprocess.SubprocessError, FileNotFoundError):
            print("No accelerator detected")
            return "cpu"


def auto_switch_platform():
    device = get_platform()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(arg.to(device=device))
                else:
                    new_args.append(arg)

            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    new_kwargs[key] = value.to(device)
                else:
                    new_kwargs[key] = value

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
