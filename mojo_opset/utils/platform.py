import functools
import importlib
import inspect
import os
import pkgutil

from typing import Literal
from typing import Optional

import torch

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


@functools.lru_cache
def get_npu_device_name() -> Optional[str]:
    """Return ``torch_npu``'s device name for the current NPU, or ``None`` if unavailable."""
    try:
        if not torch.npu.is_available():
            return None
    except Exception as e:
        logger.debug(f"NPU not queryable for device name: {e}")
        return None
    try:
        import torch_npu

        idx = torch.npu.current_device()
        return str(torch_npu.npu.get_device_name(idx))
    except Exception as e:
        logger.debug(f"Failed to query NPU device name: {e}")
        return None


@functools.lru_cache
def is_ascend_npu_950_series() -> bool:
    """True when the visible Ascend NPU reports a 950-series SOC name (e.g. ``Ascend950PR_*``)."""
    name = get_npu_device_name()
    if not name:
        return False
    return name.startswith("Ascend950")


@functools.lru_cache
def get_platform() -> Literal["npu", "mlu", "meta_device", "ilu"]:
    """
    Detect whether the system has NPU or MLU, fallback device is meta_device.
    """
    try:
        if torch.cuda.get_device_name().startswith("Iluvatar"):
            logger.info("Iluvatar GPU detected")
            return "ilu"
    except Exception as e:
        logger.debug(f"Failed to check Iluvatar GPU availability: {e}")
    try:
        if torch.npu.is_available():
            logger.info("Ascend NPU detected")
            soc = get_npu_device_name()
            if soc and is_ascend_npu_950_series():
                logger.info("Ascend 950-series NPU: %s", soc)
            elif soc:
                logger.info("Ascend NPU SOC: %s", soc)
            return "npu"
    except Exception as e:
        logger.debug(f"Failed to check NPU availability: {e}")
    try:
        if torch.mlu.is_available():
            logger.info("Cambricon MLU detected")
            return "mlu"
    except Exception as e:
        logger.debug(f"Failed to check MLU availability: {e}")

    logger.warning("No accelerator detected")
    return "meta_device"


_PLATFORM_TO_TORCH_DEVICE = {
    "npu": "npu",
    "mlu": "mlu",
    "ilu": "cuda",
    "meta_device": "meta",
}


@functools.lru_cache
def get_torch_device() -> str:
    """Map the internal platform identifier to a PyTorch-recognised device string."""
    platform = get_platform()
    return _PLATFORM_TO_TORCH_DEVICE.get(platform, platform)


_PLATFORM_TO_DIST_BACKEND = {
    "npu": "hccl",
    "mlu": "cncl",
    "meta_device": "gloo",
}


@functools.lru_cache
def get_dist_backend() -> str:
    """Return the distributed communication backend for the current platform.

    Mapping:
        npu  → hccl
        mlu  → cncl
        else → gloo
    """
    return _PLATFORM_TO_DIST_BACKEND.get(get_platform(), "gloo")


def get_impl_by_platform():
    import_op_map = {}
    from mojo_opset.core.function import MojoFunction
    from mojo_opset.core.operator import MojoOperator

    platform = get_platform()

    try:
        caller_frame = inspect.stack()[1]
        caller_module = inspect.getmodule(caller_frame[0])

        if not caller_module or not hasattr(caller_module, "__file__"):
            logger.error("Could not determine the caller's module file path. Cannot discover operators.")
            return {}

        caller_dir = os.path.dirname(caller_module.__file__)
        package_name = getattr(caller_module, "__package__", "")

        api_dir_lists = ["operators", "functions"]

        for api_dir in api_dir_lists:
            api_dir_path = os.path.join(caller_dir, api_dir)
            api_package_name = f"{package_name}.{api_dir}"

            for _, module_name, _ in pkgutil.iter_modules([api_dir_path]):
                full_module_name = f"{api_package_name}.{module_name}"
                module = importlib.import_module(full_module_name)

                for name, op in inspect.getmembers(module, inspect.isclass):
                    if (
                        (issubclass(op, MojoOperator) or issubclass(op, MojoFunction))
                        and op not in [MojoOperator, MojoFunction]
                        and op.__module__ == full_module_name
                        and platform in getattr(op, "supported_platforms_list", [])
                    ):
                        logger.debug(f"Found supported operator '{name}' in {full_module_name}")
                        import_op_map[name] = op

    except (ImportError, IndexError):
        import traceback

        logger.error(f"Failed to discover operators: {traceback.format_exc()}")
        return {}

    return import_op_map
