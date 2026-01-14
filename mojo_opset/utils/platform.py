import functools
import importlib
import inspect
import os
import pkgutil
import subprocess

from typing import Literal

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


@functools.lru_cache
def get_platform() -> Literal["npu", "mlu", "meta_device"]:
    """
    Detect whether the system has NPU or MLU, fallback device is meta_device.
    """
    try:
        subprocess.run(
            ["npu-smi", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.debug("Ascend NPU detected")
        return "npu"
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            subprocess.run(["cnmon"], check=True)
            logger.debug("Cambricon MLU detected")
            return "mlu"
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("No accelerator detected")
            return "meta_device"


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
                        issubclass(op, MojoOperator)
                        or issubclass(op, MojoFunction)
                        and op not in [MojoOperator, MojoFunction]
                        and op.__module__ == full_module_name
                        and platform in getattr(op, "supported_platforms_list", [])
                    ):
                        logger.debug(f"Found supported operator '{name}' in {full_module_name}")
                        import_op_map[name] = op

    except (ImportError, IndexError) as e:
        import traceback

        logger.error(f"Failed to discover operators: {traceback.format_exc()}")
        return {}

    return import_op_map
