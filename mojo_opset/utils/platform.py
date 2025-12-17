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


def get_impl_by_platform():
    import_op_map = {}
    from mojo_opset.core.mojo_operator import MojoOperator

    platform = get_platform()

    try:
        caller_frame = inspect.stack()[1]
        caller_module = inspect.getmodule(caller_frame[0])

        if not caller_module or not hasattr(caller_module, "__file__"):
            logger.error("Could not determine the caller's module file path. Cannot discover operators.")
            return {}

        caller_dir = os.path.dirname(caller_module.__file__)
        package_name = getattr(caller_module, "__package__", "")

        for _, module_name, _ in pkgutil.iter_modules([caller_dir]):
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            for name, op in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(op, MojoOperator)
                    and op is not MojoOperator
                    and op.__module__ == full_module_name
                    and platform in getattr(op, "supported_platforms_list", [])
                ):
                    logger.debug(f"Found supported operator '{name}' in {full_module_name}")
                    import_op_map[name] = op

    except (ImportError, IndexError) as e:
        logger.error(f"Failed to discover operators: {e}")
        return {}

    return import_op_map
