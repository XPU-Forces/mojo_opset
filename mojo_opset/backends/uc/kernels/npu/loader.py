import importlib.util

from functools import lru_cache
from pathlib import Path


_LIB_DIR = Path(__file__).resolve().parent / "kernel_lib"


@lru_cache(maxsize=None)
def load_uc_pybind_module(module_name: str):
    candidates = sorted(_LIB_DIR.glob(f"{module_name}*.so"))
    if not candidates:
        raise FileNotFoundError(
            f"UC pybind module '{module_name}' was not found in {_LIB_DIR}. "
            "Build it from unified_compiler/external/uc-test with "
            "mojo/ascend/export_kernel.py."
        )

    so_path = candidates[0]
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load UC pybind module from {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
