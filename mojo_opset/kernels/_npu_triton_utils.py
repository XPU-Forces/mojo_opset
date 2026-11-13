from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

import torch
import triton
import triton.language as tl

try:
    from triton.runtime.libentry import libentry
except ImportError:

    def libentry():
        """No-op fallback when triton.runtime.libentry is unavailable."""
        def _decorator(fn):
            return fn

        return _decorator

VEC_ALIGN_BYTES = 256
SRAM_ALIGN_BYTES = 32


@lru_cache(maxsize=1)
@torch.compiler.assume_constant_result
def get_num_cores(op_type="vector"):
    assert op_type in ["vector", "cube", "mix"], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
    return (
        triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        if op_type == "vector"
        else triton.runtime.driver.active.utils.get_device_properties("npu")["num_aicore"]
    )
    
def is_910() -> bool:
    target=triton.runtime.driver.active.get_current_target()
    return "Ascend910" in target.arch

# npu triton only
exp = tl.exp
exp2 = tl.math.exp2
log = tl.log
log2 = tl.log2
gather = tl.gather


def _loaded_distribution_version(package):
    spec = find_spec("triton")
    if spec is None or spec.origin is None:
        return None
    try:
        installed = distribution(package)
    except PackageNotFoundError:
        return None
    # A venv or PYTHONPATH can override code while leaving system metadata visible.
    if Path(installed.locate_file("triton/__init__.py")).resolve() != Path(spec.origin).resolve():
        return None
    return installed.version


def get_ascend_compiler_version():
    """Return (public/internal, parsed version), or (None, None) if unknown/ambiguous."""
    candidates = [("public", _loaded_distribution_version("triton-ascend"))]
    private_name = __name__ + "_private"
    if find_spec(private_name) is not None:
        private = import_module(private_name)
        candidates.append(("internal", private.get_ascend_compiler_version()))
    matches = [(kind, version) for kind, version in candidates if version is not None]
    if len(matches) != 1:
        return None, None
    try:
        from packaging.version import InvalidVersion
        from packaging.version import Version
    except ImportError:
        return None, None
    kind, version = matches[0]
    try:
        return kind, Version(version)
    except InvalidVersion:
        return None, None
