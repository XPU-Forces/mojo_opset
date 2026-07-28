"""Runtime dispatch layer for Ascend SoC-generation kernel selection.

Kernels live under ``a2/`` (Ascend 910 series) and ``a5/`` (Ascend 950 series).
This module inspects the active NPU at import time, picks a primary
generation, and registers each kernel module under its unqualified public
path (``mojo_opset.backends.ttx.kernels.npu.<name>``).

For same-name ``.py`` files present in both ``a2/`` and ``a5/``, attribute
lookup on the public module tries the primary generation first and falls
back to the secondary — so an a5 file that only overrides some symbols
still lets the rest resolve to the a2 implementation.

Sub-packages (e.g. ``over_encoding/``) use whole-package substitution:
no cross-generation symbol merging.

Detection order:
    1. ``MOJO_ASCEND_ARCH`` env var (``a2`` or ``a5``).
    2. ``triton.runtime.driver.active.get_current_target().arch`` — string
       containing ``"Ascend950"`` → ``a5``; otherwise ``a2``.
    3. If step 2 raises (no NPU / driver not initialized), default ``a2``.
"""

import importlib
import logging
import os
import pathlib
import sys
import types

_logger = logging.getLogger(__name__)

_A2 = "a2"
_A5 = "a5"
_VALID_ARCHES = (_A2, _A5)


def _detect_arch() -> str:
    override = os.environ.get("MOJO_ASCEND_ARCH")
    if override:
        if override in _VALID_ARCHES:
            _logger.info("Ascend arch = %s (from MOJO_ASCEND_ARCH env var).", override)
            return override
        _logger.warning(
            "Ignoring MOJO_ASCEND_ARCH=%r; expected one of %s. Defaulting to %s.",
            override, _VALID_ARCHES, _A2,
        )
        return _A2

    try:
        import triton  # local import to avoid hard dep at package import
        arch_str = str(triton.runtime.driver.active.get_current_target().arch)
    except Exception as exc:
        _logger.warning(
            "Could not detect Ascend arch via triton (%s: %s); defaulting to %s.",
            type(exc).__name__, exc, _A2,
        )
        return _A2

    if "Ascend950" in arch_str:
        _logger.info("Ascend arch = a5 (triton reports %r).", arch_str)
        return _A5
    if "Ascend910" in arch_str:
        _logger.info("Ascend arch = a2 (triton reports %r).", arch_str)
        return _A2
    _logger.warning("Unrecognized Ascend arch %r; defaulting to %s.", arch_str, _A2)
    return _A2


class _MergedModule(types.ModuleType):
    """Module facade: attribute lookup prefers ``primary``, falls back to ``secondary``."""

    def __init__(self, name: str, primary: types.ModuleType, secondary: types.ModuleType):
        super().__init__(name)
        self.__dict__["_primary"] = primary
        self.__dict__["_secondary"] = secondary
        primary_file = getattr(primary, "__file__", None)
        if primary_file is not None:
            self.__file__ = primary_file

    def __getattr__(self, item: str):
        primary = self.__dict__["_primary"]
        try:
            return getattr(primary, item)
        except AttributeError:
            pass
        secondary = self.__dict__["_secondary"]
        try:
            return getattr(secondary, item)
        except AttributeError:
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {item!r} "
                f"in either {primary.__name__!r} or {secondary.__name__!r}"
            )

    def __dir__(self):
        entries = set(dir(self.__dict__["_primary"]))
        entries.update(dir(self.__dict__["_secondary"]))
        return sorted(entries)


def _discover(subdir: pathlib.Path) -> dict:
    """Return {name: is_package} for public modules directly under subdir."""
    found = {}
    if not subdir.is_dir():
        return found
    for entry in subdir.iterdir():
        name = entry.name
        if name.startswith("_") or name.startswith("."):
            continue
        if entry.is_file() and name.endswith(".py"):
            found[name[:-3]] = False
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            found[name] = True
    return found


def _try_import(fullname: str):
    try:
        return importlib.import_module(fullname)
    except ModuleNotFoundError:
        return None


def _install(package: str, name: str, arch: str) -> None:
    a2_mod = _try_import(f"{package}.{_A2}.{name}")
    a5_mod = _try_import(f"{package}.{_A5}.{name}") if arch == _A5 else None

    if a2_mod is None and a5_mod is None:
        return

    if arch == _A5 and a5_mod is not None and a2_mod is not None:
        is_package = hasattr(a5_mod, "__path__") or hasattr(a2_mod, "__path__")
        if is_package:
            # Whole-package substitution for subpackages (no symbol merging).
            chosen = a5_mod
            chosen_prefix = f"{package}.{_A5}.{name}"
        else:
            chosen = _MergedModule(f"{package}.{name}", primary=a5_mod, secondary=a2_mod)
            chosen_prefix = None
    elif arch == _A5 and a5_mod is not None:
        chosen = a5_mod
        chosen_prefix = f"{package}.{_A5}.{name}" if hasattr(a5_mod, "__path__") else None
    else:
        chosen = a2_mod
        chosen_prefix = f"{package}.{_A2}.{name}" if hasattr(a2_mod, "__path__") else None

    public_name = f"{package}.{name}"
    sys.modules[public_name] = chosen
    setattr(sys.modules[package], name, chosen)

    # For subpackages, also alias every already-loaded submodule under the
    # public name so that later `from <package>.<name>.<sub> import X`
    # reuses the same module object instead of triggering a second load
    # (which would duplicate top-level defs like triton JITted functions).
    if chosen_prefix is not None:
        prefix_dot = chosen_prefix + "."
        for full, mod in list(sys.modules.items()):
            if full.startswith(prefix_dot):
                alias = public_name + full[len(chosen_prefix):]
                sys.modules.setdefault(alias, mod)


def _install_all() -> str:
    package = __package__
    here = pathlib.Path(__file__).resolve().parent
    arch = _detect_arch()

    names = _discover(here / _A2)
    if arch == _A5:
        for name, is_pkg in _discover(here / _A5).items():
            names.setdefault(name, is_pkg)

    for name in names:
        _install(package, name, arch)

    _logger.debug("Dispatched npu kernels for arch=%s (%d modules).", arch, len(names))
    return arch


ACTIVE_ARCH = _install_all()
