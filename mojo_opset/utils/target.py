"""Target identity and on-demand platform/device detection."""

import os
import re

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
from typing import Optional
from typing import Union


@dataclass(frozen=True)
class Target:
    platform: str
    arch: str
    sku: str = ""

    def __post_init__(self):
        for field in ("platform", "arch", "sku"):
            value = getattr(self, field)
            if not isinstance(value, str):
                raise ValueError(f"Target {field} must be a string, got {value!r}.")
            value = value.strip().lower()
            if not value and field == "sku":
                object.__setattr__(self, field, "")
                continue
            if not value or not value.isascii() or not value.replace("_", "").isalnum():
                raise ValueError(f"Invalid target {field}: {value!r}.")
            object.__setattr__(self, field, value)

    @classmethod
    def parse(cls, value: "TargetLike") -> "Target":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"Target must be a Target or string, got {value!r}.")
        parts = value.strip().split(".")
        if len(parts) not in (2, 3) or any(not part.strip() for part in parts):
            raise ValueError(f"Target must have '<platform>.<arch>[.<sku>]' form, got {value!r}.")
        return cls(*parts)

    @property
    def arch_key(self) -> str:
        return f"{self.platform}.{self.arch}"

    @property
    def key(self) -> str:
        return f"{self.arch_key}.{self.sku}" if self.sku else self.arch_key

    def __str__(self) -> str:
        return self.key


TargetLike = Union[str, Target]
_AUTO_TARGET = None
_PLATFORM_TO_TORCH_DEVICE = {"npu": "npu", "mlu": "mlu", "ilu": "cuda", "meta_device": "meta"}


@lru_cache(maxsize=1)
def get_platform() -> Literal["npu", "mlu", "meta_device", "ilu"]:
    """Detect an accelerator lazily; meta_device means none was detected."""
    import torch

    try:
        if torch.cuda.get_device_name().startswith("Iluvatar"):
            return "ilu"
    except Exception:
        pass
    for platform in ("npu", "mlu"):
        backend = getattr(torch, platform, None)
        try:
            if backend is not None and backend.is_available():
                return platform
        except Exception:
            pass
    return "meta_device"


def get_torch_device() -> str:
    """Return the device type, not a device ordinal; reuse cached platform detection."""
    return _PLATFORM_TO_TORCH_DEVICE[get_platform()]


def _detect_npu_target() -> Target:
    override = os.getenv("MOJO_ASCEND_ARCH")
    if override:
        selected = Target.parse(f"npu.{override.strip()}")
        if selected.arch not in {"a2", "a5"}:
            raise ValueError(f"MOJO_ASCEND_ARCH must select 'a2' or 'a5', optionally with a SKU, got {override!r}.")
        return selected

    import torch

    try:
        name = str(torch.npu.get_device_name())
    except Exception as exc:
        raise RuntimeError("Unable to detect the NPU target; set MOJO_ASCEND_ARCH before first use.") from exc
    sku_match = re.match(r"^Ascend\s*950\s*(PR|DT)(?=$|[^a-z])", name, re.IGNORECASE)
    if sku_match:
        return Target("npu", "a5", f"950{sku_match.group(1).lower()}")
    if re.match(r"^Ascend\s*950", name, re.IGNORECASE):
        return Target("npu", "a5")
    a2_match = re.match(r"^Ascend\s*(910B[0-9][a-z0-9]*)(?=$|[^a-z0-9])", name, re.IGNORECASE)
    if a2_match:
        return Target("npu", "a2", a2_match.group(1).lower())
    if re.match(r"^Ascend\s*910B", name, re.IGNORECASE):
        return Target("npu", "a2")
    raise RuntimeError(f"Unsupported NPU device {name!r}; set MOJO_ASCEND_ARCH before first use.")


def detect_target(explicit: Optional[TargetLike] = None) -> Target:
    """Explicit selection wins; otherwise detect and cache the process default."""
    global _AUTO_TARGET
    if explicit is not None:
        return Target.parse(explicit)
    if _AUTO_TARGET is not None:
        return _AUTO_TARGET

    import torch

    if torch.compiler.is_compiling():
        raise RuntimeError("Detect the Mojo target before compilation with mojo_opset.preload or eager warmup.")
    platform = get_platform()
    if platform == "npu":
        result = _detect_npu_target()
    elif platform in ("ilu", "mlu"):
        result = Target(platform, os.getenv(f"MOJO_{platform.upper()}_ARCH", "generic"))
    else:
        result = Target("cpu", "generic")
    _AUTO_TARGET = result
    return result
