"""Framework RNG contracts, native bridge loading, and raw state reservation.

Importing this module never loads the native bridge or initializes an NPU.
"""

from threading import Lock

ELEMENTS_PER_PHILOX_BLOCK = 16
RNG_INCREMENT = 4  # Minimum reservation, in32-bit words; actual budget is shape-dependent.
RNG_VIRTUAL_STREAMS = 16384
# Each current SIMD RNG batch generates 64 blocks, even for a partial batch.
MAX_PADDED_GROUPS = 63
UINT64_MAX = (1 << 64) - 1
_BRIDGE_LOCK = Lock()
_BRIDGE = None


def _check_uint64(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, not bool or a converted scalar")
    if not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64-1]")


def _rng_increment(numel: int) -> int:
    """Reserve complete Philox blocks per virtual stream, including padding.

    Offset follows the framework/CANN convention: four32-bit words per block.
    Physical core count and tiling do not affect logical random addresses.
    """
    _check_uint64("numel", numel)
    if numel == 0 or numel % ELEMENTS_PER_PHILOX_BLOCK:
        raise ValueError("numel must be a positive multiple of 16")
    groups = numel // ELEMENTS_PER_PHILOX_BLOCK + MAX_PADDED_GROUPS
    return 4 * ((groups + RNG_VIRTUAL_STREAMS - 1) // RNG_VIRTUAL_STREAMS)


def _validate_rng_geometry(numel: int, quant_length: int) -> None:
    """Validate logical geometry; linear groups fit uint64 including padding."""
    _rng_increment(numel)
    _check_uint64("quant_length", quant_length)
    if quant_length == 0 or quant_length % ELEMENTS_PER_PHILOX_BLOCK:
        raise ValueError("quant_length must be a positive multiple of 16")
    if numel % quant_length:
        raise ValueError("quant_length must divide numel")


def _validate_rng_args(seed: int, block_offset: int) -> None:
    """Validate a trusted replay word offset, aligned to a Philox block."""
    for name, value in (("seed", seed), ("block_offset", block_offset)):
        _check_uint64(name, value)
    if block_offset % 4:
        raise ValueError("block_offset is a32-bit word offset and must be four-aligned")
    if block_offset > UINT64_MAX - RNG_INCREMENT:
        raise ValueError("block_offset + 4 would overflow uint64")


def _validate_framework_increment(increment: int) -> None:
    """The NPU generator requires a positive four-aligned request."""
    _check_uint64("increment", increment)
    if increment == 0 or increment % 4:
        raise ValueError("increment must be a positive multiple of four")


def _load_bridge():
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is not None:
            return _BRIDGE
        import hashlib
        import sys

        from pathlib import Path

        import torch
        import torch_npu

        from torch.utils.cpp_extension import load

        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm up SR RNG bridge before graph capture")
        source = Path(__file__).resolve().parent / "csrc" / "rng_bridge.cpp"
        npu_root = Path(torch_npu.__file__).resolve().parent
        identity = (
            sys.implementation.cache_tag,
            sys.version,
            torch.__version__,
            torch_npu.__version__,
            getattr(torch_npu.version, "git_version", ""),
            torch._C._GLIBCXX_USE_CXX11_ABI,
        )
        digest = hashlib.sha256(source.read_bytes() + repr(identity).encode()).hexdigest()[:20]
        includes = [
            npu_root / "include",
            npu_root / "include/third_party/acl/inc",
            npu_root / "include/third_party/hccl/inc",
        ]
        _BRIDGE = load(
            name=f"sr_rng_bridge_{digest}",
            sources=[str(source)],
            extra_include_paths=[str(path) for path in includes],
            extra_cflags=["-O2"],
            extra_ldflags=[f"-L{npu_root / 'lib'}", "-ltorch_npu", f"-Wl,-rpath,{npu_root / 'lib'}"],
            with_cuda=False,
            verbose=False,
        )
        return _BRIDGE


def _resolved_device(device):
    import torch
    import torch_npu  # noqa: F401  # provider registration, no implicit device choice

    device = torch.device(device)
    if device.type not in ("npu", "privateuseone") or device.index is None:
        raise ValueError("RNG state requires an explicitly indexed NPU device")
    return device


def _reserve_rng_carrier(device, increment):
    """Return framework state once, preserving device owners for capture."""
    _validate_framework_increment(increment)
    import torch

    device = _resolved_device(device)
    with torch.npu.device(device):
        bridge = _load_bridge()
        return bridge.reserve_philox_state(device.index, increment)


def _make_rng_state(seed: int, block_offset: int, *, device):
    """Construct trusted replay state without consuming the default generator."""
    _validate_rng_args(seed, block_offset)
    import torch

    device = _resolved_device(device)
    signed = [value if value < 2**63 else value - 2**64 for value in (seed, block_offset)]
    with torch.npu.device(device):
        return torch.tensor(signed, dtype=torch.int64, device=device)
