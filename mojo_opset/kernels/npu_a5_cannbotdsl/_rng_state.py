# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the repository.

"""RNG reservation and device state materialization, initialized on demand.

Importing this module never loads the native bridge or initializes an NPU.
"""

from threading import Lock


ELEMENTS_PER_PHILOX_BLOCK = 16
RNG_INCREMENT = 1024
# Each current SIMD RNG batch generates 64 blocks, even for a partial batch.
MAX_PADDED_GROUPS = 63
UINT64_MAX = (1 << 64) - 1
_BRIDGE_LOCK = Lock()
_BRIDGE = None
_UNPACK_INIT_LOCK = Lock()
_UNPACK_LAUNCH = None
_UNPACK_PROGRAMS = None


def _check_uint64(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, not bool or a converted scalar")
    if not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64-1]")


def _rng_increment(numel: int) -> int:
    """TE NVFP4's fixed raw reservation; each logical substream uses <=8 blocks."""
    _check_uint64("numel", numel)
    if numel == 0 or numel % ELEMENTS_PER_PHILOX_BLOCK:
        raise ValueError("numel must be a positive multiple of 16")
    return RNG_INCREMENT


def _validate_rng_geometry(numel: int, quant_length: int) -> None:
    """Bound original-matrix tasks, including every generated padding group."""
    _rng_increment(numel)
    _check_uint64("quant_length", quant_length)
    if quant_length == 0 or quant_length % ELEMENTS_PER_PHILOX_BLOCK:
        raise ValueError("quant_length must be a positive multiple of 16")
    if numel % quant_length:
        raise ValueError("quant_length must divide numel")
    axis_groups = quant_length // ELEMENTS_PER_PHILOX_BLOCK
    grid_x = (axis_groups + 7) // 8
    last_row = (numel // ELEMENTS_PER_PHILOX_BLOCK + MAX_PADDED_GROUPS - 1) // axis_groups
    subsequence_bound = (last_row // 128 + 1) * grid_x * 128 - 1
    if subsequence_bound > UINT64_MAX:
        raise ValueError("original-matrix subsequence would overflow uint64")


def _validate_rng_args(seed: int, block_offset: int) -> None:
    """Validate a trusted raw-block replay interval, without word alignment."""
    for name, value in (("seed", seed), ("block_offset", block_offset)):
        _check_uint64(name, value)
    if block_offset > UINT64_MAX - RNG_INCREMENT:
        raise ValueError("block_offset + 1024 would overflow uint64")


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
        identity = (sys.implementation.cache_tag, sys.version, torch.__version__,
                    torch_npu.__version__, getattr(torch_npu.version, "git_version", ""),
                    torch._C._GLIBCXX_USE_CXX11_ABI)
        digest = hashlib.sha256(source.read_bytes() + repr(identity).encode()).hexdigest()[:20]
        includes = [npu_root / "include", npu_root / "include/third_party/acl/inc",
                    npu_root / "include/third_party/hccl/inc"]
        _BRIDGE = load(
            name=f"sr_rng_bridge_{digest}", sources=[str(source)],
            extra_include_paths=[str(path) for path in includes],
            extra_cflags=["-O2"], extra_ldflags=[f"-L{npu_root / 'lib'}", "-ltorch_npu",
                                              f"-Wl,-rpath,{npu_root / 'lib'}"],
            with_cuda=False, verbose=False,
        )
        return _BRIDGE


def _resolved_device(device):
    import torch
    import torch_npu  # noqa: F401: provider registration, no implicit device choice
    device = torch.device(device)
    if device.type not in ("npu", "privateuseone") or device.index is None:
        raise ValueError("RNG state requires an explicitly indexed NPU device")
    return device


def _reserve_rng_state(device, increment: int):
    """Return int64[2] with seed and framework offset consumed as raw blocks."""
    _validate_framework_increment(increment)
    import torch
    device = _resolved_device(device)
    with torch.npu.device(device):
        bridge = _load_bridge()
        programs = prepare_unpack(device)
        output = torch.empty((2,), dtype=torch.int64, device=device)
        state = bridge.reserve_philox_state(device.index, increment)
        unpack_rng_state(state, output, programs)
    return output


def _make_rng_state(seed: int, block_offset: int, *, device):
    """Construct trusted replay state without consuming the default generator."""
    _validate_rng_args(seed, block_offset)
    import torch
    device = _resolved_device(device)
    signed = [value if value < 2**63 else value - 2**64 for value in (seed, block_offset)]
    with torch.npu.device(device):
        return torch.tensor(signed, dtype=torch.int64, device=device)



def _load_unpack_launch():
    """Define the DSL entry only when its first Program is built."""
    global _UNPACK_LAUNCH
    with _UNPACK_INIT_LOCK:
        if _UNPACK_LAUNCH is not None:
            return _UNPACK_LAUNCH
        from cannbotdsl import dtypes
        from cannbotdsl.lang import const_expr, host, kernel, vf

        @kernel
        class _UnpackRngKernel:
            def __init__(self, captured: const_expr):
                self.captured = captured

            def __call__(self, seed_tensor, offset_tensor, output,
                         seed_bits, offset_bits, intra_bits):
                # Scalar GM access reads exactly 8 bytes from each one-element tensor.
                with vf(mode="simt", thread=1):
                    if self.captured:
                        seed = dtypes.uint64(seed_tensor[0])
                        offset = dtypes.uint64(offset_tensor[0]) + dtypes.uint64(intra_bits)
                        output[0] = dtypes.int64(seed)
                        output[1] = dtypes.int64(offset)
                    else:
                        output[0] = seed_bits
                        output[1] = offset_bits

        class _UnpackLaunch:
            def __init__(self, captured):
                self.captured = captured

            @host
            def run(self, seed_tensor, offset_tensor, output,
                    seed_bits, offset_bits, intra_bits):
                op = _UnpackRngKernel(self.captured)
                op[1](seed_tensor, offset_tensor, output, seed_bits, offset_bits, intra_bits)

        _UNPACK_LAUNCH = _UnpackLaunch
        return _UNPACK_LAUNCH


def _unpack_program_cache():
    """Reuse the main module's cache after its imports have completed."""
    global _UNPACK_PROGRAMS
    from .dynamic_mx_quant_sr import _ProgramCache
    with _UNPACK_INIT_LOCK:
        if _UNPACK_PROGRAMS is None:
            _UNPACK_PROGRAMS = _ProgramCache(8)
        return _UNPACK_PROGRAMS


def _build_unpack_program(captured):
    import cannbotdsl
    from cannbotdsl import TensorSpec, dtypes
    launch = _load_unpack_launch()
    return cannbotdsl.compile(
        launch(captured).run,
        TensorSpec((1,), dtypes.int64), TensorSpec((1,), dtypes.int64),
        TensorSpec((2,), dtypes.int64), dtypes.int64, dtypes.int64, dtypes.int64,
    )


def prepare_unpack(device):
    """Warm both entry variants before reserve/capture; never cache RNG values."""
    import torch
    from .dynamic_mx_quant_sr import capture_auxiliary_context
    cache = _unpack_program_cache()
    programs = []
    for captured in (False, True):
        context = capture_auxiliary_context("sr_rng_unpack_v2", captured, device)

        def build():
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm up SR RNG state unpack before graph capture")
            return _build_unpack_program(captured)

        program = cache.get_or_build(
            context.key, build, bypass=context.bypass_reason is not None,
        )
        if not program.so_path:
            raise RuntimeError("SR RNG state unpack requires an executable AOT Program")
        programs.append(program)
    return tuple(programs)


def _signed64(value):
    return value if value < 2**63 else value - 2**64


def unpack_rng_state(state, output, programs):
    captured, secondary, seed, offset, intragraph = state
    if secondary:
        raise NotImplementedError("secondary-stream RNG capture requires separate validation")
    if captured:
        # Program's torch_npu integration records all tensor arguments on the
        # current stream; retaining this tuple owns framework state handles.
        args = (seed, offset, output, 0, 0, _signed64(intragraph))
    else:
        # Compile-time branch does not read these aliasing placeholder inputs.
        args = (output[:1], output[:1], output, _signed64(seed), _signed64(offset), 0)
    program = programs[int(captured)]
    program(*args)
    # Retain the launch/state owners while this output is live. Captured graph
    # lifetime beyond it is delegated to the verified native graph/runtime.
    output._sr_rng_owners = (program, state)


def clear_caches():
    """Release unpack Program references without touching bridge/RNG state."""
    with _UNPACK_INIT_LOCK:
        cache = _UNPACK_PROGRAMS
    if cache is not None:
        cache.clear()
