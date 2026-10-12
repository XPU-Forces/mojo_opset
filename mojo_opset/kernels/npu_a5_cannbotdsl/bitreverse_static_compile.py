# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the repository.

"""Compile mixed SR kernels and validate native static UB without rewriting ASC.

Build files live as long as their loaded program. Set
CANNBOT_MX_QUANT_SR_KEEP_BUILD=1 before compilation to retain them for debugging.
Requires DSL native static UB and simt.brev support.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import weakref

_RESERVE = 40 * 1024
_KEEP_BUILD_ENV = "CANNBOT_MX_QUANT_SR_KEEP_BUILD"
# Layout -> (sample compile entry, generated ASC filename).
_LAYOUTS = {
    "tail": ("_compile_dsl_kernel", "run.asc"),
    "non_tail": ("_compile_dsl_non_tail_kernel", "run_non_tail.asc"),
    "non_tail_fused": ("_compile_dsl_non_tail_fused_kernel", "run_non_tail_fused.asc"),
    "wide_non_tail": ("_compile_dsl_wide_non_tail_kernel", "run_wide_non_tail.asc"),
    "wide32_non_tail": ("_compile_dsl_wide32_non_tail_kernel", "run_wide32_non_tail.asc"),
    "wide8_non_tail": ("_compile_dsl_wide8_non_tail_kernel", "run_wide8_non_tail.asc"),
    # Width 16 reuses the width-8 kernel with a compile-time lane count.
    "wide16_non_tail": ("_compile_dsl_wide16_non_tail_kernel", "run_wide8_non_tail.asc"),
}
_STATIC_ARRAY = re.compile(
    r"^    __ubuf__ uint8_t (?P<storage>ub_storage\d+)\[(?P<size>\d+)\];$",
    re.MULTILINE,
)
_STATIC_ROOT = re.compile(
    r"^    __ubuf__ (?P<ctype>\w+)\* (?P<pointer>buf\d+) = "
    r"reinterpret_cast<__ubuf__ (?P=ctype)\*>\("
    r"reinterpret_cast<__ubuf__ uint8_t\*>\((?P<storage>ub_storage\d+)\)"
    r" \+ (?P<offset>\d+)\);$",
    re.MULTILINE,
)


def _static_ub_source(source, input_dtype_name, tiling, layout="tail"):
    """Validate native UB declarations against tiling; return unchanged ASC."""
    if layout not in _LAYOUTS:
        raise ValueError(f"Unknown mixed SR layout: {layout!r}")
    if input_dtype_name not in ("fp32", "bf16"):
        raise ValueError("Static UB validation supports only fp32/bf16 input")
    tile, scales = tiling.tile_elements, tiling.scale_capacity
    item = 4 if input_dtype_name == "fp32" else 2
    input_type = "float" if input_dtype_name == "fp32" else "bfloat16_t"
    buffers = [
        ("input0", input_type, tile * item), ("input1", input_type, tile * item),
        ("output0", "uint8_t", tile), ("output1", "uint8_t", tile),
        ("scale0", "uint8_t", tile // 16), ("scale1", "uint8_t", tile // 16),
        ("scale32", "uint32_t", scales * 4),
        ("words", "int32_t", tile), ("reversed_words", "int32_t", tile),
    ]
    if layout == "non_tail_fused":
        buffers.extend([
            ("saved_output", "uint8_t", tile),
            ("saved_scale", "uint8_t", tile // 16),
            ("interleaved_output", "uint8_t", 2 * tile),
            ("interleaved_scale", "uint16_t", tile // 8),
        ])
    used = sum(size for _, _, size in buffers)
    budget = tiling.ub_bytes - _RESERVE
    if used != tiling.buffer_bytes or any(size <= 0 or size % 32 for _, _, size in buffers):
        raise ValueError("Static UB layout disagrees with the sample tiling")
    if used > budget:
        raise ValueError(f"Static UB uses {used} bytes, exceeding the {budget}-byte budget")

    kernels = list(re.finditer(
        r"^__vector__ __global__ __aicore__ void (kernel__BitReverseKernel_\w+)"
        r"\([^\n]*\)\n\{.*?^\}", source, re.MULTILINE | re.DOTALL,
    ))
    arrays = list(_STATIC_ARRAY.finditer(source))
    roots = list(_STATIC_ROOT.finditer(source))
    launches = re.findall(r"(\w+)<<<([^,\n]+),\s*([^,\n]+),\s*([^\n]+?)>>>", source)
    if len(kernels) != 1:
        raise ValueError("Expected one mixed kernel")
    if len(launches) != 1 or launches[0][0] != kernels[0][1] or launches[0][2].strip() != "0":
        raise ValueError("Expected one kernel launch with dynamic UB parameter 0")
    if len(arrays) != 1 or int(arrays[0]["size"]) != used:
        raise ValueError("Expected native DSL static UB with the sample tiling size")
    kernel = kernels[0]
    if not (kernel.start() < arrays[0].start() < arrays[0].end() < kernel.end()):
        raise ValueError("Static UB array lies outside the mixed kernel")

    omitted = set()
    # Some fused VFs use direct byte offsets and omit the unused scale32 root.
    if layout not in ("tail", "non_tail") and len(roots) == len(buffers) - 1:
        omitted.add("scale32")
    expected = []
    offset = 0
    for name, ctype, size in buffers:
        if name not in omitted:
            expected.append((name, ctype, size, offset))
        offset += size
    if len(roots) != len(expected):
        raise ValueError("Unexpected number of native static UB roots")

    manifest_buffers = []
    storage = arrays[0]["storage"]
    for root, (name, ctype, size, offset) in zip(roots, expected):
        if not (kernel.start() < root.start() < root.end() < kernel.end()):
            raise ValueError("Static UB root lies outside the mixed kernel")
        if root["storage"] != storage or root["ctype"] != ctype or int(root["offset"]) != offset:
            raise ValueError(f"Unexpected {name} static UB root: {root.group(0).strip()}")
        manifest_buffers.append(dict(
            name=name, ctype=ctype, pointer=root["pointer"],
            offset=offset, bytes=size, end=offset + size,
        ))
    return source, dict(
        ub_bytes=tiling.ub_bytes, reserve_bytes=_RESERVE, budget_bytes=budget,
        used_bytes=used, buffers=manifest_buffers, omitted_roots=sorted(omitted),
        dynamic_ub_bytes=0, static_array=storage, static_layout="dsl-emitted",
    )


def _phase(
    phase, sample_path, input_dtype_name, e5m2, ub_bytes, output_dir,
    scale_alg=1, max_low_bound_bits=0, layout="tail",
):
    # Isolate dump/cache options in the compiler child, preserving the target.
    arch = os.environ.get("CANNBOTDSL_NPU_ARCH", "dav-3510")
    for name in list(os.environ):
        if name.startswith("CANNBOTDSL_"):
            del os.environ[name]
    os.environ.update(
        CANNBOTDSL_NPU_ARCH=arch, CANNBOTDSL_DUMP_DIR=str(output_dir),
        CANNBOTDSL_CACHE_MEM_DISABLE="1", CANNBOTDSL_CACHE_DISK_DISABLE="1",
        CANNBOTDSL_PIPE_STAGE=phase,
        CANNBOTDSL_DUMP_ASCENDC="1" if phase == "translate" else "0",
        CANNBOTDSL_REUSE_ASCENDC="0" if phase == "translate" else "1",
    )
    import cannbotdsl
    from cannbotdsl import dtypes

    name = "_dynamic_mx_quant_static_source"
    spec = importlib.util.spec_from_file_location(name, sample_path)
    sample = importlib.util.module_from_spec(spec)
    sys.modules[name] = sample
    spec.loader.exec_module(sample)
    dtype = dtypes.float32 if input_dtype_name == "fp32" else dtypes.bfloat16
    tiling = sample._make_tiling(dtype, ub_bytes, layout == "non_tail_fused")
    compile_name, asc_name = _LAYOUTS[layout]
    program = getattr(sample, compile_name)(
        dtype, e5m2, ub_bytes, scale_alg, max_low_bound_bits,
    )
    if phase == "translate":
        source = (output_dir / asc_name).read_text()
        validated, manifest = _static_ub_source(source, input_dtype_name, tiling, layout)
        (output_dir / "static_ub.asc").write_text(validated)
        (output_dir / "original.asc").write_text(source)
        manifest.update(
            dsl_version=cannbotdsl.__version__, npu_arch=arch,
            input_dtype=input_dtype_name, e5m2=e5m2, scale_alg=scale_alg,
            max_low_bound_bits=max_low_bound_bits, layout=layout, sample=str(sample_path),
            original_sha256=hashlib.sha256(source.encode()).hexdigest(),
            static_sha256=hashlib.sha256(validated.encode()).hexdigest(),
            bitreverse="native DSL simt.brev",
            adaptation="Native static UB validated; ASC and launch unchanged.",
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        program.save(str(output_dir / "program.so"))
    print(f"STATIC_UB_{phase.upper()}_PASS: {output_dir}", flush=True)


def build_static_program(
    sample_path, input_dtype_name, e5m2, ub_bytes, scale_alg=1,
    max_low_bound_bits=0, *, layout="tail",
):
    """Load a specialization and own its files for the lifetime of the program."""
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {tuple(_LAYOUTS)}")
    sample_path = Path(sample_path).resolve()
    owner = None
    if os.environ.get(_KEEP_BUILD_ENV) == "1":
        output_dir = Path(tempfile.mkdtemp(prefix="dynamic-mx-bitreverse-"))
    else:
        owner = tempfile.TemporaryDirectory(prefix="dynamic-mx-bitreverse-")
        output_dir = Path(owner.name)
    try:
        for phase in ("translate", "compile"):
            command = [
                sys.executable, str(Path(__file__).resolve()), phase, str(sample_path),
                input_dtype_name, str(int(e5m2)), str(ub_bytes), str(output_dir),
                "--scale-alg", str(scale_alg),
                "--max-low-bound-bits", str(max_low_bound_bits), "--layout", layout,
            ]
            log_path = output_dir / f"{phase}.log"
            try:
                with log_path.open("w") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        timeout=600, check=False,
                    )
            except subprocess.TimeoutExpired as error:
                details = log_path.read_text(errors="replace")[-12000:]
                raise RuntimeError(
                    f"Static UB {phase} timed out. Set {_KEEP_BUILD_ENV}=1 to retain builds.\n"
                    f"{details}"
                ) from error
            if result.returncode:
                details = log_path.read_text(errors="replace")[-12000:]
                raise RuntimeError(
                    f"Static UB {phase} failed (exit {result.returncode}). "
                    f"Set {_KEEP_BUILD_ENV}=1 to retain builds.\n{details}"
                )
        from cannbotdsl import load

        program = load(str(output_dir / "program.so"))
        if owner is not None:
            # ProviderCallable.save and binary inspection still need these files.
            weakref.finalize(program, owner.cleanup)
        return program
    except BaseException:
        if owner is not None:
            owner.cleanup()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("translate", "compile"))
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("input_dtype_name", choices=("fp32", "bf16"))
    parser.add_argument("e5m2", type=int, choices=(0, 1))
    parser.add_argument("ub_bytes", type=int)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scale-alg", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-low-bound-bits", type=int, default=0)
    parser.add_argument("--layout", choices=tuple(_LAYOUTS), default="tail")
    args = parser.parse_args()
    _phase(
        args.phase, args.sample_path, args.input_dtype_name, bool(args.e5m2),
        args.ub_bytes, args.output_dir, args.scale_alg, args.max_low_bound_bits, args.layout,
    )
