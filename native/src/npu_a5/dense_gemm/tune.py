"""Offline AOT profiling. Never compiles kernels or tunes in the serving process.

Run from the repository root. --catalog is the catalog.json from the native build.
Use --baseline with an older AOT wheel on PYTHONPATH for paired measurements.
"""

import argparse
import ctypes
import gc
import json
import os
import random
import runpy
import statistics

from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from mojo_opset import functions as F

SHAPES = [
    (32, 4096, 4096),
    (256, 512, 1024),
    (1024, 1024, 1024),
    (16, 1024, 1024),
    (64, 2048, 2048),
    (128, 1024, 4096),
    (512, 2048, 1024),
    (2048, 1024, 1024),
    (130, 258, 510),
    (514, 258, 510),
]
HOLDOUT = [(48, 1536, 2048), (192, 768, 1536), (768, 1536, 2048), (1536, 768, 512)]


def layouts(op):
    # The MXFP8 API exposes NT forward/dgrad and TN wgrad, not arbitrary transposes.
    return [(False, True), (True, False)] if op == "mxfp8_gemm" else [
        (False, False), (False, True), (True, False), (True, True)
    ]


def parse_case(case):
    op = case["op"]
    if op == "mxfp8_gemm":
        mode = case.get("mode", "forward")
        if mode not in ("forward", "wgrad"):
            raise ValueError("MXFP8 mode must be forward or wgrad")
        ta, tb = mode == "wgrad", mode == "forward"
        if case.get("trans_input", ta) != ta or case.get("trans_weight", tb) != tb:
            raise ValueError("MXFP8 transpose flags must agree with mode; use mode instead")
    else:
        ta, tb = case.get("trans_input", False), case.get("trans_weight", True)
    return op, case["shape"], ta, tb, {"bf16": torch.bfloat16, "fp32": torch.float32}[case.get("output", "bf16")]


def prepare_case(h, op, shape, ta, tb, dtype, device):
    """Prepare operands/reference outside timing and return the actual AOT geometry."""
    if op not in h["AOT_OPS"]:
        raise ValueError(f"Unsupported tuning op: {op}; use mxfp8_gemm for MXFP8")
    if len(shape) != 3 or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("Tuning requires positive integer M, N, K")
    m, n, k = shape
    if op == "mxfp8_gemm":
        if (ta, tb) not in layouts(op):
            raise ValueError("MXFP8 tuning requires NT forward or TN wgrad")
        mode, axis = ("wgrad", 0) if ta else ("forward", 1)
        shapes = ((k, m), (k, n)) if ta else ((m, k), (n, k))
        cpu_operands = tuple(
            F.mxfp8_quantize(torch.randn(s, dtype=torch.bfloat16, device="cpu"), axis=axis, implementation="torch")
            for s in shapes
        )
        kwargs = dict(mode=mode, output_dtype=dtype)
        reference = F.mxfp8_gemm(*cpu_operands, **kwargs, implementation="torch")
        tensors = tuple(replace(x, data=x.data.to(device), scale=x.scale.to(device)) for x in cpu_operands)
        a, b = tensors
        # Dispatch sees padded payloads, before the adapter crops the logical output.
        kernel_shape = (a.data.shape[1], b.data.shape[1], a.data.shape[0]) if ta else (
            a.data.shape[0], b.data.shape[0], a.data.shape[1]
        )
    else:
        tensors = h["inputs"](op, m, n, k, ta, tb, device, packed_sf=True)
        kwargs = dict(trans_input=ta, trans_weight=tb, output_dtype=dtype)
        reference = h["cpu_reference"](op, tensors, kwargs)
        kernel_shape = tuple(shape)
    key = h["AOT_OPS"].index(op) * 40 + ta * 20 + (not tb) * 10 + (dtype == torch.float32) * 5
    return tensors, kwargs, reference, kernel_shape, key


def device_info():
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    # Use torch-npu's device properties rather than assuming a 32-Cube chip.
    props = torch.npu.get_device_properties(torch.npu.current_device())
    return acl.aclrtGetSocName().decode(), props.cube_core_num


def profile(args):
    import torch_npu  # noqa: F401 - registers the NPU backend only for profiling

    torch.set_num_threads(4)
    h = runpy.run_path("tests/functions/test_dense_gemm.py")
    torch.npu.set_device(0)
    soc, cores = device_info()
    catalog = json.loads(args.catalog.read_text()) if args.catalog else []
    rng = random.Random(20261023)
    rows, tuning = [], {}
    os.environ.pop("MOJO_GEMM_FORCE_CONFIG", None)
    if not args.baseline and not args.compare_default:
        os.environ["MOJO_GEMM_DISABLE_TUNING"] = "1"
    else:
        os.environ.pop("MOJO_GEMM_DISABLE_TUNING", None)
    cases = [
        (op, shape, False, True, torch.bfloat16) for op in h["AOT_OPS"] for shape in (HOLDOUT if args.holdout else SHAPES)
    ]
    if args.layouts:
        cases = [
            (op, (128, 256, 512), ta, tb, dtype)
            for op in h["AOT_OPS"]
            for ta, tb in layouts(op)
            for dtype in (torch.bfloat16, torch.float32)
        ]
    if args.cases:
        requested = json.loads(args.cases.read_text())
        cases = [parse_case(c) for c in requested]
    matches = runpy.run_path(str(Path(__file__).with_name("generate.py")))["specialization_matches"]
    for op, (m, n, k), ta, tb, dtype in cases:
        torch.manual_seed(123)
        tensors, kwargs, reference, kernel_shape, key = prepare_case(h, op, (m, n, k), ta, tb, dtype, "npu")
        candidates = [c for c in catalog if c["key"] == key] if not args.baseline else []
        candidates = [c for c in candidates if matches(c, *kernel_shape, cores)]
        if args.specializations_only:
            base_tags = {c["base_tag"] for c in candidates if c.get("base_tag", c["tag"]) != c["tag"]}
            candidates = [c for c in candidates if c.get("base_tag", c["tag"]) in base_tags]
        rng.shuffle(candidates)
        # The model/baseline is measured in the same randomized replay rounds.
        candidates.insert(0, {"tag": None})
        graphs, measurements = [], {}
        for c in candidates:
            tag = c["tag"]
            if tag:
                os.environ["MOJO_GEMM_FORCE_CONFIG"] = tag
            else:
                os.environ.pop("MOJO_GEMM_FORCE_CONFIG", None)
            call = partial(getattr(F, op), *tensors, **kwargs, implementation="native")
            try:
                out = call()
            except RuntimeError as error:
                if tag and "Requested AOT configuration is unavailable or unsafe" in str(error):
                    continue
                raise
            h["compare"](out, reference)
            for _ in range(3):
                call()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(args.repeats):
                    captured = call()
            graph.replay()
            torch.npu.synchronize()
            h["compare"](captured, reference)
            graphs.append((tag, graph, captured))
            measurements[tag] = []
        for _ in range(args.rounds):
            rng.shuffle(graphs)
            for tag, graph, _captured in graphs:
                for _ in range(3):
                    graph.replay()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                measurements[tag].append(start.elapsed_time(end) * 1000 / args.repeats)
        medians = {tag: statistics.median(times) for tag, times in measurements.items()}
        tags = [tag for tag in medians if tag is not None]
        best = min(tags, key=medians.get) if tags else None
        row = dict(
            op=op,
            shape=[m, n, k],
            kernel_shape=list(kernel_shape),
            key=key,
            soc=soc,
            cores=cores,
            model_us=medians[None],
            best_tag=best,
            best_us=medians[best],
            candidates=[dict(tag=tag, us=medians[tag], samples=times) for tag, times in measurements.items()],
        )
        rows.append(row)
        if best and medians[best] < 0.97 * medians[None]:
            # Multiple logical MXFP8 shapes can share one padded dispatch shape.
            tuning[(key, kernel_shape)] = dict(soc=soc, cores=cores, key=key, shape=list(kernel_shape), tag=best)
        print(json.dumps({k: v for k, v in row.items() if k != "candidates"}), flush=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        if args.export:
            args.export.write_text(json.dumps(list(tuning.values()), indent=2) + "\n")
        del graphs, graph, captured, _captured, call, out, tensors, reference
        gc.collect()
        torch.npu.empty_cache()
    os.environ.pop("MOJO_GEMM_FORCE_CONFIG", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--cases", type=Path, help="JSON cases: op, shape (M,N,K), output (bf16/fp32); MXFP8 mode, otherwise trans_input/trans_weight"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", type=Path, help="Validated exact-shape entries for tuning.json")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--compare-default", action="store_true", help="Keep current tuning as the reference")
    parser.add_argument(
        "--specializations-only", action="store_true", help="Compare specializations and their base entries"
    )
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--layouts", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
    options = parser.parse_args()
    if not options.baseline and not options.catalog:
        parser.error("--catalog is required unless using --baseline")
    if options.rounds < 3 or options.repeats < 1:
        parser.error("use at least three rounds and one replay call")
    profile(options)
