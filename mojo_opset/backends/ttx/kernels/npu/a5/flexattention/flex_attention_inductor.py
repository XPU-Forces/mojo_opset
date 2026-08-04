import importlib.util
import re
from pathlib import Path

import torch
from torch_npu import _inductor  # noqa: F401
import torch.nn.attention.flex_attention as flex_attention_module
from torch.nn.attention.flex_attention import create_block_mask as create_block_mask_native

from perf_utils import profile_npu_op
from perf_utils import profile_phase_guard
from perf_utils import sync_device
from perf_utils import timed_call


def build_native_block_mask(mask_func, problem):
    mask_mod = mask_func(problem)
    seq = problem["total_s"]
    return create_block_mask_native(mask_mod, 1, 1, seq, seq, device=problem["q"].device)


def run_without_flex_attention_device_check(fn):
    old_validate_device = flex_attention_module._validate_device
    flex_attention_module._validate_device = lambda *args, **kwargs: None
    try:
        return fn()
    finally:
        flex_attention_module._validate_device = old_validate_device


def make_inductor_flex_attention_runner(block_mask, allow_dynamo_fallback=False):
    try:
        torch._dynamo.config.suppress_errors = allow_dynamo_fallback
    except (AttributeError, NameError):
        pass

    def _native_flex_attention(q, k, v):
        return flex_attention_module.flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=True,
            return_lse=False,
        )

    compiled_flex_attention = torch.compile(_native_flex_attention, backend="inductor")

    def _run(q, k, v):
        return run_without_flex_attention_device_check(lambda: compiled_flex_attention(q, k, v))

    return _run


def discover_inductor_output_code_dir(output_code_dir):
    root = Path(output_code_dir).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"INDUCTOR_OUTPUT_CODE_DIR is not a directory: {root}")

    stage_patterns = {
        "forward": "_forward_",
        "inference": "_inference_",
        "backward": "_backward_",
    }
    found = {}
    for stage, token in stage_patterns.items():
        candidates = sorted(
            path / "output_code.py"
            for path in root.iterdir()
            if path.is_dir() and token in path.name and (path / "output_code.py").is_file()
        )
        if not candidates:
            raise RuntimeError(f"INDUCTOR_OUTPUT_CODE_DIR missing *{token}*/output_code.py: {root}")
        if len(candidates) > 1:
            raise RuntimeError(f"INDUCTOR_OUTPUT_CODE_DIR has multiple {stage} output_code.py: {candidates}")
        found[stage] = candidates[0]
    return found


def write_direct_output_code_copy(output_code_path):
    output_code_path = Path(output_code_path)
    direct_dir = output_code_path.parents[1] / ".direct_output_code"
    direct_dir.mkdir(exist_ok=True)
    direct_path = direct_dir / f"{output_code_path.parent.name}_output_code_direct.py"

    lines = output_code_path.read_text().splitlines(keepends=True)
    rewritten = []
    in_call = False
    for line in lines:
        stripped = line.strip()
        if line.startswith("def call(args):"):
            in_call = True
            rewritten.append(line)
            continue
        if in_call and (
            stripped.startswith("args_path = ")
            or stripped.startswith("os.makedirs(os.path.dirname(args_path)")
            or stripped.startswith("torch.save(args, args_path)")
        ):
            continue
        if in_call and stripped.startswith("args.clear()"):
            in_call = False
        rewritten.append(line)

    direct_path.write_text("".join(rewritten))
    return direct_path


def load_output_code_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load output_code module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_inductor_output_code_modules(output_code_dir):
    paths = discover_inductor_output_code_dir(output_code_dir)
    direct_paths = {stage: write_direct_output_code_copy(path) for stage, path in paths.items()}
    modules = {
        stage: load_output_code_module(path, f"direct_{stage}_{abs(hash(str(path)))}")
        for stage, path in direct_paths.items()
    }
    return paths, direct_paths, modules


def count_output_code_call_args(path, stage):
    text = Path(path).read_text()
    match = re.search(r"^\s*([A-Za-z_]\w*(?:,\s*[A-Za-z_]\w*)*)\s*=\s*args\s*$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"unable to parse {stage} output_code call(args) unpacking: {path}")
    return len([name.strip() for name in match.group(1).split(",")])


def validate_output_code_arg_count(stage, path, expected_count, arg_names, mask_type):
    actual_count = count_output_code_call_args(path, stage)
    if actual_count != expected_count:
        raise RuntimeError(
            f"{stage} output_code arg count mismatch: mask={mask_type}, path={path}, "
            f"generated_expected={actual_count}, test_constructed={expected_count}, args={arg_names}"
        )


def split_output_code_forward_result(ctx, fwd_ret):
    num_forward_args = len(ctx["forward_args"])
    lse_idx = 2 + num_forward_args
    if len(fwd_ret) <= lse_idx:
        raise RuntimeError(
            f"forward output_code returned too few values: returned={len(fwd_ret)}, "
            f"forward_args={num_forward_args}, need_at_least={lse_idx + 1}"
        )
    raw_out = fwd_ret[0]
    saved_primals = list(fwd_ret[1:1 + num_forward_args])
    saved_raw_out = fwd_ret[1 + num_forward_args]
    lse = fwd_ret[lse_idx]
    return raw_out, saved_raw_out, lse, saved_primals


def output_code_mask_input_items(mask_type, problem):
    if mask_type == "sparse_doc_image":
        return [
            ("doc_start", problem["doc_start"].to(torch.float32)),
            ("modality", problem["modality"].to(torch.float32)),
            ("segment_ids", problem["segment_ids"].to(torch.float32)),
        ]
    if mask_type == "causal":
        return []
    raise RuntimeError(f"unsupported output_code mask type: {mask_type}")


def build_inductor_output_code_context(output_code_dir, mask_func, problem, mask_func_to_type):
    paths, direct_paths, modules = prepare_inductor_output_code_modules(output_code_dir)
    block_mask = build_native_block_mask(mask_func, problem)
    meta = getattr(block_mask, "_npu_compact_sparse_mask_metadata", None)
    if meta is None:
        raise RuntimeError("block_mask missing _npu_compact_sparse_mask_metadata")
    for key in ("q_offsets", "flat_to_row", "flat_to_blk"):
        if key not in meta:
            raise RuntimeError(f"compact sparse metadata missing {key}")

    mask_type = mask_func_to_type[id(mask_func)]
    mask_input_items = output_code_mask_input_items(mask_type, problem)

    forward_args = [
        problem["q"].detach(),
        problem["k"].detach(),
        problem["v"].detach(),
        block_mask.kv_indices.to(torch.int32),
        block_mask.kv_num_blocks.to(torch.int32),
        meta["flat_to_blk"].to(torch.int32),
        meta["flat_to_row"].to(torch.int32),
        meta["q_offsets"].to(torch.int32),
        *[value for _, value in mask_input_items],
        block_mask.full_kv_num_blocks.to(torch.int32),
        block_mask.full_kv_indices.to(torch.int32),
        block_mask.q_num_blocks.to(torch.int32),
        block_mask.q_indices.to(torch.int32),
        block_mask.full_q_num_blocks.to(torch.int32),
        block_mask.full_q_indices.to(torch.int32),
    ]
    forward_arg_names = [
        "q",
        "k",
        "v",
        "kv_indices",
        "kv_num_blocks",
        "flat_to_blk",
        "flat_to_row",
        "q_offsets",
        *[name for name, _ in mask_input_items],
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    ]
    validate_output_code_arg_count(
        "forward", direct_paths["forward"], len(forward_args), forward_arg_names, mask_type
    )
    validate_output_code_arg_count(
        "backward",
        direct_paths["backward"],
        len(forward_args) + 3,
        forward_arg_names + ["raw_out", "lse", "grad_raw"],
        mask_type,
    )
    return {
        "output_code_dir": str(Path(output_code_dir).expanduser().resolve()),
        "paths": paths,
        "direct_paths": direct_paths,
        "modules": modules,
        "block_mask": block_mask,
        "forward_args": forward_args,
        "forward_arg_names": forward_arg_names,
        "mask_type": mask_type,
    }


def run_inductor_output_code_once(ctx, return_grid):
    fwd_ret = ctx["modules"]["forward"].call(list(ctx["forward_args"]))
    raw_out, saved_raw_out, lse, saved_primals = split_output_code_forward_result(ctx, fwd_ret)
    final_out = ctx["modules"]["inference"].call([raw_out])[0]
    grad_raw = torch.ones_like(raw_out) * (return_grid / raw_out.numel())
    bwd_ret = ctx["modules"]["backward"].call(saved_primals + [saved_raw_out, lse, grad_raw])
    return fwd_ret, raw_out, lse, final_out, grad_raw, bwd_ret


def profile_output_code(ctx, label, return_grid, prof_root):
    mask_name = ctx["mask_type"]

    def _forward_step():
        fwd_ret = ctx["modules"]["forward"].call(list(ctx["forward_args"]))
        raw_out, saved_raw_out, lse, saved_primals = split_output_code_forward_result(ctx, fwd_ret)
        final_out = ctx["modules"]["inference"].call([raw_out])[0]
        sync_device()
        del fwd_ret, raw_out, saved_raw_out, lse, saved_primals, final_out

    forward_result = profile_phase_guard(
        label,
        "forward",
        lambda: profile_npu_op(label, mask_name, "forward", _forward_step, prof_root),
    )

    def _profile_backward():
        fwd_ret = ctx["modules"]["forward"].call(list(ctx["forward_args"]))
        raw_out, saved_raw_out, lse, saved_primals = split_output_code_forward_result(ctx, fwd_ret)
        grad_raw = torch.ones_like(raw_out) * (return_grid / raw_out.numel())
        sync_device()

        def _backward_step():
            bwd_ret = ctx["modules"]["backward"].call(saved_primals + [saved_raw_out, lse, grad_raw])
            sync_device()
            del bwd_ret

        try:
            return profile_npu_op(label, mask_name, "backward", _backward_step, prof_root)
        finally:
            del fwd_ret, raw_out, saved_raw_out, lse, saved_primals, grad_raw
            torch.npu.empty_cache()

    backward_result = profile_phase_guard(label, "backward", _profile_backward)
    return {"forward": forward_result, "backward": backward_result}


def benchmark_inductor_runner(label, runner, q, k, v, return_tensor):
    out, fwd_ms, fwd_mem = timed_call(lambda: runner(q, k, v))

    def _backward():
        if tuple(out.shape) != tuple(return_tensor.shape):
            raise RuntimeError(
                f"return_tensor shape {tuple(return_tensor.shape)} does not match output shape {tuple(out.shape)}"
            )
        out.backward(return_tensor)

    _, bwd_ms, bwd_mem = timed_call(_backward)
    return {
        "label": label,
        "output": out,
        "fwd_ms": fwd_ms,
        "fwd_mem_mb": fwd_mem,
        "bwd_ms": bwd_ms,
        "bwd_mem_mb": bwd_mem,
    }


__all__ = [
    "benchmark_inductor_runner",
    "build_inductor_output_code_context",
    "build_native_block_mask",
    "make_inductor_flex_attention_runner",
    "profile_output_code",
    "run_inductor_output_code_once",
]
