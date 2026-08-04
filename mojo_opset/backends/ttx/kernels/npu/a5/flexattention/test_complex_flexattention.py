import argparse
import ast
import copy
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

try:
    torch._functorch.config.donated_buffer = False
except (AttributeError, NameError):
    pass

import flex_attention_triton as triton_backend
from flex_attention_inductor import build_native_block_mask
from flex_attention_inductor import make_inductor_flex_attention_runner
from flex_attention_triton import create_block_mask_patched
from flex_attention_triton import make_sdpa_reference_runner
from flex_attention_triton import make_triton_flex_attention_runner
from perf_utils import clear_profile_grads
from perf_utils import current_memory_mb
from perf_utils import empty_cache
from perf_utils import profile_autograd_op
from perf_utils import sync_device
from perf_utils import timed_call

try:
    from torch.nn.attention import flex_attention as _fa_module

    _fa_module._validate_device = lambda *args, **kwargs: None
except Exception:
    pass


DTYPE = torch.bfloat16
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
SLIDING_WINDOW = 1024
GLOBAL_WINDOW = 4
DATA_LENGTH = [[128, 128]]
DATA_INPUT_TYPE = [["text", "image_gen"]]
DATA_LENGTH_VIDEO = [[128, 128]]
VIDEO_FRAME_LENGTH = [[[128], [128]]]
RETURN_SCALE = 1000.0
RETURN_TENSOR = None
PARENT_MASK_NAMES = None
MASK_ALIASES = {
    "sparse_mask_mod": "sparse",
    "full_mask_mod": "full",
}

NPU_PROF_DIR = os.environ.get("NPU_PROF_DIR", "./prof_dir")
PARENT_MASK_MODULE = None
PARENT_CONFIG_INITIALIZED = False
PARENT_CONFIG_NAMES = (
    "DTYPE",
    "HEAD_DIM",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "SLIDING_WINDOW",
    "GLOBAL_WINDOW",
    "DATA_LENGTH",
    "DATA_INPUT_TYPE",
    "DATA_LENGTH_VIDEO",
    "VIDEO_FRAME_LENGTH",
)


def _copy_parent_config_value(value):
    if isinstance(value, (dict, list, tuple)):
        return copy.deepcopy(value)
    return value


def _apply_parent_config_defaults(parent):
    for name in PARENT_CONFIG_NAMES:
        if hasattr(parent, name):
            globals()[name] = _copy_parent_config_value(getattr(parent, name))


def _install_parent_test_stubs():
    if "mojo_opset" not in sys.modules:
        package = types.ModuleType("mojo_opset")
        package.__path__ = []
        sys.modules["mojo_opset"] = package

    experimental_name = "mojo_opset.experimental"
    if experimental_name not in sys.modules:
        experimental = types.ModuleType(experimental_name)

        def _missing_mojo_flex_attention(*args, **kwargs):
            raise RuntimeError("mojo_flex_attention is not used by this flexattention development test")

        experimental.mojo_flex_attention = _missing_mojo_flex_attention
        sys.modules[experimental_name] = experimental

    tests_utils_name = "mojo_opset.tests.utils"
    if "mojo_opset.tests" not in sys.modules:
        tests_pkg = types.ModuleType("mojo_opset.tests")
        tests_pkg.__path__ = []
        sys.modules["mojo_opset.tests"] = tests_pkg
    if tests_utils_name not in sys.modules:
        tests_utils = types.ModuleType(tests_utils_name)

        def _identity_decorator(fn):
            return fn

        tests_utils.bypass_not_implemented = _identity_decorator
        sys.modules[tests_utils_name] = tests_utils

    platform_name = "mojo_opset.utils.platform"
    if "mojo_opset.utils" not in sys.modules:
        utils_pkg = types.ModuleType("mojo_opset.utils")
        utils_pkg.__path__ = []
        sys.modules["mojo_opset.utils"] = utils_pkg
    if platform_name not in sys.modules:
        platform_module = types.ModuleType(platform_name)
        platform_module.get_platform = lambda: "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
        platform_module.get_torch_device = (
            lambda: torch.device("npu") if hasattr(torch, "npu") and torch.npu.is_available() else torch.device("cpu")
        )
        sys.modules[platform_name] = platform_module

    kernel_name = "mojo_opset.backends.ttx.kernels.npu.flex_attention"
    if kernel_name not in sys.modules:
        parent_flex = triton_backend._parent_flex_attention()
        sys.modules[kernel_name] = parent_flex
        for package_name in (
            "mojo_opset.backends",
            "mojo_opset.backends.ttx",
            "mojo_opset.backends.ttx.kernels",
            "mojo_opset.backends.ttx.kernels.npu",
        ):
            if package_name not in sys.modules:
                package = types.ModuleType(package_name)
                package.__path__ = []
                sys.modules[package_name] = package


def load_parent_mask_module():
    global PARENT_MASK_MODULE
    if PARENT_MASK_MODULE is not None:
        return PARENT_MASK_MODULE

    _install_parent_test_stubs()
    module_path = (
        Path(__file__).resolve().parents[6]
        / "tests"
        / "accuracy"
        / "functions"
        / "test_flex_attention.py"
    )
    spec = importlib.util.spec_from_file_location("_parent_flex_attention_cases", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load parent flex attention cases: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PARENT_MASK_MODULE = module
    return module


def initialize_parent_config_defaults():
    global PARENT_CONFIG_INITIALIZED
    if PARENT_CONFIG_INITIALIZED:
        return
    _apply_parent_config_defaults(load_parent_mask_module())
    PARENT_CONFIG_INITIALIZED = True


def parse_npu_prof_modes(raw_mode):
    if not raw_mode:
        return set()
    modes = set()
    for mode in raw_mode.split(","):
        mode = mode.strip()
        if mode:
            modes.add(mode)
    return modes


def npu_prof_mode_enabled(mode, selected_modes):
    return "all" in selected_modes or mode in selected_modes


def reset_compiler_state():
    try:
        torch.compiler.reset()
    except (AttributeError, RuntimeError):
        pass
    try:
        torch._dynamo.reset()
    except (AttributeError, RuntimeError):
        pass


def get_device(device_name):
    if device_name == "npu":
        return torch.device("npu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def total_seq_len_from_data_length(data_length):
    return sum(sum(sample) for sample in data_length)


def parse_data_length(raw):
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"--data-length must be a Python-style nested list: {raw!r}") from exc

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("--data-length must contain at least one sample")

    parsed = []
    for sample in value:
        if not isinstance(sample, (list, tuple)) or not sample:
            raise ValueError("--data-length samples must be non-empty lists")
        parsed_sample = []
        for item in sample:
            if not isinstance(item, int) or item <= 0:
                raise ValueError(f"--data-length entries must be positive ints: {item!r}")
            parsed_sample.append(int(item))
        parsed.append(parsed_sample)
    return parsed


def parse_data_input_type(raw):
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"--data-input-type must be a Python-style nested list: {raw!r}") from exc

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("--data-input-type must contain at least one sample")

    parsed = []
    for sample in value:
        if not isinstance(sample, (list, tuple)) or not sample:
            raise ValueError("--data-input-type samples must be non-empty lists")
        parsed_sample = []
        for item in sample:
            if not isinstance(item, str) or not item:
                raise ValueError(f"--data-input-type entries must be non-empty strings: {item!r}")
            parsed_sample.append(item)
        parsed.append(parsed_sample)
    return parsed


def default_data_input_type(data_length):
    data_input_type = []
    for sample in data_length:
        if len(sample) == 1:
            data_input_type.append(["text"])
        elif len(sample) == 2:
            data_input_type.append(["text", "image_gen"])
        else:
            data_input_type.append(["text", *(["image_gen"] * (len(sample) - 2)), "text"])
    return data_input_type


def configure_data_layout(data_length_raw, data_input_type_raw=None):
    global DATA_LENGTH, DATA_INPUT_TYPE

    if data_length_raw is None:
        return

    data_length = parse_data_length(data_length_raw)
    data_input_type = (
        parse_data_input_type(data_input_type_raw)
        if data_input_type_raw is not None
        else default_data_input_type(data_length)
    )

    if len(data_length) != len(data_input_type):
        raise ValueError(
            f"DATA_LENGTH sample count {len(data_length)} does not match "
            f"DATA_INPUT_TYPE sample count {len(data_input_type)}"
        )
    for sample_lens, sample_types in zip(data_length, data_input_type):
        if len(sample_lens) != len(sample_types):
            raise ValueError(
                f"DATA_LENGTH sample {sample_lens} does not match DATA_INPUT_TYPE sample {sample_types}"
            )

    DATA_LENGTH = data_length
    DATA_INPUT_TYPE = data_input_type


def build_parent_problem(mask_func, device, seed=23):
    initialize_parent_config_defaults()
    parent = load_parent_mask_module()
    old_values = {
        "DTYPE": parent.DTYPE,
        "HEAD_DIM": parent.HEAD_DIM,
        "NUM_Q_HEADS": parent.NUM_Q_HEADS,
        "NUM_KV_HEADS": parent.NUM_KV_HEADS,
        "SLIDING_WINDOW": parent.SLIDING_WINDOW,
        "GLOBAL_WINDOW": parent.GLOBAL_WINDOW,
        "DATA_LENGTH": parent.DATA_LENGTH,
        "DATA_INPUT_TYPE": parent.DATA_INPUT_TYPE,
        "DATA_LENGTH_VIDEO": parent.DATA_LENGTH_VIDEO,
        "VIDEO_FRAME_LENGTH": parent.VIDEO_FRAME_LENGTH,
        "SEED": parent.SEED,
    }
    try:
        parent.DTYPE = DTYPE
        parent.HEAD_DIM = HEAD_DIM
        parent.NUM_Q_HEADS = NUM_Q_HEADS
        parent.NUM_KV_HEADS = NUM_KV_HEADS
        parent.SLIDING_WINDOW = SLIDING_WINDOW
        parent.GLOBAL_WINDOW = GLOBAL_WINDOW
        parent.DATA_LENGTH = DATA_LENGTH
        parent.DATA_INPUT_TYPE = DATA_INPUT_TYPE
        parent.DATA_LENGTH_VIDEO = DATA_LENGTH_VIDEO
        parent.VIDEO_FRAME_LENGTH = VIDEO_FRAME_LENGTH
        parent.SEED = seed
        problem = parent.build_problem(mask_func)
    finally:
        for name, value in old_values.items():
            setattr(parent, name, value)
    problem["q"] = problem["q"].to(device=device, dtype=DTYPE)
    problem["k"] = problem["k"].to(device=device, dtype=DTYPE)
    problem["v"] = problem["v"].to(device=device, dtype=DTYPE)
    return problem


def build_problem_from_parent(mask_func, device):
    return build_parent_problem(mask_func, device)


def make_return_tensor(problem):
    global RETURN_TENSOR

    expected_shape = problem["q"].shape
    expected_numel = int(problem["q"].numel())
    if RETURN_TENSOR is None or tuple(RETURN_TENSOR.shape) != tuple(expected_shape):
        RETURN_TENSOR = torch.full(
            expected_shape,
            RETURN_SCALE / expected_numel,
            dtype=DTYPE,
            device=problem["q"].device,
        )
        return RETURN_TENSOR

    if RETURN_TENSOR.device != problem["q"].device or RETURN_TENSOR.dtype != DTYPE:
        RETURN_TENSOR = RETURN_TENSOR.to(device=problem["q"].device, dtype=DTYPE)
    return RETURN_TENSOR


def get_mask_cases():
    initialize_parent_config_defaults()
    parent = load_parent_mask_module()
    cases = {}
    for mask_name, mask_func in parent._MASK_FUNCS:
        if PARENT_MASK_NAMES is not None and mask_name not in PARENT_MASK_NAMES:
            continue
        cases[mask_name] = {
            "build_problem": build_problem_from_parent,
            "mask_mod": mask_func,
            "dense_mask": lambda problem, fn=mask_func: parent._build_dense_mask(fn, problem)[0, 0],
        }
    return cases


def clone_for_grad(problem):
    q = problem["q"].detach().clone().requires_grad_(True)
    k = problem["k"].detach().clone().requires_grad_(True)
    v = problem["v"].detach().clone().requires_grad_(True)
    return q, k, v


def max_abs_diff(actual, expected):
    return (actual.detach().float().cpu() - expected.detach().float().cpu()).abs().max().item()


def assert_close(name, actual, expected, atol, rtol):
    diff = max_abs_diff(actual, expected)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=rtol)
    print(f"  [precision] {name}: PASS max_abs_diff={diff:.6f}")
    return diff


def build_runners(mask_name, problem, target, allow_dynamo_fallback=False, include_reference=True):
    case = get_mask_cases()[mask_name]
    seq_len = problem["total_s"]
    mask_mod = case["mask_mod"](problem)
    runners = {}
    if include_reference:
        dense_mask = case["dense_mask"](problem)
        runners["dense_mask"] = dense_mask
        runners["sdpa"] = make_sdpa_reference_runner(dense_mask)
    if target in ("all", "triton"):
        triton_block_mask = create_block_mask_patched(mask_mod, seq_len, device=problem["q"].device)
        runners["triton"] = make_triton_flex_attention_runner(triton_block_mask)
    if target in ("all", "inductor"):
        native_block_mask = build_native_block_mask(case["mask_mod"], problem)
        runners["inductor"] = make_inductor_flex_attention_runner(
            native_block_mask,
            allow_dynamo_fallback=allow_dynamo_fallback,
        )
    return runners


def run_path(path_name, runner, problem, return_tensor):
    q, k, v = clone_for_grad(problem)
    out, fwd_ms, fwd_mem = timed_call(lambda: runner(q, k, v))

    def _backward():
        if tuple(out.shape) != tuple(return_tensor.shape):
            raise RuntimeError(
                f"RETURN_TENSOR shape {tuple(return_tensor.shape)} does not match output shape {tuple(out.shape)}"
            )
        out.backward(return_tensor)

    _, bwd_ms, bwd_mem = timed_call(_backward)
    sync_device()
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "fwd_ms": fwd_ms,
        "fwd_mem_mb": fwd_mem,
        "bwd_ms": bwd_ms,
        "bwd_mem_mb": bwd_mem,
        "label": path_name,
    }


def benchmark_path(path_name, runner, problem, return_tensor, warmup, iters):
    q, k, v = clone_for_grad(problem)

    def _backward(out):
        if tuple(out.shape) != tuple(return_tensor.shape):
            raise RuntimeError(
                f"RETURN_TENSOR shape {tuple(return_tensor.shape)} does not match output shape {tuple(out.shape)}"
            )
        out.backward(return_tensor)

    for _ in range(warmup):
        clear_profile_grads(q, k, v)
        out = runner(q, k, v)
        _backward(out)
        sync_device()
        del out

    fwd_times = []
    bwd_times = []
    fwd_mems = []
    bwd_mems = []
    for _ in range(iters):
        clear_profile_grads(q, k, v)
        out, fwd_ms, fwd_mem = timed_call(lambda: runner(q, k, v))
        _, bwd_ms, bwd_mem = timed_call(lambda: _backward(out))
        fwd_times.append(fwd_ms)
        bwd_times.append(bwd_ms)
        fwd_mems.append(fwd_mem)
        bwd_mems.append(bwd_mem)
        sync_device()
        del out

    clear_profile_grads(q, k, v)
    empty_cache()
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": None,
        "fwd_ms": sum(fwd_times) / len(fwd_times),
        "fwd_mem_mb": max(fwd_mems),
        "bwd_ms": sum(bwd_times) / len(bwd_times),
        "bwd_mem_mb": max(bwd_mems),
        "fwd_min_ms": min(fwd_times),
        "bwd_min_ms": min(bwd_times),
        "iters": iters,
        "warmup": warmup,
        "label": path_name,
    }


def compare_result(label, result, ref_result, args):
    precision = {
        "forward": assert_close(
            f"{label}.forward",
            result["out"],
            ref_result["out"],
            args.forward_atol,
            args.forward_rtol,
        ),
        "q_grad": assert_close(
            f"{label}.q_grad",
            result["q"].grad,
            ref_result["q"].grad,
            args.grad_atol,
            args.grad_rtol,
        ),
        "k_grad": assert_close(
            f"{label}.k_grad",
            result["k"].grad,
            ref_result["k"].grad,
            args.grad_atol,
            args.grad_rtol,
        ),
        "v_grad": assert_close(
            f"{label}.v_grad",
            result["v"].grad,
            ref_result["v"].grad,
            args.grad_atol,
            args.grad_rtol,
        ),
    }
    return precision


def print_perf(label, result):
    suffix = ""
    if "fwd_min_ms" in result:
        suffix = (
            f", min_fwd: {result['fwd_min_ms']:.2f}ms, "
            f"min_bwd: {result['bwd_min_ms']:.2f}ms, "
            f"warmup={result['warmup']}, iters={result['iters']}"
        )
    print(
        f"[{label}] fwd: {result['fwd_ms']:.2f}ms({result['fwd_mem_mb']:.1f}MB), "
        f"bwd: {result['bwd_ms']:.2f}ms({result['bwd_mem_mb']:.1f}MB){suffix}"
    )


def run_case(mask_name, args):
    case = get_mask_cases()[mask_name]
    device = get_device(args.device)
    problem = case["build_problem"](case["mask_mod"], device)
    problem["mask_name"] = mask_name
    return_tensor = make_return_tensor(problem)
    runners = build_runners(
        mask_name,
        problem,
        args.target,
        allow_dynamo_fallback=args.allow_dynamo_fallback,
        include_reference=not args.perf_only,
    )

    print(f"\n=== mask={mask_name} total_s={problem['total_s']} data_length={DATA_LENGTH} ===")
    print(f"data_input_type={DATA_INPUT_TYPE}")
    print(f"memory_before={current_memory_mb():.1f}MB")

    results = {}
    ref_result = None
    if not args.perf_only:
        ref_result = run_path("sdpa", runners["sdpa"], problem, return_tensor)
        print_perf("sdpa", ref_result)
        results["sdpa"] = ref_result

    if args.target in ("all", "triton"):
        triton_result = (
            benchmark_path("triton", runners["triton"], problem, return_tensor, args.warmup, args.iters)
            if args.perf_only
            else run_path("triton", runners["triton"], problem, return_tensor)
        )
        print_perf("triton", triton_result)
        if ref_result is not None:
            compare_result("triton", triton_result, ref_result, args)
        results["triton"] = triton_result

    if args.target in ("all", "inductor"):
        reset_compiler_state()
        inductor_result = (
            benchmark_path("inductor", runners["inductor"], problem, return_tensor, args.warmup, args.iters)
            if args.perf_only
            else run_path("inductor", runners["inductor"], problem, return_tensor)
        )
        print_perf("inductor", inductor_result)
        if ref_result is not None:
            compare_result("inductor", inductor_result, ref_result, args)
        results["inductor"] = inductor_result
        reset_compiler_state()

    if args.enable_profiler:
        if "triton" in results:
            q, k, v = clone_for_grad(problem)
            profile_autograd_op(
                "triton",
                mask_name,
                runners["triton"],
                q,
                k,
                v,
                return_tensor,
                args.prof_dir,
                enable_profiler=True,
            )
        if "inductor" in results:
            q, k, v = clone_for_grad(problem)
            profile_autograd_op(
                "inductor",
                mask_name,
                runners["inductor"],
                q,
                k,
                v,
                return_tensor,
                args.prof_dir,
                enable_profiler=True,
            )

    for result in results.values():
        clear_profile_grads(result["q"], result["k"], result["v"])
    del results, runners, problem
    empty_cache()
    print(f"[PASS] mask={mask_name}")


def parse_args(argv):
    mask_cases = get_mask_cases()
    parser = argparse.ArgumentParser(description="FlexAttention Triton/Inductor development cases")
    parser.add_argument("mask", nargs="?", default="all", choices=["all", *mask_cases.keys(), *MASK_ALIASES.keys()])
    parser.add_argument("prof_mode", nargs="?", default=os.environ.get("NPU_PROF_MODE", ""))
    parser.add_argument("--target", choices=["all", "triton", "inductor"], default="all")
    parser.add_argument("--device", choices=["npu", "cuda", "cpu"], default="npu")
    parser.add_argument(
        "--data-length",
        default=None,
        help='Python-style nested list, e.g. "[[90, 800, 110]]"',
    )
    parser.add_argument(
        "--data-input-type",
        default=None,
        help='Python-style nested list matching --data-length, e.g. "[[\'text\', \'image_gen\', \'text\']]"',
    )
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="Benchmark selected target paths without SDPA reference correctness comparison.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--prof-dir", default=NPU_PROF_DIR)
    parser.add_argument("--allow-dynamo-fallback", action="store_true")
    parser.add_argument("--forward-atol", type=float, default=5e-2)
    parser.add_argument("--forward-rtol", type=float, default=5e-2)
    parser.add_argument("--grad-atol", type=float, default=8e-2)
    parser.add_argument("--grad-rtol", type=float, default=8e-2)
    args = parser.parse_args(argv)
    args.mask = MASK_ALIASES.get(args.mask, args.mask)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    try:
        configure_data_layout(args.data_length, args.data_input_type)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    prof_modes = parse_npu_prof_modes(args.prof_mode)
    args.enable_profiler = npu_prof_mode_enabled("profile", prof_modes)

    selected = list(get_mask_cases()) if args.mask == "all" else [args.mask]
    for mask_name in selected:
        run_case(mask_name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
