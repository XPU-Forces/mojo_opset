import ast
from pathlib import Path


_NPU_KERNEL_ROOT = Path(__file__).parents[2] / "backends" / "ttx" / "kernels" / "npu"
_TRAINING_CONFIGS = {
    "_swa_fwd_kernel": {"BLOCK_M": 128, "BLOCK_N": 128, "multibuffer": True},
    "_swa_bwd_preprocess": {"BLOCK_SIZE": 64},
    "_swa_bwd_dkdv_kernel": {"BLOCK_M": 256, "BLOCK_N": 64, "multibuffer": True},
    "_swa_bwd_dq_kernel": {"BLOCK_M": 128, "BLOCK_N": 128, "multibuffer": True},
}
_RUNTIME_SHAPE_ARGUMENTS = {
    "_swa_fwd_kernel": ["bsz", "stride_lse_h"],
    "_swa_bwd_preprocess": ["num_tokens", "d_stride_h"],
    "_swa_bwd_dkdv_kernel": ["bsz", "stride_delta_h", "stride_lse_h"],
    "_swa_bwd_dq_kernel": ["bsz", "stride_delta_h", "stride_lse_h"],
}


def _functions(arch: str) -> dict[str, ast.FunctionDef]:
    source = (_NPU_KERNEL_ROOT / arch / "swa.py").read_text()
    tree = ast.parse(source)
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _decorator_call(node: ast.FunctionDef, name: str) -> ast.Call:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr == name:
                return decorator
    raise AssertionError(f"{node.name} is missing @{name}")


def _single_triton_config(node: ast.FunctionDef) -> dict[str, object]:
    autotune = _decorator_call(node, "autotune")
    configs = next(keyword.value for keyword in autotune.keywords if keyword.arg == "configs")
    assert isinstance(configs, ast.List)
    assert len(configs.elts) == 1

    config = configs.elts[0]
    assert isinstance(config, ast.Call)
    if config.args:
        return ast.literal_eval(config.args[0])

    kwargs = next(keyword.value for keyword in config.keywords if keyword.arg == "kwargs")
    return ast.literal_eval(kwargs)


def _do_not_specialize(node: ast.FunctionDef) -> list[str]:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Attribute) and decorator.attr == "jit":
            return []
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr == "jit":
                values = [keyword.value for keyword in decorator.keywords if keyword.arg == "do_not_specialize"]
                return ast.literal_eval(values[0]) if values else []
    raise AssertionError(f"{node.name} is missing @jit")


def test_npu_swa_training_configs_are_pinned():
    for arch in ("a2", "a5"):
        functions = _functions(arch)
        for kernel, expected in _TRAINING_CONFIGS.items():
            assert _single_triton_config(functions[kernel]) == expected


def test_only_a2_swa_uses_runtime_sequence_shape():
    a2_functions = _functions("a2")
    a5_functions = _functions("a5")

    for kernel, expected in _RUNTIME_SHAPE_ARGUMENTS.items():
        assert _do_not_specialize(a2_functions[kernel]) == expected
        assert _do_not_specialize(a5_functions[kernel]) == []


def test_npu_swa_mask_cache_is_bounded_per_device():
    for arch in ("a2", "a5"):
        functions = _functions(arch)
        cached = functions["_get_mask_causal_with_window_cached"]
        lru_cache = _decorator_call(cached, "lru_cache")
        maxsize = next(keyword.value for keyword in lru_cache.keywords if keyword.arg == "maxsize")
        assert ast.literal_eval(maxsize) == 32

        wrapper_source = ast.unparse(functions["get_mask_causal_with_window"])
        assert "torch.npu.current_device()" in wrapper_source
        assert "str(target)" in wrapper_source
