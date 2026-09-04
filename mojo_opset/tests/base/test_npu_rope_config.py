import ast
from pathlib import Path


_NPU_KERNEL_ROOT = Path(__file__).parents[2] / "backends" / "ttx" / "kernels" / "npu"


def _rope_kernel(arch: str, name: str) -> ast.FunctionDef:
    source = (_NPU_KERNEL_ROOT / arch / "rope.py").read_text()
    tree = ast.parse(source)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _do_not_specialize(node: ast.FunctionDef) -> list[str]:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr == "jit":
                values = [
                    keyword.value
                    for keyword in decorator.keywords
                    if keyword.arg == "do_not_specialize"
                ]
                return ast.literal_eval(values[0]) if values else []
    raise AssertionError(f"{node.name} is missing @jit")


def test_a2_rope_uses_runtime_sequence_shape():
    assert _do_not_specialize(_rope_kernel("a2", "_rope_inplace_kernel")) == [
        "seq_len",
        "num_seq_blocks",
        "bs",
    ]
    assert _do_not_specialize(_rope_kernel("a5", "_rope_kernel")) == ["seq_len"]
