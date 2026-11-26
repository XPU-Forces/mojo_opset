"""Apply pinned-source patches and compare the actual before/after model methods.

This isolates the position-interpolation method. It does not load a full vLLM
model or claim end-to-end model inference coverage.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch

from mojo_opset.functions import triton_pos_embed_interpolate
from mojo_opset.kernels.torch.vision_pos_embed_interp import vision_pos_embed_interp_fwd


def method(source, name, cls=None, has_triton=True):
    tree = ast.parse(source)
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "torch": torch,
        "Sequence": Sequence,
        "HAS_TRITON": has_triton,
        "pos_embed_interpolate_native": vision_pos_embed_interp_fwd,
        "triton_pos_embed_interpolate": triton_pos_embed_interpolate,
    }
    exec(compile(module, "<pinned-vllm-method>", "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    patch_dir = Path(__file__).resolve().parent
    references = json.loads((patch_dir / "reference_sources.json").read_text())
    for name, source in references.items():
        digest = hashlib.sha256((args.reference_dir / name).read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise RuntimeError(f"wrong pinned source: {name}, SHA-256 {digest}")
    before = (args.reference_dir / "patch_qwen3vl.py").read_text()
    result = {
        "scope": "pinned-source patch and interpolation-method integration; full model not loaded",
        "reference_commits": {name: source["commit"] for name, source in references.items()},
        "source_sha256": {
            name: hashlib.sha256((args.reference_dir / name).read_bytes()).hexdigest()
            for name in ["qwen3_vl.py", "patch_qwen3vl.py"]
        },
        "cases": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = [
            ("qwen3_vl.py", "vllm/model_executor/models/qwen3_vl.py", "vllm-aa31fa0.patch"),
            ("patch_qwen3vl.py", "vllm_ascend/patch/worker/patch_qwen3vl.py", "vllm-ascend-d5923bc.patch"),
        ]
        for src, rel, patch in inputs:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((args.reference_dir / src).read_bytes())
            subprocess.run(["git", "apply", "--check", str(patch_dir / patch)], cwd=root, check=True)
            subprocess.run(["git", "apply", str(patch_dir / patch)], cwd=root, check=True)
            ast.parse(p.read_text())
        after = (root / inputs[0][1]).read_text()
        ascend_after = (root / inputs[1][1]).read_text()
        assert "Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate" not in ascend_after
        old = method(before, "_fast_pos_embed_interpolate")
        new = method(after, "fast_pos_embed_interpolate", "Qwen3_VisionTransformer")
        fallback = method(after, "fast_pos_embed_interpolate", "Qwen3_VisionTransformer", has_triton=False)
        for dtype in [torch.float32, torch.bfloat16]:
            torch.manual_seed(42)
            weight = torch.randn(48 * 48, 1152, dtype=dtype, device="npu") * 0.25
            model = SimpleNamespace(
                pos_embed=SimpleNamespace(weight=weight), num_grid_per_side=48, spatial_merge_size=2, dtype=dtype
            )
            grid = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
            expected = old(model, grid)
            actual = new(model, grid)
            torch.testing.assert_close(
                actual,
                expected,
                atol=5e-5 if dtype == torch.float32 else 1e-2,
                rtol=1e-5 if dtype == torch.float32 else 1e-2,
            )
            torch.testing.assert_close(fallback(model, grid), expected, atol=0, rtol=0)
            result["cases"].append(
                {
                    "dtype": str(dtype),
                    "shape": list(actual.shape),
                    "passed": True,
                    "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
                }
            )
    result["passed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
