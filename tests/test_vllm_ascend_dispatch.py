"""Tests for the vllm-ascend dispatch patch shipped as ``docs/vllm_ascend_pos_embed_dispatch.py``.

That file is a drop-in replacement for the ``fast_pos_embed_interpolate`` tail of
``vllm_ascend/patch/worker/patch_qwen3vl.py``, which currently forces the eager native path
on NPUs. Running it for real would need the whole vllm serving stack, so these tests stub the
three things it touches (``vllm``, ``torch_npu`` / ``torch.npu``, ``mojo_opset.functions``) and
assert the decision it makes in each case:

1. Triton unusable          -> the native per-image path runs, once per image;
2. Triton usable            -> the batched operator runs, exactly once, with the arguments it
   expects (including the tuple coercion);
3. Triton raises            -> the native path still runs, so a shape the kernel cannot handle
   costs speed, not correctness;
4. the probe is cached, because it would otherwise run on every forward.

Note on the device probe: the stub ``torch_npu`` module deliberately has **no** module-level
``is_available``. Importing ``torch_npu`` only registers the ``npu`` backend on ``torch``, so
the patch must call ``torch.npu.is_available()``; a regression to ``torch_npu.is_available()``
would raise ``AttributeError`` inside the probe and silently disable the Triton route, which
these tests would catch as "native ran where Triton was expected".

The bit-exactness of the batched operator against per-image calls is covered on hardware by
``tests/functions/test_pos_embed_interp.py::test_pos_embed_interp_multi_matches_cat``; this
file does not duplicate it.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

DISPATCH_MODULE = Path(__file__).resolve().parent.parent / "docs" / "vllm_ascend_pos_embed_dispatch.py"
DOC = Path(__file__).resolve().parent.parent / "docs" / "vllm_ascend_integration.md"
GRIDS = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
HIDDEN = 1152
NUM_GRID_PER_SIDE = 48
MERGE = 2


class _FakeNpu:
    """Stand-in for ``torch.npu``; only ``is_available`` is touched by the patch."""

    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _Self:
    """Minimal stand-in for ``Qwen3_VisionTransformer``."""

    def __init__(self, hidden=HIDDEN, grid=NUM_GRID_PER_SIDE, merge=MERGE, dtype=torch.bfloat16):
        self.pos_embed = torch.nn.Module()
        self.pos_embed.weight = torch.randn(grid * grid, hidden, dtype=dtype) * 0.25
        self.num_grid_per_side = grid
        self.spatial_merge_size = merge
        self.dtype = dtype


def _install_stubs(monkeypatch, has_triton, npu_ok, multi, native):
    """Stub vllm / torch_npu / torch.npu / mojo so the patch module can be executed standalone."""
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm.") or name == "torch_npu":
            monkeypatch.delitem(sys.modules, name, raising=False)

    vllm = types.ModuleType("vllm")
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.HAS_TRITON = has_triton
    model_exec = types.ModuleType("vllm.model_executor")
    models = types.ModuleType("vllm.model_executor.models")
    qwen3_vl = types.ModuleType("vllm.model_executor.models.qwen3_vl")
    qwen3_vl.pos_embed_interpolate_native = native
    qwen3_vl.Qwen3_VisionTransformer = type("Qwen3_VisionTransformer", (), {})
    for name, module in (
        ("vllm", vllm),
        ("vllm.triton_utils", triton_utils),
        ("vllm.model_executor", model_exec),
        ("vllm.model_executor.models", models),
        ("vllm.model_executor.models.qwen3_vl", qwen3_vl),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    # No module-level is_available on purpose: see the module docstring.
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "npu", _FakeNpu(npu_ok), raising=False)

    funcs = types.ModuleType("mojo_opset.functions")
    funcs.pos_embed_interp_multi_infer = multi
    funcs.pos_embed_interp_infer = lambda *a, **k: None
    mojo = types.ModuleType("mojo_opset")
    mojo.functions = funcs
    monkeypatch.setitem(sys.modules, "mojo_opset", mojo)
    monkeypatch.setitem(sys.modules, "mojo_opset.functions", funcs)

    return qwen3_vl.Qwen3_VisionTransformer


def _load_patch(monkeypatch):
    """Execute the patch module fresh so the probe cache starts empty."""
    name = "vllm_ascend_dispatch_under_test"
    monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(name, DISPATCH_MODULE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _run_case(monkeypatch, has_triton, npu_ok, multi_raises=False):
    """Drive one forward and return ``(output, multi_calls, native_calls, module)``."""
    multi_calls = []
    native_calls = []
    marker = torch.zeros(sum(t * h * w for t, h, w in GRIDS), HIDDEN, dtype=torch.bfloat16)

    def multi(*args, **kwargs):
        if multi_raises:
            raise RuntimeError("simulated kernel failure")
        multi_calls.append((args, kwargs))
        return marker

    def native(*args, **kwargs):
        native_calls.append((args, kwargs))
        return torch.zeros(args[1] * args[2] * args[3], HIDDEN, dtype=torch.bfloat16)

    _install_stubs(monkeypatch, has_triton, npu_ok, multi, native)
    module = _load_patch(monkeypatch)
    out = module._fast_pos_embed_interpolate(_Self(), GRIDS)
    return out, multi_calls, native_calls, module


def _expected_rows():
    return sum(t * h * w for t, h, w in GRIDS)


@pytest.mark.api("docs.vllm_ascend_pos_embed_dispatch")
@pytest.mark.parametrize(
    "has_triton, npu_ok, expect_multi",
    [(False, True, False), (True, False, False), (True, True, True)],
    ids=["no-triton", "no-npu", "both"],
)
def test_dispatch_picks_the_expected_path(monkeypatch, has_triton, npu_ok, expect_multi):
    out, multi_calls, native_calls, _module = _run_case(monkeypatch, has_triton, npu_ok)

    assert out.shape == (_expected_rows(), HIDDEN)
    if expect_multi:
        assert len(multi_calls) == 1, "the batched operator must be used exactly once"
        assert native_calls == []
    else:
        assert multi_calls == [], "Triton must not be used when it is unusable"
        assert len(native_calls) == len(GRIDS), "native must run once per image"


def test_dispatch_falls_back_when_the_kernel_raises(monkeypatch):
    out, multi_calls, native_calls, _module = _run_case(monkeypatch, True, True, multi_raises=True)

    assert out.shape == (_expected_rows(), HIDDEN)
    assert multi_calls == []
    assert len(native_calls) == len(GRIDS), "a kernel failure must cost speed, not correctness"


def test_dispatch_probe_is_cached(monkeypatch):
    _out, multi_calls, native_calls, module = _run_case(monkeypatch, True, True)
    first = (len(multi_calls), len(native_calls))
    assert first == (1, 0)

    # A second forward must reuse the cached probe and pick the same path. If the probe ran
    # again it would still succeed here, so also assert the cache flag itself.
    multi_calls.clear()
    native_calls.clear()
    module._fast_pos_embed_interpolate(_Self(), GRIDS)
    assert (len(multi_calls), len(native_calls)) == first
    assert module._TRITON_POS_EMBED_OK is True


def test_dispatch_coerces_the_grid_argument(monkeypatch):
    _out, multi_calls, _native_calls, _module = _run_case(monkeypatch, True, True)
    args, kwargs = multi_calls[0]

    grids_arg = args[1]
    assert isinstance(grids_arg, list)
    assert all(isinstance(g, tuple) and all(isinstance(v, int) for v in g) for g in grids_arg)
    assert grids_arg == [tuple(g) for g in GRIDS]
    assert args[2] == NUM_GRID_PER_SIDE
    assert args[3] == MERGE
    assert kwargs.get("implementation") == "triton"


def test_docs_snippet_uses_the_registered_npu_entry_point():
    """Guard the exact bug this patch shipped with: ``torch_npu.is_available`` does not exist."""
    snippet = DOC.read_text(encoding="utf-8")

    assert "npu_ready = torch.npu.is_available()" in snippet
    assert "if torch_npu.is_available()" not in snippet
