import itertools

import pytest
import torch

from mojo_opset import functions as F

OPS = ('gemm', 'mxfp8_dequant_gemm', 'mxfp4_gemm')
AOT_OPS = ('gemm', 'mxfp8_gemm', 'mxfp8_dequant_gemm', 'mxfp4_gemm')
pytestmark = [pytest.mark.accuracy, pytest.mark.api(*(f'functions.{op}' for op in OPS))]


def operand(rows, cols, dtype, device, strided=False):
    if dtype == torch.int8:
        assert cols % 2 == 0
        # Independent signed E2M1 nibbles, including negative values.
        x = torch.randint(-128, 128, (rows, cols // 2), dtype=dtype, device='cpu')
    else:
        x = torch.randn(rows, cols, device='cpu').to(dtype)
    x = x.to(device)
    if strided:
        padded = torch.empty((x.shape[0], x.shape[1] + 16), dtype=dtype, device=device)
        padded[:, :x.shape[1]].copy_(x)
        x = padded[:, :x.shape[1]]
    return x


def inputs(op, m, n, k, ta, tb, device, packed_sf=False, recipe=(1, 1, 32), strided=False):
    da = torch.bfloat16 if op == 'gemm' else torch.int8 if op == 'mxfp4_gemm' else torch.float8_e4m3fn
    db = torch.int8 if op in ('mxfp4_gemm', 'mxfp8_dequant_gemm') else da
    a = operand(k, m, da, device, strided) if ta else operand(m, k, da, device, strided)
    b = operand(n, k, db, device, strided) if tb else operand(k, n, db, device, strided)
    if op == 'gemm':
        return a, b
    scales = []
    for rows, gran in ((m, recipe[0]), (n, recipe[1])):
        shape = ((rows + gran - 1) // gran, (k + 31) // 32)
        exp = torch.randint(123, 128, shape, dtype=torch.int32, device='cpu')
        if packed_sf:
            if exp.shape[1] % 2:
                exp = torch.cat((exp, torch.full((shape[0], 1), 127, dtype=exp.dtype)), dim=1)
            sf = (exp[:, ::2] | (exp[:, 1::2] << 8)).to(torch.int16).t().contiguous().t()
        else:
            sf = torch.pow(2., exp.float() - 127)
        scales.append(sf.to(device))
    return a, b, *scales


def cpu_reference(op, args, kwargs):
    args = tuple(x.cpu() for x in args)
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    return getattr(F, op)(*args, **kwargs, implementation='torch')


def compare(actual, expected):
    actual = actual.cpu().float()
    expected = expected.float()
    assert torch.isfinite(actual).all()
    # BF16 output/atomic C accumulation may round at different stages.
    error = (actual - expected).square().sum()
    assert error <= 3e-5 * expected.square().sum().clamp_min(1.)
    torch.testing.assert_close(actual, expected, rtol=.025, atol=.125)


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('ta,tb', itertools.product((False, True), repeat=2))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_dense_layouts(accuracy_backend, op, ta, tb, dtype):
    impl, _, device = accuracy_backend
    args = inputs(op, 64, 96, 256, ta, tb, device)
    kwargs = dict(trans_input=ta, trans_weight=tb, output_dtype=dtype)
    compare(getattr(F, op)(*args, **kwargs, implementation=impl), cpu_reference(op, args, kwargs))


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('mode', ('tail', 'bias', 'accumulate', 'alpha', 'packed_scale', 'recipe', 'strided'))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_dense_options(accuracy_backend, op, mode, dtype):
    impl, _, device = accuracy_backend
    m, n, k = (18, 34, 70) if mode == 'tail' else (64, 96, 256)
    recipe = (2, 4, 32) if mode == 'recipe' else (1, 1, 32)
    args = inputs(op, m, n, k, False, True, device, mode == 'packed_scale', recipe, mode == 'strided')
    kwargs = dict(trans_weight=True, output_dtype=dtype)
    if op != 'gemm':
        kwargs['recipe'] = recipe
    if mode in ('bias', 'accumulate'):
        kwargs['c'] = torch.randn(1 if mode == 'bias' else m, n, device=device, dtype=dtype)
    if mode == 'alpha':
        kwargs['alpha'] = .5
    before = [x.cpu().clone() for x in args]
    c_before = kwargs['c'].cpu().clone() if 'c' in kwargs else None
    compare(getattr(F, op)(*args, **kwargs, implementation=impl), cpu_reference(op, args, kwargs))
    for x, old in zip(args, before):
        assert torch.equal(x.cpu().contiguous().view(torch.uint8), old.contiguous().view(torch.uint8))
    if c_before is not None:
        assert torch.equal(kwargs['c'].cpu(), c_before)


@pytest.mark.parametrize('op', OPS)
def test_dense_current_stream_and_compile(accuracy_backend, op):
    impl, _, device = accuracy_backend
    if device != 'npu' or impl not in (None, 'native'):
        pytest.skip('native NPU stream contract')
    args = inputs(op, 64, 96, 256, False, True, device)
    kwargs = dict(trans_weight=True, output_dtype=torch.float32)
    call = lambda *xs: getattr(F, op)(*xs, **kwargs, implementation='native')
    call(*args)  # preload Python wrapper before Dynamo capture
    compiled = torch.compile(call, backend='eager', fullgraph=True)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        result = compiled(*args)
    torch.npu.current_stream().wait_stream(stream)
    compare(result, cpu_reference(op, args, kwargs))


@pytest.mark.parametrize('shape', [(0, 8, 32), (4, 0, 32), (4, 8, 0), (1, 1, 1), (4, 8, 1)])
def test_bf16_empty_and_singleton_native(accuracy_backend, shape):
    impl, _, device = accuracy_backend
    m, n, k = shape
    a = torch.ones(m, k, dtype=torch.bfloat16, device=device)
    b = torch.ones(k, n, dtype=torch.bfloat16, device=device)
    out = F.gemm(a, b, implementation=impl)
    torch.testing.assert_close(out.cpu(), torch.full((m, n), k, dtype=torch.bfloat16, device='cpu'))


@pytest.mark.parametrize('packed', (False, True))
@pytest.mark.parametrize('mn_major', (False, True))
@pytest.mark.parametrize('gran', (1, 2, 4, 8, 16, 32, 64, 128))
def test_dense_aot_scale_catalogue(accuracy_backend, packed, mn_major, gran):
    impl, _, device = accuracy_backend
    recipe = (gran, gran, 32)
    a, b, sa, sb = inputs('mxfp8_dequant_gemm', 256, 256, 128, False, True, device, packed, recipe)
    # Exercise all 32 compiled SF converters, including packed K-major input.
    layout = lambda s: s.t().contiguous().t() if mn_major else s.contiguous()
    args = a, b, layout(sa), layout(sb)
    kwargs = dict(trans_weight=True, recipe=recipe, output_dtype=torch.float32)
    compare(F.mxfp8_dequant_gemm(*args, **kwargs, implementation=impl), cpu_reference('mxfp8_dequant_gemm', args, kwargs))


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('ta,tb', itertools.product((False, True), repeat=2))
@pytest.mark.parametrize('shape', [(514, 258, 510), (1024, 1024, 128)])
def test_dense_aot_dynamic_shapes_and_core_count(accuracy_backend, op, ta, tb, shape):
    impl, _, device = accuracy_backend
    m, n, k = shape
    args = inputs(op, m, n, k, ta, tb, device)
    kwargs = dict(trans_input=ta, trans_weight=tb, output_dtype=torch.float32,
                  alpha=.5, c=torch.randn(m, n, device=device), compiled_dims='')
    compare(getattr(F, op)(*args, **kwargs, implementation=impl), cpu_reference(op, args, kwargs))


def test_aot_override_cannot_bypass_store_alignment(accuracy_backend, monkeypatch):
    import runpy

    from pathlib import Path

    impl, _, device = accuracy_backend
    if device != "npu" or impl not in (None, "native"):
        pytest.skip("AOT dispatch contract")
    generator = runpy.run_path(str(Path(__file__).resolve().parents[2] / "native/src/npu_a5/dense_gemm/generate.py"))
    config = next(c for c in generator["configs"](0, 0, 0, 0) if c["direct_store"])
    args = inputs("gemm", 17, 33, 256, False, True, device)
    for tag in ("not-compiled", generator["config_tag"](config)):
        monkeypatch.setenv("MOJO_GEMM_FORCE_CONFIG", tag)
        with pytest.raises(RuntimeError, match="unavailable or unsafe"):
            F.gemm(*args, trans_weight=True, implementation="native")
    monkeypatch.delenv("MOJO_GEMM_FORCE_CONFIG")
    compare(
        F.gemm(*args, trans_weight=True, implementation="native"),
        cpu_reference("gemm", args, dict(trans_weight=True)),
    )


def test_aot_tuning_is_scoped_to_shape_and_strides(accuracy_backend, monkeypatch, capfd):
    import ctypes
    import json

    from pathlib import Path

    impl, _, device = accuracy_backend
    if device != "npu" or impl not in (None, "native"):
        pytest.skip("AOT dispatch contract")
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    soc = acl.aclrtGetSocName().decode()
    cores = torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num
    entries = json.loads((Path(__file__).resolve().parents[2] / "native/src/npu_a5/dense_gemm/tuning.json").read_text())
    entry = next((r for r in entries if r["soc"] == soc and r["cores"] == cores and r["key"] == 0), None)
    if entry is None:
        pytest.skip("no tuned BF16 shape for this device")
    monkeypatch.setenv("MOJO_GEMM_PRINT_CONFIG", "1")
    monkeypatch.delenv("MOJO_GEMM_FORCE_CONFIG", raising=False)
    monkeypatch.delenv("MOJO_GEMM_DISABLE_TUNING", raising=False)
    for strided, disabled, source in [(False, False, "tuned"), (True, False, "model"), (False, True, "model")]:
        if disabled:
            monkeypatch.setenv("MOJO_GEMM_DISABLE_TUNING", "1")
        args = inputs("gemm", *entry["shape"], False, True, device, strided=strided)
        actual = F.gemm(*args, trans_weight=True, implementation="native")
        compare(actual, cpu_reference("gemm", args, dict(trans_weight=True)))
        ctypes.CDLL(None).fflush(None)
        assert f"source={source}" in capfd.readouterr().out


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_aot_direct_store_k_tail(accuracy_backend, monkeypatch, op, dtype):
    import runpy

    from pathlib import Path

    impl, _, device = accuracy_backend
    if device != "npu" or impl not in (None, "native"):
        pytest.skip("AOT dispatch contract")
    generator = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                                  "native/src/npu_a5/dense_gemm/generate.py"))
    args = inputs(op, 18, 96, 70, False, True, device, packed_sf=True, strided=True)
    kwargs = dict(trans_weight=True, output_dtype=dtype)
    expected = cpu_reference(op, args, kwargs)
    for config in generator["configs"](AOT_OPS.index(op), 0, 0, 0):
        if config["direct_store"]:
            monkeypatch.setenv("MOJO_GEMM_FORCE_CONFIG", generator["config_tag"](config))
            compare(getattr(F, op)(*args, **kwargs, implementation="native"), expected)


@pytest.mark.parametrize("restriction", ("shape_k", "shape_n", "single_m_block", "resident_a", "resident_b"))
def test_aot_specialization_rejects_incompatible_shape(accuracy_backend, monkeypatch, restriction):
    import json
    import runpy

    from pathlib import Path

    impl, _, device = accuracy_backend
    if device != "npu" or impl not in (None, "native"):
        pytest.skip("AOT specialization guards")
    root = Path(__file__).resolve().parents[2] / "native/src/npu_a5/dense_gemm"
    generator = runpy.run_path(str(root / "generate.py"))
    specs = json.loads((root / "specializations.json").read_text())
    spec = next((s for s in specs if s["overrides"].get(restriction) and s["key"] % 40 == 0 and s["key"] // 40 != 1), None)
    if spec is None:
        pytest.skip(f"no retained {restriction} specialization")
    kind = spec["key"] // 40
    base = next(c for c in generator["configs"](kind, 0, 0, 0) if generator["config_tag"](c) == spec["base_tag"])
    c = dict(base, **spec["overrides"])
    m, n, k = min(16, c["block_m"]), c.get("shape_n", c["block_n"]), c.get("shape_k", 256)
    if restriction == "shape_k":
        k += 64
    elif restriction == "shape_n":
        n += 64
    elif restriction in ("single_m_block", "resident_a"):
        m = c["block_m"] + 16
    else:
        n = c["block_n"] + 64
    args = inputs(AOT_OPS[kind], m, n, k, False, True, device, packed_sf=True)
    monkeypatch.setenv("MOJO_GEMM_FORCE_CONFIG", generator["config_tag"](c))
    with pytest.raises(RuntimeError, match="unavailable or unsafe"):
        getattr(F, AOT_OPS[kind])(*args, trans_weight=True, implementation="native")


@pytest.mark.parametrize("restriction", ("resident_a", "single_m_block"))
@pytest.mark.parametrize("strided", (False, True))
def test_aot_specialized_small_m_tails(accuracy_backend, monkeypatch, restriction, strided):
    import json
    import runpy

    from pathlib import Path

    impl, _, device = accuracy_backend
    if device != "npu" or impl not in (None, "native"):
        pytest.skip("AOT specialization accuracy")
    root = Path(__file__).resolve().parents[2] / "native/src/npu_a5/dense_gemm"
    generator = runpy.run_path(str(root / "generate.py"))
    specs = json.loads((root / "specializations.json").read_text())
    spec = next((s for s in specs if s["overrides"].get(restriction) and s["key"] % 40 == 0 and s["key"] // 40 != 1), None)
    if spec is None:
        pytest.skip(f"no retained {restriction} specialization")
    kind = spec["key"] // 40
    base = next(c for c in generator["configs"](kind, 0, 0, 0) if generator["config_tag"](c) == spec["base_tag"])
    c = dict(base, **spec["overrides"])
    n, k = c.get("shape_n", 256), c.get("shape_k", 1024)
    monkeypatch.setenv("MOJO_GEMM_FORCE_CONFIG", generator["config_tag"](c))
    for m in (1, 18, c["block_m"]):
        args = inputs(AOT_OPS[kind], m, n, k, False, True, device, packed_sf=True, strided=strided)
        kwargs = dict(trans_weight=True)
        compare(getattr(F, AOT_OPS[kind])(*args, **kwargs, implementation="native"), cpu_reference(AOT_OPS[kind], args, kwargs))
