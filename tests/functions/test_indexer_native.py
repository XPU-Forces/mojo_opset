"""A5 native LI entry, layout, compiler, and repeatability regressions."""

from functools import partial
from itertools import product

import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import Target
from mojo_opset import functions
from mojo_opset import preload
from mojo_opset.functions._dispatch import resolve_implementation
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph
from tests.functions.test_indexer import assert_lightning_scores


DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.int8)
# Sixteen packed entries and four separate FP16/BF16 H128 decode entries.
ENTRY_CASES = [
    pytest.param(dtype, heads, dim, 5, 129, id=f"{dtype}-h{heads}-d{dim}-prefill")
    for dtype, heads, dim in product(DTYPES, (64, 128), (64, 128))
] + [
    pytest.param(dtype, 128, dim, 1, 2051, id=f"{dtype}-h128-d{dim}-decode")
    for dtype, dim in product((torch.float16, torch.bfloat16), (64, 128))
]
COMPILE_CASES = [
    pytest.param(dtype, 64, 128, 5, id=f"{dtype}-prefill") for dtype in DTYPES
] + [
    pytest.param(dtype, 128, dim, 1, id=f"{dtype}-d{dim}-decode")
    for dtype, dim in product((torch.float16, torch.bfloat16), (64, 128))
]


@pytest.fixture
def native_li_backend(accuracy_backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.a5") or (
        implementation or resolve_implementation("lightning_indexer", Target.parse(target))
    ) != "native":
        pytest.skip("Native LI requires the selected A5 native provider")
    from mojo_opset.kernels.npu_a5_native.lightning_indexer import lightning_indexer_fwd

    return device, lightning_indexer_fwd


def _inputs(device, dtype, heads, dim, queries, keys, *, batch=2, scale_dtype=torch.float32):
    def values(shape):
        if dtype == torch.int8:
            return torch.randint(-128, 128, shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype)

    # Exercise normalization in the public API and in the native leaf itself.
    query = values((batch, queries, heads, dim * 2))[..., ::2]
    key = values((batch, keys, dim * 2))[..., ::2]
    weights = (torch.randn(batch, queries, heads * 2, device=device) * 0.02)[..., ::2]
    scale = (torch.randn(batch, keys * 2, device=device) * 0.02).to(scale_dtype)[:, ::2]
    return query, weights, key, scale


def _check_scores(inputs):
    actual = functions.lightning_indexer(*inputs, implementation="native")
    expected = functions.lightning_indexer(*inputs, implementation="torch")
    assert actual.shape == (inputs[0].shape[0], inputs[0].shape[1], inputs[2].shape[1])
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    assert_lightning_scores(actual, expected)
    return actual


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype,heads,dim,queries,keys", ENTRY_CASES)
def test_native_entries_and_tails(native_li_backend, dtype, heads, dim, queries, keys):
    # M=5 exercises packed/pair-packed group tails; N=129/2051 crosses tile boundaries.
    _check_scores(_inputs(native_li_backend[0], dtype, heads, dim, queries, keys))


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_kind", ("omitted", "broadcast", "fp16", "bf16"))
def test_native_key_scale_contract(native_li_backend, dtype, scale_kind):
    scale_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(scale_kind, torch.float32)
    query, weights, key, scale = _inputs(
        native_li_backend[0], dtype, 128, 64, 3, 257, scale_dtype=scale_dtype
    )
    if scale_kind == "omitted":
        scale = None
    elif scale_kind == "broadcast":
        scale = scale[0]
    _check_scores((query, weights, key, scale))


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,queries,keys", ((0, 3, 65), (2, 0, 65), (2, 3, 0)))
def test_native_empty_contract(native_li_backend, dtype, batch, queries, keys):
    device, forward = native_li_backend
    inputs = _inputs(device, dtype, 64, 128, queries, keys, batch=batch)
    _check_scores(inputs)
    torch.library.opcheck(forward, inputs, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype,heads,dim,queries,keys", ENTRY_CASES)
def test_native_fake_and_strided_leaf(native_li_backend, dtype, heads, dim, queries, keys):
    device, forward = native_li_backend
    inputs = _inputs(device, dtype, heads, dim, queries, keys, scale_dtype=torch.bfloat16)
    actual = forward(*inputs)
    expected = functions.lightning_indexer(*inputs, implementation="torch")
    assert_lightning_scores(actual, expected)
    with FakeTensorMode() as mode:
        fake = forward(*(mode.from_tensor(t) for t in inputs))
    assert (actual.shape, actual.dtype, actual.device, actual.stride()) == (
        fake.shape, fake.dtype, fake.device, fake.stride()
    )
    assert actual.is_contiguous()
    torch.library.opcheck(forward, inputs, test_utils=("test_schema", "test_faketensor"))


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype,heads,dim,queries", COMPILE_CASES)
@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_native_fullgraph(native_li_backend, dtype, heads, dim, queries, backend):
    preload("lightning_indexer", implementation="native")

    def run(*inputs):
        output = functions.lightning_indexer(*inputs, implementation="native")
        return output, output.stride(), output.is_contiguous()

    try:
        compiled = compile_fullgraph(run, backend=backend)
        for keys in (129, 257):
            inputs = _inputs(native_li_backend[0], dtype, heads, dim, queries, keys)
            # Compile through symbolic N, including its relation to key_scale.
            torch._dynamo.mark_dynamic(inputs[2], 1)
            torch._dynamo.mark_dynamic(inputs[3], 1)
            actual, stride, contiguous = compiled(*inputs)
            expected = functions.lightning_indexer(*inputs, implementation="torch")
            assert_lightning_scores(actual, expected)
            assert actual.dtype == torch.float32 and contiguous and actual.is_contiguous()
            assert tuple(stride) == actual.stride()
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.bitwise
@pytest.mark.parametrize("dtype,heads,dim,queries,keys", ENTRY_CASES)
def test_native_multiwave_bitwise(native_li_backend, dtype, heads, dim, queries, keys):
    # Prefill has enough work for multiple persistent tasks per physical core.
    queries = 65 if queries > 1 else 1
    inputs = _inputs(native_li_backend[0], dtype, heads, dim, queries, 2051)
    assert_repeatable(partial(functions.lightning_indexer, implementation="native"), inputs, backward=False)


@pytest.mark.api("functions.lightning_indexer")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES)
def test_native_nondefault_stream(native_li_backend, dtype):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        inputs = _inputs(native_li_backend[0], dtype, 64, 128, 5, 257, scale_dtype=torch.float16)
        actual = functions.lightning_indexer(*inputs, implementation="native")
        expected = functions.lightning_indexer(*inputs, implementation="torch")
    stream.synchronize()
    assert_lightning_scores(actual, expected)
