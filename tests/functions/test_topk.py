import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import functions
from mojo_opset import preload
from tests._compile import compile_fullgraph

pytestmark = [pytest.mark.api("functions.topk_infer"), pytest.mark.accuracy]


def padded_input(batch, width, device):
    return torch.randn(batch, max(16, (width + 15) // 16 * 16), dtype=torch.bfloat16, device=device)[:, :width]


def check_topk(
    input, values, indices, k, *, end=None, offset=None, sorted_index=False,
    index_fill=2147483647, value_fill=float("-inf"),
):
    lengths = end.cpu().tolist() if end is not None else [input.shape[1]] * input.shape[0]
    offsets = offset.cpu().tolist() if offset is not None else [0] * input.shape[0]
    x, idx = input.cpu(), indices.cpu().long()
    assert indices.dtype == torch.int32 and indices.shape == (input.shape[0], k)
    assert indices.stride(1) == 1 and indices.stride(0) * indices.element_size() % 32 == 0
    if values is not None:
        assert values.dtype == torch.bfloat16 and values.shape == indices.shape
        assert values.stride(1) == 1 and values.stride(0) * values.element_size() % 32 == 0
        val = values.cpu()
    for row, length in enumerate(lengths):
        count = min(length, k)
        raw = idx[row, :count] - offsets[row]
        assert bool(((raw >= 0) & (raw < length)).all())
        assert raw.unique().numel() == count
        assert bool((idx[row, count:] == index_fill).all())
        if length <= k:
            assert torch.equal(raw, torch.arange(count))
        if sorted_index:
            assert bool((raw[1:] >= raw[:-1]).all())
        # Equal scores may select different indices; compare the selected scores and value/index pairing.
        expected = x[row, :length].topk(count).values.sort().values
        assert torch.equal(x[row, raw].sort().values, expected)
        if values is not None:
            assert torch.equal(val[row, :count].view(torch.int16), x[row, raw].view(torch.int16))
            fill = torch.tensor(value_fill, dtype=input.dtype)
            assert torch.equal(val[row, count:], fill.expand(k - count))


@pytest.mark.parametrize(
    "batch,width,k",
    [
        (1, 16, 1), (3, 7, 8), (3, 128, 128), (2, 2048, 511), (2, 2048, 512),
        (2, 4096, 513), (2, 4096, 1023), (2, 4096, 1024), (2, 8192, 1025),
        (2, 8192, 4095), (2, 8192, 4096), (1, 131073, 512), (65, 8192, 512),
        (0, 128, 8), (2, 0, 8), (1, 2**21 - 1, 512),
    ],
)
def test_topk_accuracy(accuracy_backend, batch, width, k):
    implementation, _, device = accuracy_backend
    x = padded_input(batch, width, device)
    values, indices = functions.topk_infer(x, k, implementation=implementation)
    check_topk(x, values, indices, k)


@pytest.mark.parametrize("sorted_index", [False, True])
@pytest.mark.parametrize("return_value", [False, True])
def test_topk_options(accuracy_backend, sorted_index, return_value):
    implementation, _, device = accuracy_backend
    x = padded_input(5, 65539, device)
    end = torch.tensor([0, 7, 513, 20001, 65539], dtype=torch.int32, device=device)
    offset = torch.tensor([10, -10, 100, 200, 300], dtype=torch.int32, device=device)
    out = torch.empty((5, 520), dtype=torch.int32, device=device)[:, :513]
    out.fill_(-123)
    before = x.clone()
    values, indices = functions.topk_infer(
        x, 513, end=end, sorted_index=sorted_index, output_idx=out, output_idx_offset=offset,
        idx_oob_fill_value=-17, value_oob_fill_value=-1.2345, return_value=return_value,
        implementation=implementation,
    )
    assert indices is out
    assert (values is not None) == return_value
    check_topk(
        x, values, indices, 513, end=end, offset=offset, sorted_index=sorted_index,
        index_fill=-17, value_fill=-1.2345,
    )
    assert torch.equal(x.view(torch.int16), before.view(torch.int16))


@pytest.mark.parametrize("case", ["equal", "negative", "infinity", "negative_inf", "zero", "subnormal"])
def test_topk_special_values(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    x = padded_input(2, 32769, device)
    if case == "equal":
        x.fill_(1)
    elif case == "negative":
        x.copy_(-x.abs())
    elif case == "infinity":
        x[:, :16] = float("inf")
        x[:, 16:32] = -float("inf")
    elif case == "negative_inf":
        x.fill_(-float("inf"))
    elif case == "zero":
        x.zero_()
        x[:, ::2] = -0.0
    else:
        tiny = torch.tensor([1, 2, 3, 127, 128, -32767], dtype=torch.int16, device=device).view(torch.bfloat16)
        x.copy_(tiny.repeat((x.numel() + 5) // 6)[:x.numel()].reshape_as(x))
    values, indices = functions.topk_infer(x, 512, implementation=implementation)
    check_topk(x, values, indices, 512)


@pytest.mark.parametrize("width,k", [(16, 32), (131072, 512)])
def test_topk_nan_sentinel(accuracy_backend, width, k):
    implementation, _, device = accuracy_backend
    x = padded_input(3, width, device)
    x[0, width - 1] = float("nan")
    x[1, 0] = float("nan")
    end = torch.tensor([width - 1, width, width], dtype=torch.int32, device=device)
    before = x.view(torch.int16).clone()
    values, indices = functions.topk_infer(
        x, k, end=end, abort_when_nan_found=False, implementation=implementation,
    )
    assert bool((indices[1] == 0x3f3f3f3f).all())
    for row in [0, 2]:
        check_topk(x[row:row + 1], values[row:row + 1], indices[row:row + 1], k, end=end[row:row + 1])
    assert torch.equal(before, x.view(torch.int16))


@pytest.mark.parametrize("options", [{"sorted": True}, {"indices_type": torch.int64}, {"begin": 0}, {"hint": 0}])
def test_topk_unsupported(accuracy_backend, options):
    implementation, _, device = accuracy_backend
    with pytest.raises(NotImplementedError):
        functions.topk_infer(padded_input(1, 16, device), 8, implementation=implementation, **options)


def test_topk_requires_inference(accuracy_backend):
    implementation, _, device = accuracy_backend
    x = padded_input(1, 16, device).requires_grad_()
    with pytest.raises(RuntimeError, match="does not support autograd"):
        functions.topk_infer(x, 8, implementation=implementation)


@pytest.mark.parametrize(
    "case", ["dtype", "inner_stride", "row_stride", "k_zero", "k_large", "n_large", "end", "output"]
)
def test_topk_native_rejections(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    if implementation == "torch":
        pytest.skip("native buffer contract")
    x, k, options = padded_input(2, 32, device), 8, {}
    if case == "dtype":
        x = x.half()
    elif case == "inner_stride":
        x = x[:, ::2]
    elif case == "row_stride":
        x = torch.randn(2, 31, dtype=torch.bfloat16, device=device)
    elif case == "k_zero":
        k = 0
    elif case == "k_large":
        k = 4097
    elif case == "n_large":
        x = padded_input(1, 2**21, device)
    elif case == "end":
        options["end"] = torch.full((2,), 32, dtype=torch.int64, device=device)
    else:
        options["output_idx"] = torch.empty(2, 9, dtype=torch.int32, device=device)[:, :8]
    with pytest.raises(RuntimeError):
        functions.topk_infer(x, k, implementation=implementation, **options)


FP32_OPTIONS = dict(sorted=True, indices_type=torch.int64)


def check_fp32(input, values, indices):
    k = values.shape[1]
    assert values.shape == indices.shape == (input.shape[0], k)
    assert values.dtype == torch.float32 and indices.dtype == torch.int64
    assert values.device == indices.device == input.device
    for output in (values, indices):
        assert output.stride(1) == 1 and output.stride(0) * output.element_size() % 32 == 0
    reference = input.cpu()
    expected = reference.argsort(dim=-1, descending=True, stable=True)[:, :k]
    assert torch.equal(indices.cpu(), expected)
    assert torch.equal(values.cpu().view(torch.int32), reference.gather(1, expected).view(torch.int32))


@pytest.mark.parametrize("rows", [1024, 8192, 16384, 32768])
def test_m15_accuracy(accuracy_backend, rows):
    implementation, _, device = accuracy_backend
    x = torch.randn(rows, 640, dtype=torch.float32, device=device).softmax(dim=-1)
    before = x.view(torch.int32).clone()
    values, indices = functions.topk_infer(x, 8, implementation=implementation, **FP32_OPTIONS)
    check_fp32(x, values, indices)
    repeated = functions.topk_infer(x, 8, implementation=implementation, **FP32_OPTIONS)
    assert torch.equal(repeated[0].view(torch.int32), values.view(torch.int32))
    assert torch.equal(repeated[1], indices)
    assert torch.equal(x.view(torch.int32), before)


@pytest.mark.parametrize("case", ["negative", "increasing", "decreasing", "equal", "negative_inf",
                                 "infinity", "zero", "same_high_bits", "subnormal", "tail"])
def test_m15_special_values(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    row = torch.randn(640)
    if case == "negative":
        row = -row.abs()
    elif case in ("increasing", "decreasing"):
        row = torch.arange(640, dtype=torch.float32)
        if case == "decreasing":
            row = row.flip(0)
    elif case == "equal":
        row.fill_(1)
    elif case == "negative_inf":
        row.fill_(-float("inf"))
    elif case == "infinity":
        row[:16] = float("inf")
        row[16:32] = -float("inf")
    elif case == "zero":
        row.zero_()
        row[::2] = -0.0
    elif case == "same_high_bits":
        row = (torch.arange(640, dtype=torch.int32) + 0x3f800000).view(torch.float32)
    elif case == "subnormal":
        bits = torch.arange(1, 641, dtype=torch.int32)
        bits[::2] |= -2147483648
        row = bits.view(torch.float32)
    else:
        row.fill_(-1)
        row[-8:] = torch.arange(8, dtype=torch.float32)
    x = row.repeat(1024, 1).to(device)
    before = x.view(torch.int32).clone()
    check_fp32(x, *functions.topk_infer(x, 8, implementation=implementation, **FP32_OPTIONS))
    assert torch.equal(x.view(torch.int32), before)


def test_m15_padded_buffers(accuracy_backend):
    implementation, _, device = accuracy_backend
    x = torch.randn(1024, 648, dtype=torch.float32, device=device)[:, :640]
    out = torch.empty(1024, 12, dtype=torch.int64, device=device)[:, :8]
    values, indices = functions.topk_infer(x, 8, output_idx=out, implementation=implementation, **FP32_OPTIONS)
    assert indices is out
    check_fp32(x, values, indices)


def test_m15_nan_sentinel(accuracy_backend):
    implementation, _, device = accuracy_backend
    x = torch.randn(1024, 640, dtype=torch.float32, device=device)
    x[0, 639] = float("nan")
    x[16, 0] = float("nan")
    before = x.view(torch.int32).clone()
    values, indices = functions.topk_infer(
        x, 8, abort_when_nan_found=False, implementation=implementation, **FP32_OPTIONS,
    )
    assert bool((indices[[0, 16]] == 0x3f3f3f3f).all())
    valid = torch.ones(1024, dtype=torch.bool, device=device)
    valid[[0, 16]] = False
    check_fp32(x[valid], values[valid], indices[valid])
    assert torch.equal(x.view(torch.int32), before)


@pytest.mark.parametrize(
    "shape,k", [((3, 0), 1), ((3, 1025), 8), ((3, 640), 0), ((3, 640), 33), ((3, 7), 8), ((1, 3, 640), 8)]
)
def test_fp32_shape_contract(accuracy_backend, shape, k):
    implementation, _, device = accuracy_backend
    with pytest.raises(ValueError, match="FP32 topk_infer requires"):
        functions.topk_infer(torch.empty(shape, device=device), k, implementation=implementation, **FP32_OPTIONS)


@pytest.mark.parametrize(
    "option", ["sorted", "indices_type", "sorted_index", "return_value", "end", "output_idx_offset"]
)
def test_fp32_unsupported_options(accuracy_backend, option):
    implementation, _, device = accuracy_backend
    options = {**FP32_OPTIONS, "sorted_index": False, "return_value": True}
    if option in ("end", "output_idx_offset"):
        options[option] = torch.zeros(1024, dtype=torch.int32, device=device)
    else:
        options[option] = torch.int32 if option == "indices_type" else not options[option]
    with pytest.raises(NotImplementedError, match="FP32 topk_infer"):
        functions.topk_infer(torch.empty(1024, 640, device=device), 8, implementation=implementation, **options)


def test_m15_requires_inference(accuracy_backend):
    implementation, _, device = accuracy_backend
    x = torch.randn(1024, 640, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        functions.topk_infer(x, 8, implementation=implementation, **FP32_OPTIONS)
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            values, indices = functions.topk_infer(x, 8, implementation=implementation, **FP32_OPTIONS)
        assert not values.requires_grad and not indices.requires_grad


RESCAN_WIDTHS = [1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 608, 639, 640, 641, 1023, 1024]


@pytest.mark.parametrize("width,k", [(n, k) for n in RESCAN_WIDTHS for k in range(1, min(n, 32) + 1)])
def test_fp32_rescan_accuracy(accuracy_backend, width, k):
    implementation, _, device = accuracy_backend
    source = torch.randn(17, width, dtype=torch.float32)
    source[0].fill_(-float("inf"))
    source[1] = -source[1].abs() - 1
    source[2].fill_(1)
    source[3].zero_()
    source[3, ::2] = -0.0
    source[4, ::3] = float("inf")
    source[4, 1::3] = -float("inf")
    source[5] = (torch.arange(width, dtype=torch.int32) + 0x3f800000).view(torch.float32)
    x = source.to(device)
    check_fp32(x, *functions.topk_infer(x, k, implementation=implementation, **FP32_OPTIONS))
    assert torch.equal(x.cpu().view(torch.int32), source.view(torch.int32))


@pytest.mark.parametrize("rows", [0, 1, 7, 15, 16, 17, 33])
def test_fp32_rescan_batches(accuracy_backend, rows):
    implementation, _, device = accuracy_backend
    x = torch.randn(rows, 641, dtype=torch.float32, device=device)
    check_fp32(x, *functions.topk_infer(x, 32, implementation=implementation, **FP32_OPTIONS))


@pytest.mark.parametrize("width", RESCAN_WIDTHS)
def test_fp32_rescan_padded_buffers(accuracy_backend, width):
    implementation, _, device = accuracy_backend
    k = min(width, 32)
    storage = torch.randn(17, width + 5, dtype=torch.float32, device=device)
    storage[:, width + 1:] = float("nan")
    x = storage[:, 1:width + 1]
    before = storage.view(torch.int32).clone()
    out = torch.empty(17, (k + 3) // 4 * 4 + 4, dtype=torch.int64, device=device)[:, :k]
    values, indices = functions.topk_infer(x, k, output_idx=out, implementation=implementation, **FP32_OPTIONS)
    assert indices is out
    check_fp32(x, values, indices)
    assert torch.equal(storage.view(torch.int32), before)


@pytest.mark.parametrize("width", [1, 33, 256, 257, 512, 513, 640, 641, 1024])
def test_fp32_rescan_nan(accuracy_backend, width):
    implementation, _, device = accuracy_backend
    x = torch.randn(17, width, dtype=torch.float32, device=device)
    x[0, width - 1] = float("nan")
    x[16, 0] = float("nan")
    k = min(width, 32)
    values, indices = functions.topk_infer(
        x, k, abort_when_nan_found=False, implementation=implementation, **FP32_OPTIONS,
    )
    assert bool((indices[[0, 16]] == 0x3f3f3f3f).all())
    check_fp32(x[1:16], values[1:16], indices[1:16])


@pytest.mark.parametrize("width,k", [(0, 1), (1025, 8), (640, 0), (640, 33), (7, 8)])
def test_fp32_native_bounds(accuracy_backend, width, k):
    implementation, _, device = accuracy_backend
    if implementation == "torch":
        pytest.skip("native host bounds")
    from mojo_opset_lib.npu_a5 import _native

    x = torch.empty(3, width, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError):
        _native.topk_forward(x, k, None, False, None, None, 2147483647, -float("inf"), True, True)


@pytest.fixture
def native_topk_backend(accuracy_backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.a5") or implementation not in (None, "native"):
        pytest.skip("A5 native TopK framework checks")
    return device


def graph_input(device, dtype, rows, width, k, sorted_index=False, return_value=True):
    options = dict(sorted_index=sorted_index, return_value=return_value)
    if dtype == torch.float32:
        x = torch.randn(rows, width, device=device).softmax(-1)
        options.update(FP32_OPTIONS)
    else:
        x = padded_input(rows, width, device)
        options.update(end=torch.full((rows,), width, dtype=torch.int32, device=device),
                       output_idx_offset=torch.arange(rows, dtype=torch.int32, device=device),
                       idx_oob_fill_value=-17, value_oob_fill_value=-1.2345)
    return x, options


def check_graph_output(x, output, k, options):
    if x.dtype == torch.float32:
        check_fp32(x, *output)
    else:
        check_topk(x, *output, k, end=options["end"], offset=options["output_idx_offset"],
                   sorted_index=options["sorted_index"], index_fill=-17, value_fill=-1.2345)


@pytest.mark.parametrize("dtype,rows,width,k,return_value", [
    (torch.bfloat16, 3, 8192, 513, True),
    (torch.bfloat16, 3, 8192, 513, False),
    (torch.bfloat16, 0, 0, 8, True),
    (torch.float32, 17, 33, 3, True),
    (torch.float32, 0, 256, 8, True),
])
def test_topk_fake_and_opcheck(native_topk_backend, dtype, rows, width, k, return_value):
    x, options = graph_input(native_topk_backend, dtype, rows, width, k, return_value=return_value)
    actual = functions.topk_infer(x, k, implementation="native", **options)
    check_graph_output(x, actual, k, options)
    from mojo_opset.kernels.npu_a5_native.topk import _topk

    args = (x, k, options.get("end"), False, options.get("output_idx_offset"),
            options.get("idx_oob_fill_value", 2147483647), options.get("value_oob_fill_value", -float("inf")),
            return_value, True)
    with FakeTensorMode() as mode:
        fake = _topk(*(mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args))
    for real, meta in zip(actual, fake):
        if real is None:
            assert meta is None
        else:
            assert (real.shape, real.dtype, real.device, real.stride()) == (
                meta.shape, meta.dtype, meta.device, meta.stride())
    torch.library.opcheck(_topk, args, test_utils=("test_schema", "test_faketensor"))


GRAPH_CASES = [
    (torch.bfloat16, 3, 8192, 512, False, True),
    (torch.bfloat16, 3, 16384, 513, True, False),
    (torch.bfloat16, 3, 32768, 1025, False, True),
    *[(torch.float32, rows, 640, 8, False, True) for rows in (1024, 8192, 16384, 32768)],
    *[(torch.float32, 17, width, 32, False, True) for width in (255, 256, 511, 512, 639, 1024)],
    (torch.float32, 17, 641, 32, False, True),
]


@pytest.mark.parametrize("dtype,rows,width,k,sorted_index,return_value", GRAPH_CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
def test_topk_npu_graph(native_topk_backend, dtype, rows, width, k, sorted_index, return_value, compiled):
    preload("topk_infer", implementation="native")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    try:
        with torch.npu.stream(stream):
            x, options = graph_input(native_topk_backend, dtype, rows, width, k, sorted_index, return_value)

            def run():
                return functions.topk_infer(x, k, implementation="native", **options)

            run()
            call = compile_fullgraph(run, backend="eager") if compiled else run
            for _ in range(3):
                call()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                actual = call()
            for iteration in range(1, 4):
                x.copy_(torch.randn_like(x))
                if dtype == torch.bfloat16:
                    options["end"].fill_(width - iteration)
                    options["end"][0] = min(iteration, k)
                    options["output_idx_offset"].add_(13)
                graph.replay()
                expected = run()
                stream.synchronize()
                check_graph_output(x, actual, k, options)
                for a, b in zip(actual, expected):
                    if a is not None:
                        assert torch.equal(a, b)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_topk_nondefault_stream_lifetime(native_topk_backend, dtype):
    source = torch.randn(17, 640, dtype=dtype)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        x = source.to(native_topk_backend)
        options = FP32_OPTIONS if dtype == torch.float32 else {}
        output = functions.topk_infer(x, 8, implementation="native", **options)
        del x
        _churn = torch.empty(4, 17, 640, dtype=dtype, device=native_topk_backend).fill_(123)
    stream.synchronize()
    if dtype == torch.float32:
        check_fp32(source, *(tensor.cpu() for tensor in output))
    else:
        check_topk(source, *output, 8)
