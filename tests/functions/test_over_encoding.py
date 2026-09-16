from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

NF4_DIMS = [128, 1024, 4096]


NGRAM_CASES = [
    "basic-prefill",
    "basic-decode",
    "history-decode",
    "prime-prefill",
    "varlen-97",
    "varlen-193",
    "varlen-257",
]


def make_nf4_lut(vocab, dim, device):
    indices = torch.randint(16, (vocab, dim), device="cpu", dtype=torch.uint8)
    packed = (indices[:, ::2] | (indices[:, 1::2] << 4)).to(device=device, dtype=torch.int8)
    return packed, torch.randn(vocab, dim // 64, device=device), torch.randn(vocab, dim // 64, device=device)


def make_ngram_case(case, device):
    if case.startswith("basic") or case == "history-decode":
        vocab, sizes, grams = 10, [10000] * 6, [2, 2, 3, 3, 4, 4]
        ids = torch.arange(1, 6, device=device)
        if case.endswith("prefill"):
            lens = [5]
            history = [[1] * 16 + [0] * 3]
        else:
            ids, lens = ids[:, None], None
            history = [([1] * 16 if case == "history-decode" else []) + [1, 2, 3]] * 5
    elif case == "prime-prefill":
        vocab, sizes, grams = 257, [263, 269, 271, 277, 281, 283, 293, 307], [2, 2, 3, 3, 4, 4, 5, 5]
        ids = torch.tensor(
            [11, 13, 15, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89], device=device
        )
        lens, history = [5, 7, 9], [[3, 5, 7, 9], [2, 4, 6, 8], [11, 13, 17, 19]]
    else:
        vocab, sizes, grams, lens, history = {
            "varlen-97": (97, [101, 103, 107, 109], [2, 2, 3, 3], [4, 6], [[5, 7], [11, 13]]),
            "varlen-193": (
                193,
                [197, 199, 211, 223, 227, 229],
                [2, 2, 3, 3, 4, 4],
                [3, 5, 7],
                [[2, 3, 5], [7, 11, 13], [17, 19, 23]],
            ),
            "varlen-257": (
                257,
                [263, 269, 271, 277, 281, 283, 293, 307],
                [2, 2, 3, 3, 4, 4, 5, 5],
                [8, 9, 11],
                [[3, 5, 7, 9], [2, 4, 6, 8], [11, 13, 17, 19]],
            ),
        }[case]
        ids = torch.arange(1, sum(lens) + 1, device=device) * 3 % vocab
    sizes, grams, history = (torch.tensor(x, dtype=torch.int64, device=device) for x in (sizes, grams, history))
    offsets = torch.cat((sizes.new_zeros(1), sizes[:-1].cumsum(0)))
    lens = torch.tensor(lens, device=device, dtype=torch.int64) if lens is not None else None
    return ids, history, lens, sizes, offsets, grams, vocab


def make_encoding_case(case, device):
    mode = case.rsplit("-", 1)[-1]
    kind = case.split("-", 1)[0]
    options = {}
    if kind == "basic":
        vocab, embed_dim, oe_dim, sizes, grams = 10, 128, 21, [10] * 6, [2, 2, 3, 3, 4, 4]
        ids = torch.arange(1, 7, device=device)
        lens = torch.tensor([3, 3], device=device) if mode == "prefill" else None
        history = torch.zeros(
            2 if lens is not None else 6, 3 if lens is not None else 4, device=device, dtype=torch.int64
        )
    elif kind in ("wide", "cpu"):
        vocab, embed_dim, oe_dim = 10086, 1536, 192
        sizes, grams = [10086 + 2**i for i in range(12)], [i for i in range(2, 8) for _ in range(2)]
        ids = torch.arange(1, 129, device=device)
        lens = torch.tensor([64, 64], device=device) if mode == "prefill" else None
        history = torch.ones(2 if lens is not None else 128, 6, device=device, dtype=torch.int64)
        if kind == "cpu":
            # Explicit custom weights, not uninitialized storage as in the old test.
            options = dict(
                _ori_embedding_weight=torch.randn(vocab, embed_dim, device=device),
                _mega_embedding_weight=torch.randn(sum(sizes), oe_dim, device="cpu"),
                mega_embedding_cpu_only=True,
            )
    elif kind == "prime":
        vocab, sizes, grams = 257, [263, 269, 271, 277, 281, 283, 293, 307], [2, 2, 3, 3, 4, 4, 5, 5]
        if mode == "prefill":
            embed_dim, oe_dim = 640, 80
            ids, history, lens, *_ = make_ngram_case("prime-prefill", device)
        else:
            embed_dim, oe_dim = 320, 40
            ids, lens = torch.arange(1, 49, device=device) % vocab, None
            history = torch.arange(48 * 4, device=device).view(48, 4) % 17
    else:
        vocab, embed_dim, oe_dim = 128, 64, 64
        sizes, grams = [131, 133, 135, 139], [2, 2, 3, 3]
        if mode == "prefill":
            ids = torch.arange(1, 33, device=device).expand(32, 32).flatten()
            lens = torch.full((32,), 32, device=device)
            history = torch.zeros(32, 2, device=device, dtype=torch.int64)
        else:
            ids, lens = torch.arange(1, 17, device=device), None
            history = torch.zeros(16, 2, device=device, dtype=torch.int64)
        weight, scale, mean = make_nf4_lut(sum(sizes), oe_dim, device)
        options = dict(
            _ori_embedding_weight=torch.randn(vocab, embed_dim, device=device),
            _mega_embedding_weight=weight,
            _mega_embedding_scale=scale,
            _mega_embedding_mean=mean,
            _mega_embedding_group_size=64,
        )
    if lens is None:
        ids = ids.view(-1, 1)
    return (vocab, embed_dim, oe_dim, sizes, grams), options, (ids, history, lens)


def make_nf4_case(dim, device):
    ids = torch.tensor([[0, 17, 32], [128, 256, 257]], device=device)
    return (ids, *make_nf4_lut(257, dim, device))


def _ngram_call(case, device):
    ids, history, lens, sizes, offsets, grams, vocab = make_ngram_case(case, device)
    if lens is None:
        return partial(F.n_gram_decode, vocab_size=vocab), (ids, history, sizes, offsets, grams)
    return partial(F.n_gram_prefill, vocab_size=vocab), (ids, lens, history, sizes, offsets, grams)


@pytest.mark.api("functions.n_gram_prefill", "functions.n_gram_decode")
@pytest.mark.parametrize("case", NGRAM_CASES)
@pytest.mark.accuracy
def test_ngram(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    function, inputs = _ngram_call(case, device)
    actual = function(*inputs, implementation=implementation)
    expected = function(*inputs, implementation="torch_reference")
    assert_close(actual, expected, rtol=0, atol=0)
    if case == "basic-prefill":
        golden = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [12, 12, 12, 12, 12, 12],
                [23, 23, 123, 123, 123, 123],
                [34, 34, 234, 234, 1234, 1234],
                [45, 45, 345, 345, 2345, 2345],
            ],
            device=device,
        )
        assert_close(actual, golden + torch.arange(6, device=device) * 10000, rtol=0, atol=0)


@pytest.mark.api("functions.embedding_nf4_dequant")
@pytest.mark.parametrize("dim", NF4_DIMS)
@pytest.mark.accuracy
def test_nf4(accuracy_backend, dim):
    implementation, _, device = accuracy_backend
    inputs = make_nf4_case(dim, device)
    actual = F.embedding_nf4_dequant(*inputs, group_size=64, output_dtype=torch.float32, implementation=implementation)
    expected = F.embedding_nf4_dequant(
        *inputs, group_size=64, output_dtype=torch.float32, implementation="torch_reference"
    )
    assert_close(actual, expected, rtol=0, atol=1e-5)


def _decode_case(device):
    config, weights, (ids, history, _) = make_encoding_case("nf4-decode", device)
    vocab, _, _, sizes, grams = config
    sizes, grams = (torch.tensor(x, device=device, dtype=torch.int64) for x in (sizes, grams))
    offsets = torch.cat((sizes.new_zeros(1), sizes[:-1].cumsum(0)))
    inputs = (
        ids,
        history,
        sizes,
        offsets,
        grams,
        weights["_mega_embedding_weight"],
        weights["_mega_embedding_scale"],
        weights["_mega_embedding_mean"],
    )
    return inputs, dict(ori_vocab_size=vocab, group_size=64, output_dtype=torch.float32)


@pytest.mark.api("functions.over_encoding_decode")
@pytest.mark.accuracy
def test_decode(accuracy_backend):
    implementation, _, device = accuracy_backend
    inputs, options = _decode_case(device)
    actual = F.over_encoding_decode(*inputs, **options, implementation=implementation)
    expected = F.over_encoding_decode(*inputs, **options, implementation="torch_reference")
    assert_close(actual, expected, rtol=1e-5, atol=1e-5)
