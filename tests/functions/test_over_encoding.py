from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

NGRAM_CASES = [
    "basic-prefill",
    "basic-decode",
    "history-decode",
    "prime-prefill",
    "varlen-97",
    "varlen-193",
    "varlen-257",
]


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


def _ngram_call(case, device):
    ids, history, lens, sizes, offsets, grams, vocab = make_ngram_case(case, device)
    if lens is None:
        return partial(F.n_gram_decode, vocab_size=vocab), (ids, history, sizes, offsets, grams)
    return partial(F.n_gram_prefill, vocab_size=vocab), (ids, lens, history, sizes, offsets, grams)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=pytest.mark.api("functions.n_gram_decode" if case.endswith("decode") else "functions.n_gram_prefill"),
        )
        for case in NGRAM_CASES
    ],
)
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
