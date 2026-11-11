import pytest
import torch

from mojo_opset import functions as F


def _inputs(device):
    return (
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([[5, 7], [2, 9]], dtype=torch.int32, device=device),
        torch.tensor([[0, 7, 8, -1], [1, 8, 25, -1]], dtype=torch.int32, device=device),
        4,
        4,
        True,
        torch.tensor([-1, 0], dtype=torch.int32, device=device),
        torch.tensor([100], dtype=torch.int32, device=device),
    )


@pytest.mark.reference
def test_torch_decode_and_prefill():
    result = F.convert_req_index_to_global_index(*_inputs("cpu"), implementation="torch")
    expected = torch.tensor([[20, 31, -1, -1], [101, 108, 125, -1]], dtype=torch.int32)
    assert torch.equal(result, expected)


@pytest.mark.api("functions.convert_req_index_to_global_index")
@pytest.mark.accuracy
def test_npu_matches_reference(accuracy_backend):
    implementation, _, device = accuracy_backend
    inputs = _inputs(device)
    actual = F.convert_req_index_to_global_index(*inputs, implementation=implementation)
    expected = F.convert_req_index_to_global_index(*inputs, implementation="torch")
    assert torch.equal(actual, expected)


@pytest.mark.reference
def test_rejects_unsupported_shapes():
    args = list(_inputs("cpu"))
    args[3] = 3
    with pytest.raises(ValueError, match="block_size"):
        F.convert_req_index_to_global_index(*args, implementation="torch")
    args[3], args[4] = 4, 3
    with pytest.raises(ValueError, match="num_topk_tokens"):
        F.convert_req_index_to_global_index(*args, implementation="torch")


# Shape matrix from the delivered 22-case task.
CASES = [
    (8, 4, 10, 16, 128, False, 0, 0),
    (8, 4, 10, 16, 256, False, 0, 0),
    (8, 4, 10, 16, 512, False, 0, 0),
    (8, 4, 10, 64, 128, False, 0, 0),
    (8, 4, 10, 64, 256, False, 0, 0),
    (8, 4, 10, 64, 512, False, 0, 0),
    (8, 4, 10, 128, 128, False, 0, 0),
    (8, 4, 10, 128, 256, False, 0, 0),
    (8, 4, 10, 128, 512, False, 0, 0),
    (8, 4, 10, 8, 128, False, 0, 0),
    (12, 4, 8, 16, 128, True, 6, 2),
    (1, 1, 10, 16, 128, False, 0, 0),
    (64, 8, 10, 16, 128, False, 0, 0),
    (512, 32, 64, 64, 128, False, 0, 0),
    (2048, 64, 128, 128, 128, False, 0, 0),
    (8192, 256, 1024, 64, 512, False, 0, 0),
    (512, 16, 64, 16, 512, False, 0, 0),
    (2048, 128, 1024, 64, 512, False, 0, 0),
    (2048, 128, 1024, 128, 2048, False, 0, 0),
    (512, 32, 64, 128, 2048, False, 0, 0),
    (8192, 512, 1024, 128, 2048, False, 0, 0),
    (2048, 64, 1024, 64, 512, True, 1024, 8),
]


@pytest.mark.api("functions.convert_req_index_to_global_index")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", CASES)
def test_matrix(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    tokens, requests, blocks, block_size, topk, has_prefill, prefill_tokens, num_prefill = case
    req = torch.randint(requests, (tokens,), dtype=torch.int32, device=device)
    table = torch.randint(4096, (requests, blocks), dtype=torch.int32, device=device)
    indices = torch.randint(blocks * block_size, (tokens, topk), dtype=torch.int32, device=device)
    indices[0, :4] = -1
    if tokens > 1:
        indices[1, :4] = blocks * block_size
    ids = starts = None
    if has_prefill:
        ids = torch.full((tokens,), -1, dtype=torch.int32, device=device)
        ids[-prefill_tokens:] = (
            torch.arange(prefill_tokens, device=device, dtype=torch.int32) * num_prefill // prefill_tokens
        )
        starts = torch.arange(num_prefill, device=device, dtype=torch.int32) * 100
    args = (req, table, indices, block_size, topk, has_prefill, ids, starts)
    actual = F.convert_req_index_to_global_index(*args, implementation=implementation)
    expected = F.convert_req_index_to_global_index(*args, implementation="torch")
    assert torch.equal(actual, expected)
