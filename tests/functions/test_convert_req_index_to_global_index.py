import pytest
import torch

from mojo_opset import functions as F


def _inputs(device):
    return (
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([[5, 7], [2, 9]], dtype=torch.int32, device=device),
        torch.tensor([[0, 7, 8, -1], [1, 8, 25, -1]], dtype=torch.int32, device=device),
        4, 4, True,
        torch.tensor([-1, 0], dtype=torch.int32, device=device),
        torch.tensor([100], dtype=torch.int32, device=device),
    )


def test_torch_decode_and_prefill():
    result = F.convert_req_index_to_global_index(*_inputs("cpu"), implementation="torch")
    expected = torch.tensor([[20, 31, -1, -1], [101, 108, 125, -1]], dtype=torch.int32)
    assert torch.equal(result, expected)


@pytest.mark.api("functions.convert_req_index_to_global_index")
@pytest.mark.accuracy
def test_npu_matches_reference(device, implementation):
    inputs = _inputs(device)
    actual = F.convert_req_index_to_global_index(*inputs, implementation=implementation)
    expected = F.convert_req_index_to_global_index(*inputs, implementation="torch")
    assert torch.equal(actual, expected)


def test_rejects_unsupported_shapes():
    args = list(_inputs("cpu"))
    args[3] = 3
    with pytest.raises(ValueError, match="block_size"):
        F.convert_req_index_to_global_index(*args, implementation="torch")
    args[3], args[4] = 4, 3
    with pytest.raises(ValueError, match="num_topk_tokens"):
        F.convert_req_index_to_global_index(*args, implementation="torch")
