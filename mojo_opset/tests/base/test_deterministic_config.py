import pytest
import torch

from mojo_opset.utils.misc import configure_torch_deterministic


@pytest.fixture(autouse=True)
def restore_torch_state():
    rng_state = torch.random.get_rng_state()
    deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic

    yield

    torch.random.set_rng_state(rng_state)
    torch.use_deterministic_algorithms(deterministic_algorithms_enabled)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


def test_configure_torch_deterministic_enables_deterministic_algorithms():
    torch.use_deterministic_algorithms(False)

    configure_torch_deterministic()

    assert torch.are_deterministic_algorithms_enabled()


def test_configure_torch_deterministic_sets_manual_seed():
    configure_torch_deterministic()
    actual = torch.rand(4)

    torch.manual_seed(0)
    expected = torch.rand(4)

    assert torch.equal(actual, expected)


def test_configure_torch_deterministic_sets_cudnn_flags_when_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    configure_torch_deterministic()

    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic
