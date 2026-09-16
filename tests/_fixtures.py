import pytest
import torch


@pytest.fixture(autouse=True)
def seed_operator(accuracy_seed):
    """Apply the shared accelerator target and random seed to operator tests."""


@pytest.fixture(autouse=True)
def full_precision(accuracy_backend, monkeypatch):
    # A reference must not silently use reduced-mantissa FP32 GEMMs/convolutions.
    # Scope these flags to operator tests; do not change production/perf defaults.
    if accuracy_backend[2] == "npu":
        monkeypatch.setattr(torch.npu.conv, "allow_hf32", False)
        monkeypatch.setattr(torch.npu.matmul, "allow_hf32", False)
    elif accuracy_backend[2] == "cuda":
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
