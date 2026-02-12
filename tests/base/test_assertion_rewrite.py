import os
import pytest
import torch
from mojo_opset.core.operators.kv_cache import MojoStorePagedKVCache


def test_assertion_rewrite():
    store_kv_cache = MojoStorePagedKVCache._registry.get("torch")()
    with pytest.raises(AssertionError, match=".*2 == 3.*"):
        store_kv_cache(*[torch.empty(64, 64)] * 7)


if __name__ == "__main__":
    test_assertion_rewrite()
