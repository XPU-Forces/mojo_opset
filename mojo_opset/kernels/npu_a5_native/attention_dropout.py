"""Native Philox4x32-10 generation of a packed, least-significant-bit-first mask.

Each decision uses the low eight bits of a uint16 Philox lane. Each 512-bit
packet uses 64 Philox counters and all four words per counter. The two 128-draw halves in each
256-draw group are interleaved before packing. No raw entropy buffer is stored.
"""

from functools import lru_cache
from typing import Optional

import torch


@lru_cache(maxsize=1)
def _launcher():
    from mojo_opset_lib.npu_a5 import _native  # noqa: F401

    if not hasattr(torch.ops.mojo_native_attention_dropout, "mask"):
        raise ImportError("Build native attention dropout: bash native/build.sh npu_a5 attention_dropout")
    return torch.ops.mojo_native_attention_dropout.mask


@torch.library.custom_op("mojo_npu_native_a5::attention_dropout_mask", mutates_args=())
def generate_mask(
    reference: torch.Tensor, num_words: int, p: float, seed: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    output = torch.empty(num_words, device=reference.device, dtype=torch.int32)
    if output.numel():
        if seed is None:
            seed = torch.randint(0, 2**31, (2,), device=reference.device, dtype=torch.int32)
        with torch.npu.device(reference.device):
            _launcher()(seed, output, int((1.0 - p) * 255))
    return output


@generate_mask.register_fake
def _generate_mask_fake(reference, num_words, p, seed=None):
    return reference.new_empty((num_words,), dtype=torch.int32)
