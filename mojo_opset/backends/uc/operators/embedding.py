"""UC backend implementation of :class:`MojoEmbedding`.

Wraps the wheel kernel APIs ``mojo_embedding_{bf16,fp16}`` (block GATHER,
``F.embedding`` semantics: flat ``out = weight[input_ids]``).

The wheel kernel is a fixed-shape bring-up build (see lessons § B.2 — no
``T.dynamic`` dims, hence no trailing INT32 scalar args), so any deviation
from ``(VOCAB, H, NUM_TOKENS) = (4096, 128, 64)`` falls back to ``F.embedding``
inherited from :class:`MojoEmbedding`.

Wheel API contract (mirrors ``mojo_relative_embedding_bf16`` which uses the
same gather pattern):

- weight       : <dtype>, shape ``(VOCAB, H)`` == ``(4096, 128)``
- input_ids    : ``int32``, shape ``(NUM_TOKENS,)`` == ``(64,)``
- out          : <dtype>, shape ``(NUM_TOKENS, H)`` == ``(64, 128)``
- arg_order    : ``inputs_first``  -> ``api(weight, input_ids, out)``

dtype routing follows the SOP in lessons § J.2 (``_resolve_api`` + dtype
white-list + soft ``kernels.get`` lookup).  ``fp32`` is intentionally not
served — falling back to ``F.embedding`` is faster than a vector-only
gather and matches the convention from :class:`UCLayerNorm` / etc.
"""

from typing import Callable, Optional

import torch

from mojo_opset.core import MojoEmbedding

from ._utils import _uc_kernels


# Wheel kernel hard-codes its tile shape during the initial bring-up. Any
# request outside this contract must fall back to ``F.embedding`` so the
# wrapper is always correct (per user preference §15: registered backend
# never raises, only fast-paths what it knows).
_FIXED_VOCAB = 4096
_FIXED_H = 128
_FIXED_NUM_TOKENS = 64

_DTYPE_TO_API = {
    torch.bfloat16: "mojo_embedding_bf16",
    torch.float16: "mojo_embedding_fp16",
}


def _resolve_api(dtype: torch.dtype) -> Optional[Callable]:
    """Soft-query the installed wheel for an embedding kernel of ``dtype``.

    Returns ``None`` when the dtype is not white-listed or the wheel does
    not yet ship the corresponding artifact (wrapper then falls back to
    ``F.embedding``).
    """
    api = _DTYPE_TO_API.get(dtype)
    if api is None:
        return None
    kernels = _uc_kernels()
    if api not in kernels.keys():
        return None
    return kernels[api]


class UCEmbedding(MojoEmbedding):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Any deviation from the fixed-shape contract -> torch F.embedding
        # via the parent class. Mirrors ``UCSdpa.forward``'s super() fallback.
        api_fn = _resolve_api(self.weight.dtype)
        if (
            api_fn is None
            or self.num_embeddings != _FIXED_VOCAB
            or self.embedding_dim != _FIXED_H
            or input.numel() != _FIXED_NUM_TOKENS
            or self.padding_idx is not None
            or self.max_norm is not None
        ):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        weight = self.weight.contiguous()

        # Kernel expects 1-D i32 indices; flatten + cast (the cast is cheap
        # and matches the lifter's int32 requirement on indirect index
        # buffers).
        input_ids = input.reshape(-1).contiguous()
        if input_ids.dtype != torch.int32:
            input_ids = input_ids.to(torch.int32)

        out_flat = torch.empty(
            (_FIXED_NUM_TOKENS, _FIXED_H),
            dtype=weight.dtype,
            device=weight.device,
        )
        api_fn(weight, input_ids, out_flat)
        # Restore original leading dims of ``input`` and append H.
        return out_flat.reshape(*input.shape, _FIXED_H)
