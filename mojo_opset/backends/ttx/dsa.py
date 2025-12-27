import torch

from mojo_opset.backends.ttx.kernels.ascend.prefill_dsa import ttx_prefill_dsa
from mojo_opset.backends.ttx.kernels.ascend.decode_dsa import ttx_decode_dsa
from mojo_opset.core import MojoPrefillDSA
from mojo_opset.core import MojoDecodeDSA

class TTXPrefillDSA(MojoPrefillDSA, default_priority=0):
    def forward_std(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        return ttx_prefill_dsa(query, key, value, topk_indices)

class TTXDecodeDSA(MojoDecodeDSA, default_priority=0):
    def forward_std(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        wkv_b: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        start_pos: int,
    ):
        return ttx_decode_dsa(q_nope, q_pe, wkv_b, kv_cache, pe_cache, topk_indices, start_pos)