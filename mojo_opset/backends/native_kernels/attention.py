from mojo_opset.core.attn.mojo_prefill_gqa import MojoPrefillGQA, MojoPagedPrefillGQA
from mojo_opset.core.attn.mojo_decode_gqa import MojoDecodeGQA
import torch
import torch_npu
import torch.nn.functional as F
import math
from typing import Optional
from mojo_opset.backends.native_kernels.ascend.attention_mask import (
    init_ascend_attention_mask_builder, 
    make_attention_mask,
)

def _prepare_decoder_attention_mask(query):
    bsz, _, seq_len, _ = query.size()
    mask = torch.full((seq_len, seq_len), True,
                      device=query.device, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.view(1, 1, seq_len, seq_len).expand(bsz, 1, -1, -1)
    return mask


class NativeMojoPrefillGQA(MojoPrefillGQA, default_priority=0):
    def __init__(self,
                is_causal: bool = True,
                is_prefill: bool = True,
                softmax_scale: float = None,
                gqa_layout: str = "ABAB",
                rm_padding: bool = False,
                window_size: int = -1,
                op_name: str = "",):
        super().__init__(is_causal, is_prefill, softmax_scale, gqa_layout, rm_padding, window_size, op_name)

    def forward_std(self,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor) -> torch.Tensor:
        attn_mask = _prepare_decoder_attention_mask(query)
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            is_causal=False
        )

        return attn_output

class NativeMojoPagedPrefillGQA(MojoPagedPrefillGQA, default_priority=0):
    def __init__(self, 
                page_size = 0, 
                is_causal = True, 
                is_prefill = True, 
                gqa_layout = "ABAB", 
                window_size = -1, 
                op_name = "", 
                layer_idx = 0):
        super().__init__(page_size, is_causal, is_prefill, gqa_layout, window_size, op_name, layer_idx)
        init_ascend_attention_mask_builder(4096, torch.bfloat16, "npu")

    def forward_std(self, query, k_cache, v_cache, cu_seqlens_q, block_tables, sm_scale = None):
        total_q_tokens, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = k_cache.shape
        block_tables = block_tables.to(torch.int32)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim)
        
        if block_size == 128:
            compress_mask = make_attention_mask(2048, None, "npu")
            out, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=k_cache,
                value=v_cache,
                atten_mask=compress_mask.npu(),
                block_table=block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=cu_seqlens_q[1:].tolist(),
                actual_seq_lengths_kv=cu_seqlens_q[1:].tolist(),
                num_key_value_heads=num_kv_heads,
                num_heads=num_q_heads,
                scale=sm_scale,
                sparse_mode=3
            )
        else:
            actual_seq_lengths_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            actual_seq_lengths_q = actual_seq_lengths_q.to(torch.int32).to("cpu")
            block_tables = block_tables.to("cpu")
            out = torch.empty_like(query, device="npu")
            mask_compress_built = make_attention_mask(128, None, None)
            # context_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            # context_lens = context_lens.to(torch.int32).to("cpu")
            torch_npu._npu_flash_attention_qlens(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                block_table=block_tables,
                mask=mask_compress_built,
                seq_len=actual_seq_lengths_q,
                context_lens=actual_seq_lengths_q,
                num_kv_heads=num_kv_heads,
                num_heads=num_q_heads,
                scale_value=sm_scale,
                out=out)
        return out

class NativeMojoDecodeGQA(MojoDecodeGQA, default_priority=0):
    def __init__(self,
        is_causal: bool = True,
        is_prefill: bool = False,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        op_name: str = "",):
        super().__init__()
        pass

    def forward_std(self,
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor):
        attn_mask = _prepare_decoder_attention_mask(q)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False
        )

        return attn_output
