import torch
import torch_npu
import torch.nn.functional as F
from .attention_mask import (
    init_ascend_attention_mask_builder, 
    make_attention_mask,
)
from mojo_opset.core.attn.mojo_prefill_gqa import (
    MojoPagedPrefillGQA,
)
class TorchOpsMojoPagedPrefillGQA(MojoPagedPrefillGQA, default_priority=0):
    def __init__(self, 
                page_size, 
                num_q_heads,
                num_kv_heads, 
                head_dim,
                is_causal = True, 
                is_prefill = True, 
                gqa_layout = "ABAB", 
                window_size = -1,
                op_name = "", 
                layer_idx = 0):
        super().__init__(page_size, is_causal, is_prefill, gqa_layout, window_size, op_name, layer_idx)
        assert page_size >= 128 and page_size <= 512, f"Currently only support block size between 128 and 512."
        assert page_size % 128 == 0, f"Currently only support block size in multiples of 128."
        assert head_dim > 0, f"HeadDim is less than and equal to 0."
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        # npu require generating a mask, when there is a model config, its best to pass in max_seq_len
        init_ascend_attention_mask_builder(65536, torch.bfloat16, "npu")
        
    def forward_std(self, query, k_cache, v_cache, cu_seqlens_q, block_tables, sm_scale = None):
        if sm_scale is None:
            sm_scale = self.head_dim ** -0.5
        
        if block_tables.dtype != torch.int32:
            block_tables = block_tables.to(torch.int32)

        # ascend specific
        compress_mask = make_attention_mask(2048, device="npu")
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_cache,
            value=v_cache,
            atten_mask=compress_mask,
            block_table=block_tables,
            input_layout="TND",
            block_size=self.page_size,
            actual_seq_lengths=cu_seqlens_q[1:], # cummulative
            actual_seq_lengths_kv=cu_seqlens_q[1:]-cu_seqlens_q[:-1], # NOTE: Not cummulative, replace with context in the future
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_q_heads,
            scale=sm_scale,
            sparse_mode=3
        )
        return out