"""NPU Fusion Attention Implementation for PyTorch."""

import torch
import torch_npu
from mojo_opset.core import MojoFusionAttention
from mojo_opset.core import MojoFusedInferAttentionScore


class TorchNpuFusionAttention(MojoFusionAttention):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query,
        key,
        value,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        pre_tockens=65535,
        next_tockens=65536,
        sparse_mode=0,
        is_varlen=False,
        is_causal=False,
        input_layout: str = "BNSD",
        **kwargs,
    ):
        input_layout = "TND" if is_varlen else input_layout

        # Prepare causal mask if needed
        if is_causal and self.mask is None:
            if is_varlen:
                if actual_seq_qlen is None or actual_seq_kvlen is None:
                    raise ValueError("actual_seq_qlen/kvlen required for varlen mode")
                seq_q = actual_seq_qlen if isinstance(actual_seq_qlen, torch.Tensor) else torch.tensor(actual_seq_qlen)
                seq_kv = actual_seq_kvlen if isinstance(actual_seq_kvlen, torch.Tensor) else torch.tensor(actual_seq_kvlen)
                max_q = torch.diff(seq_q, prepend=torch.tensor([0])).max().item()
                max_kv = torch.diff(seq_kv, prepend=torch.tensor([0])).max().item()
            else:
                max_q, max_kv = query.shape[-2], key.shape[-2]
            self.mask = torch.triu(
                torch.ones(max_q, max_kv, dtype=torch.bool, device=query.device),
                diagonal=1,
            )

        # Convert to list for NPU API
        if isinstance(actual_seq_qlen, torch.Tensor):
            actual_seq_qlen = actual_seq_qlen.tolist()
        if isinstance(actual_seq_kvlen, torch.Tensor):
            actual_seq_kvlen = actual_seq_kvlen.tolist()

        # Forward
        attn_out = \
            torch_npu.npu_fusion_attention(
                query=query, 
                key=key, 
                value=value, 
                head_num=self.head_num,
                input_layout=input_layout,
                pre_tockens=pre_tockens,
                next_tockens=next_tockens,
                sparse_mode=sparse_mode,
                atten_mask=self.mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                scale=self.scale,
                **kwargs,
            )[0]

        return attn_out


class TorchNpuFusedInferAttentionScore(MojoFusedInferAttentionScore):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query,
        key,
        value,
        actual_seq_lengths=None,
        actual_seq_lengths_kv=None,
        num_kv_heads=0,
        block_table=None,
        is_varlen=False,
        input_layout: str = "BNSD",
        block_size: int = 0,
        **kwargs,   
    ):
        input_layout = "TND" if is_varlen else input_layout     
        
        attn_out = \
            torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                atten_mask=self.mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                block_table=block_table,
                num_heads=self.head_num,
                scale=self.scale,
                input_layout=input_layout,
                num_key_value_heads=num_kv_heads,
                block_size=block_size,
            )[0]
        return attn_out

