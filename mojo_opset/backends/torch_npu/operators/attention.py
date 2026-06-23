from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch_npu


from mojo_opset.core import MojoPagedDecodeSWA
from mojo_opset.core import MojoPagedPrefillSWA
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPrefillGQA
from mojo_opset.core.operators.attention import assert_paged_decode_contract
from mojo_opset.core.operators.attention import assert_paged_prefill_contract


class TorchNpuPrefillGQA(MojoPrefillGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout)

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, block_size, _ = k_cache.shape


        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel requires head_dim % 128 == 0, got {query.shape[-1]}")

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return super().forward(query, k_cache, v_cache, cu_q_lens, softmax_scale)

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        atten_mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool, device=query.device), diagonal=1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_cache,
            value=v_cache,
            actual_seq_lengths=cu_q_lens,
            num_heads=num_q_heads,
            input_layout="BSND",
            scale=softmax_scale,
            pre_tokens=65535,
            next_tokens=0,
            sparse_mode=2,
            num_key_value_heads=num_kv_heads,
            atten_mask=atten_mask,
        )
        return out


class TorchNpuPagedPrefillGQA(MojoPagedPrefillGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout)

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        _, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = key_cache.shape
        total_seq_lens = (
            cu_q_lens[1:] - cu_q_lens[:-1]
            if cu_total_seq_lens is None
            else cu_total_seq_lens[1:] - cu_total_seq_lens[:-1]
        )
        
        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")
        if cu_total_seq_lens is not None:
            raise NotImplementedError("NPU kernel npu_fused_infer_attention_score currently does not support TND layout with sparse_mode=3 (Page Attention), raising RuntimeError: call aclnnFusedInferAttentionScoreV3 failed.")


        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return super().forward(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                cu_q_lens=cu_q_lens,
                block_tables=block_tables,
                softmax_scale=softmax_scale,
                cu_total_seq_lens=cu_total_seq_lens,
                mask=mask,
                max_q_len=max_q_len,
                max_total_seq_len=max_total_seq_len,
            )

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        compress_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=query.device), diagonal=1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key_cache,
            value=value_cache,
            atten_mask=compress_mask,
            block_table=block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=cu_q_lens[1:],
            actual_seq_lengths_kv=total_seq_lens,
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=softmax_scale,
            sparse_mode=3,
        )
        return out


class TorchNpuPagedDecodeGQA(MojoPagedDecodeGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout)

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        input_layout: Optional[str] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ) -> Tuple[Any]:
        batch_size, num_q_heads, head_dim = query.shape
        _, head_nums, block_size, _ = k_cache.shape
        assert_paged_decode_contract(block_tables, total_seq_lens)
        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")

        if block_size % 128 != 0 or block_size > 512:
            return super().forward(
                query,
                k_cache,
                v_cache,
                total_seq_lens,
                block_tables,
                softmax_scale=softmax_scale,
                mask=mask,
                max_total_seq_len=max_total_seq_len,
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        is_unsqueezed = False
        if input_layout is None:
            if query.dim() == 3:
                query = query.unsqueeze(2)
                input_layout = "BNSD"
                is_unsqueezed = True
            else:
                input_layout = "BNSD"

        # actual_seq_lengths_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=query.device)
        actual_seq_lengths_q = torch.ones(batch_size, dtype=torch.int32, device=query.device)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            k_cache,
            v_cache,
            input_layout=input_layout,
            block_table=block_tables,
            block_size=block_size,
            num_heads=num_q_heads,
            num_key_value_heads=head_nums,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=total_seq_lens,
            scale=softmax_scale,
        )

        if is_unsqueezed:
            out = out.squeeze(2)
        return out


def _generate_window_mask(
    q_seq_len: int,
    kv_seq_len: int,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
) -> torch.Tensor:
    kv_computed_len = kv_seq_len - q_seq_len
    causal_mask = (torch.arange(0, q_seq_len)[:, None] + kv_computed_len) >= torch.arange(0, kv_seq_len)[None, :]
    if local_window_size is not None or global_window_size is not None:
        local_window_mask = (
            (
                torch.arange(kv_computed_len, kv_computed_len + q_seq_len)[:, None]
                <= torch.arange(0, kv_seq_len)[None, :] + local_window_size
            )
            if local_window_size is not None
            else False
        )
        global_window_mask = (
            (torch.arange(0, kv_seq_len) < global_window_size)[None, :] if global_window_size is not None else False
        )
        mask = causal_mask & (local_window_mask | global_window_mask)
    else:
        mask = causal_mask

    return mask


def tnd_to_bnsd(tnd_tensor, cu_q_lens, max_seq_len):
    """
    Converrt and padding from TND tensor to BNSD tensor
    tnd_tensor: [total_tokens, num_heads, head_dim]
    cu_q_lens: [bsz + 1]
    max_seq_len: max(total_seq_lens)
    return: [bsz, num_heads, max_seq_len, head_dim]
    """
    bsz = cu_q_lens.shape[0] - 1
    _, num_heads, head_dim = tnd_tensor.shape
    
    bnsd_tensor = torch.zeros(
        bsz, num_heads, max_seq_len, head_dim, 
        dtype=tnd_tensor.dtype, 
        device=tnd_tensor.device
    )
    
    for i in range(bsz):
        start = cu_q_lens[i]
        end = cu_q_lens[i + 1]
        seq_len = end - start
        bnsd_tensor[i, :, :seq_len, :] = tnd_tensor[start:end].permute(1, 0, 2)

    return bnsd_tensor


def bnsd_to_tnd(bnsd_tensor, cu_q_lens):
    """
    Convert and remove padding from BNSD tensor to TND tensor
    bnsd_tensor: [bsz, num_heads, max_seq_len, head_dim]
    cu_q_lens: [bsz + 1]
    return: [total_tokens, num_heads, head_dim]
    """
    bsz, num_heads, _, head_dim = bnsd_tensor.shape
    total_tokens = cu_q_lens[-1].item()
    
    tnd_tensor = torch.zeros(
        total_tokens, num_heads, head_dim,
        dtype=bnsd_tensor.dtype,
        device=bnsd_tensor.device
    )
    
    for i in range(bsz):
        start = cu_q_lens[i]
        end = cu_q_lens[i + 1]
        seq_len = end - start
        sliced_data = bnsd_tensor[i, :, :seq_len, :]
        tnd_tensor[start:end] = sliced_data.permute(1, 0, 2)
        
    return tnd_tensor


class TorchNpuPagedPrefillSWA(MojoPagedPrefillSWA, default_priority=0):
    def forward(
        self,
        query: torch.Tensor,  # [total_q_len, n_q_heads, head_dim]
        key_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        value_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        cu_q_lens: torch.Tensor,  # [bsz + 1]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,  # [bsz + 1]
        *,
        max_q_len: Optional[int] = None,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:

        _, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = key_cache.shape

        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")

        q_seq_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        total_seq_lens = q_seq_lens if cu_total_seq_lens is None else cu_total_seq_lens[1:] - cu_total_seq_lens[:-1]

        # for comparing with ascendc
        # if block_size % 128 != 0 or block_size > 512:
        if cu_total_seq_lens is not None and not torch.equal(cu_q_lens, cu_total_seq_lens):
            print(f"[Warning] NPU kernel npu_fused_infer_attention_score don't support 'seq_kv != seq_q' temporarily")
            return super().forward(
                query,
                key_cache,
                value_cache,
                cu_q_lens,
                block_table,
                softmax_scale=softmax_scale,
                cu_total_seq_lens=cu_total_seq_lens,
                max_q_len=max_q_len,
                max_total_seq_len=max_total_seq_len,
            )

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        max_seq_len = max_q_len if max_q_len is not None else q_seq_lens.max().item()
        max_total_seq_lens = max_total_seq_len if max_total_seq_len else total_seq_lens.max().item()

        block_table_max_kv_len = block_table.shape[1] * block_size
        mask_kv_len = max(max_total_seq_len, block_table_max_kv_len)

        # convert query from tnd to bnsd
        query_bnsd = tnd_to_bnsd(query, cu_q_lens, max_seq_len)
        batch_size, _, _, _ = query_bnsd.shape

        attn_mask = ~(_generate_window_mask(mask_kv_len, mask_kv_len, self.local_window_size, self.global_window_size).to(query.device))
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query_bnsd,
            input_layout="BNSD",
            key=key_cache,
            value=value_cache,
            block_table=block_table,
            block_size=block_size,
            actual_seq_lengths=q_seq_lens,
            actual_seq_lengths_kv=total_seq_lens,
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=softmax_scale,
            atten_mask=attn_mask,
            sparse_mode=0, # seq_len mask for mode 0, support mask
        )
        # convert output from bnsd to tnd
        out_tnd = bnsd_to_tnd(out, cu_q_lens)
        return out_tnd


class TorchNpuPagedDecodeSWA(MojoPagedDecodeSWA, default_priority=0):
    def forward(
        self,
        query: torch.Tensor,  # [bsz, n_q_heads, head_dim]
        key_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        value_cache: torch.Tensor,  # [n_pages, n_kv_heads, page_size, head_dim]
        total_seq_lens: torch.Tensor,  # [bsz]
        block_table: torch.Tensor,  # [bsz, max_num_blocks]
        softmax_scale: Optional[float] = None,
        input_layout: Optional[str] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, num_q_heads, head_dim = query.shape
        _, head_nums, block_size, _ = key_cache.shape

        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")

        if block_size % 128 != 0 or block_size > 512:
            return super().forward(
                query,
                key_cache,
                value_cache,
                total_seq_lens,
                block_table,
                softmax_scale=softmax_scale,
                max_total_seq_len=max_total_seq_len,
            )

        max_total_seq_len = max_total_seq_len if max_total_seq_len else total_seq_lens.max().item()
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        is_unsqueezed = False
        if input_layout is None:
            if query.dim() == 3:
                query = query.unsqueeze(2) # (B, N, D) -> (B, N, 1, D)
                input_layout = "BNSD"
                is_unsqueezed = True
            else:
                input_layout = "BNSD"

        actual_seq_lengths_q = torch.ones(batch_size, dtype=torch.int32, device=query.device)
        
        block_table_max_kv_len = block_table.shape[1] * block_size
        mask_kv_len = max(max_total_seq_len, block_table_max_kv_len)

        attn_mask = ~(_generate_window_mask(1, mask_kv_len, self.local_window_size, self.global_window_size).to(query.device))
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key_cache,
            value=value_cache,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=total_seq_lens,
            num_key_value_heads=head_nums,
            num_heads=num_q_heads,
            scale=softmax_scale,
            atten_mask=attn_mask,
            sparse_mode=0, # seq_len mask for mode 0, support mask
        )

        if is_unsqueezed:
            out = out.squeeze(2)
        return out
