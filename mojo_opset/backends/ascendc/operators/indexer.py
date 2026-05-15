import torch

from mojo_opset.core import (
    MojoQuantLightningIndexer,
    MojoIndexerCompressEpilog,
    MojoKvCompressEpilog,
    MojoSparseAttnSharedkv,
    MojoSparseAttnSharedkvMetadata,
    MojoKvQuantSparseAttnSharedkv,
    MojoKvQuantSparseAttnSharedkvMetadata,
    MojoQuantLightningIndexerMetadata,
)
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class AscendcQuantLightningIndexer(MojoQuantLightningIndexer):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_dequant_scale: torch.Tensor,
        key_dequant_scale: torch.Tensor,
        query_quant_mode: int,
        key_quant_mode: int,
        *,
        actual_seq_lengths_query: torch.Tensor = None,
        actual_seq_lengths_key: torch.Tensor = None,
        block_table: torch.Tensor = None,
        metadata: torch.Tensor = None,
        layout_query: str = "BSND",
        layout_key: str = "PA_BSND",
        sparse_count: int = 2048,
        sparse_mode: int = 3,
        pre_tokens: int = 9223372036854775807,
        next_tokens: int = 9223372036854775807,
        cmp_ratio: int = 1,
        return_value: bool = False,
    ):
        try:
            import torch_npu

            if hasattr(torch_pu, "npu_quant_lightning_indexer"):
                print('[AscendcQuantLightningIndexer] Calling torch_npu.npu_quant_lightning_indexer')
                print('nnnnnnnnnn')
                return torch_npu.npu_quant_lightning_indexer(
                    query, key, weights, query_dequant_scale, key_dequant_scale,
                    query_quant_mode, key_quant_mode,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    block_table=block_table,
                    metadata=metadata,
                    layout_query=layout_query,
                    layout_key=layout_key,
                    sparse_count=sparse_count,
                    sparse_mode=sparse_mode,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    cmp_ratio=cmp_ratio,
                    return_value=return_value,
                )
        except Exception:
            pass

        import custom_ops
        print('[AscendcQuantLightningIndexer] Calling torch.ops.custom.npu_quant_lightning_indexer')
        # Ensure tensors are contiguous as required by the operator
        # Use clone() to create a guaranteed contiguous copy
        query = query.clone().contiguous()
        key = key.clone().contiguous()
        weights = weights.clone().contiguous() if weights is not None else None
        query_dequant_scale = query_dequant_scale.clone().contiguous()
        key_dequant_scale = key_dequant_scale.clone().contiguous()
        return torch.ops.custom.npu_quant_lightning_indexer(
            query, key, weights, query_dequant_scale, key_dequant_scale,
            query_quant_mode, key_quant_mode,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            metadata=metadata,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            cmp_ratio=cmp_ratio,
            return_value=return_value,
        )


class AscendcIndexerCompressEpilog(MojoIndexerCompressEpilog):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        attn: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ):
        try:
            import torch_npu

            if hasattr(torch_npu, "indexer_compress_epilog"):
                print('[AscendcIndexerCompressEpilog] Calling torch_npu.indexer_compress_epilog')
                return torch_npu.indexer_compress_epilog(attn, x, residual, gamma)
        except Exception:
            pass

        try:
            print('[AscendcIndexerCompressEpilog] Calling torch.ops.custom.indexer_compress_epilog')
            return torch.ops.custom.indexer_compress_epilog(attn, x, residual, gamma)
        except Exception:
            print('[AscendcIndexerCompressEpilog] Fallback to reference implementation')
            logger.warning("AscendC IndexerCompressEpilog kernel not available, falling back to reference implementation.")
            return super().forward(attn, x, residual, gamma)


class AscendcKvCompressEpilog(MojoKvCompressEpilog):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        attn: torch.Tensor,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        kv_s: torch.Tensor = None,
        kv_z: torch.Tensor = None,
    ):
        try:
            import torch_npu

            if hasattr(torch_npu, "kv_compress_epilog"):
                print('[AscendcKvCompressEpilog] Calling torch_npu.kv_compress_epilog')
                return torch_npu.kv_compress_epilog(attn, x, residual, gamma, kv_s, kv_z)
        except Exception:
            pass

        try:
            print('[AscendcKvCompressEpilog] Calling torch.ops.custom.kv_compress_epilog')
            return torch.ops.custom.kv_compress_epilog(attn, x, residual, gamma, kv_s, kv_z)
        except Exception:
            print('[AscendcKvCompressEpilog] Fallback to reference implementation')
            logger.warning("AscendC KvCompressEpilog kernel not available, falling back to reference implementation.")
            return super().forward(attn, x, residual, gamma, kv_s, kv_z)


class AscendcSparseAttnSharedkvMetadata(MojoSparseAttnSharedkvMetadata):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        *,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_ori_kv: torch.Tensor = None,
        cu_seqlens_cmp_kv: torch.Tensor = None,
        seqused_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        batch_size: int = 0,
        max_seqlen_q: int = 0,
        max_seqlen_kv: int = 0,
        cmp_topk: int = 0,
        cmp_ratio: int = 0,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = 'BSND',
        layout_kv: str = 'PA_ND',
        has_ori_kv: bool = True,
        has_cmp_kv: bool = False,
        device: str = 'npu:0',
    ):
        logger.info("Calling torch.ops.custom.npu_sparse_attn_sharedkv_metadata")
        return torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            cmp_topk=cmp_topk,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=has_ori_kv,
            has_cmp_kv=has_cmp_kv,
            device=device,
        )


class AscendcSparseAttnSharedkv(MojoSparseAttnSharedkv):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        q: torch.Tensor,
        *,
        ori_kv: torch.Tensor = None,
        cmp_kv: torch.Tensor = None,
        ori_sparse_indices: torch.Tensor = None,
        cmp_sparse_indices: torch.Tensor = None,
        ori_block_table: torch.Tensor = None,
        cmp_block_table: torch.Tensor = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_ori_kv: torch.Tensor = None,
        cu_seqlens_cmp_kv: torch.Tensor = None,
        seqused_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        sinks: torch.Tensor = None,
        metadata: torch.Tensor = None,
        softmax_scale: float = 0,
        cmp_ratio: int = 0,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = "BSND",
        layout_kv: str = "PA_ND",
        return_softmax_lse: bool = False,
    ):
        try:
            print('[AscendcSparseAttnSharedkv] Calling torch.ops.custom.npu_sparse_attn_sharedkv')
            return torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                ori_sparse_indices=ori_sparse_indices,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=seqused_q,
                seqused_kv=seqused_kv,
                sinks=sinks,
                metadata=metadata,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                return_softmax_lse=return_softmax_lse,
            )
        except Exception as e:
            logger.warning("AscendC SparseAttnSharedkv kernel not available, falling back to reference implementation.")
            return super().forward(
                q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                ori_sparse_indices=ori_sparse_indices,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=seqused_q,
                seqused_kv=seqused_kv,
                sinks=sinks,
                metadata=metadata,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                return_softmax_lse=return_softmax_lse,
            )


class AscendcKvQuantSparseAttnSharedkv(MojoKvQuantSparseAttnSharedkv):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_s: torch.Tensor,
        kv_z: torch.Tensor,
        scale: float,
        causal: bool = False,
    ):
        try:
            import torch_npu

            if hasattr(torch_npu, "npu_kv_quant_sparse_attn_sharedkv"):
                print('[AscendcKvQuantSparseAttnSharedkv] Calling torch_npu.npu_kv_quant_sparse_attn_sharedkv')
                return torch_npu.npu_kv_quant_sparse_attn_sharedkv(query, key, value, kv_s, kv_z, scale, causal)
        except Exception:
            pass

        try:
            print('[AscendcKvQuantSparseAttnSharedkv] Calling torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv')
            return torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv(query, key, value, kv_s, kv_z, scale, causal)
        except Exception:
            print('[AscendcKvQuantSparseAttnSharedkv] Fallback to reference implementation')
            logger.warning("AscendC KvQuantSparseAttnSharedkv kernel not available, falling back to reference implementation.")
            return super().forward(query, key, value, kv_s, kv_z, scale, causal)


class AscendcKvQuantSparseAttnSharedkvMetadata(MojoKvQuantSparseAttnSharedkvMetadata):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        kv_quant_mode: int = 0,
        *,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_ori_kv: torch.Tensor = None,
        cu_seqlens_cmp_kv: torch.Tensor = None,
        seqused_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        batch_size: int = 0,
        max_seqlen_q: int = 0,
        max_seqlen_kv: int = 0,
        ori_topk: int = 0,
        cmp_topk: int = 0,
        tile_size: int = 0,
        rope_head_dim: int = 0,
        cmp_ratio: int = -1,
        ori_mask_mode: int = 4,
        cmp_mask_mode: int = 3,
        ori_win_left: int = 127,
        ori_win_right: int = 0,
        layout_q: str = "BSND",
        layout_kv: str = "PA_ND",
        has_ori_kv: bool = True,
        has_cmp_kv: bool = True,
    ):
        try:
            import torch_npu

            if hasattr(torch_npu, "npu_kv_quant_sparse_attn_sharedkv_metadata"):
                print('[AscendcKvQuantSparseAttnSharedkvMetadata] Calling torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata')
                return torch_npu.npu_kv_quant_sparse_attn_sharedkv_metadata(
                    num_heads_q, num_heads_kv, head_dim, kv_quant_mode,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    seqused_q=seqused_q,
                    seqused_kv=seqused_kv,
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    ori_topk=ori_topk,
                    cmp_topk=cmp_topk,
                    tile_size=tile_size,
                    rope_head_dim=rope_head_dim,
                    cmp_ratio=cmp_ratio,
                    ori_mask_mode=ori_mask_mode,
                    cmp_mask_mode=cmp_mask_mode,
                    ori_win_left=ori_win_left,
                    ori_win_right=ori_win_right,
                    layout_q=layout_q,
                    layout_kv=layout_kv,
                    return_softmax_lse=False
                )
        except Exception:
            pass

        try:
            print('[AscendcKvQuantSparseAttnSharedkvMetadata] Calling torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata')
            return torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
                num_heads_q, num_heads_kv, head_dim, kv_quant_mode,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=seqused_q,
                seqused_kv=seqused_kv,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                ori_topk=ori_topk,
                cmp_topk=cmp_topk,
                tile_size=tile_size,
                rope_head_dim=rope_head_dim,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=has_ori_kv,
                has_cmp_kv=has_cmp_kv,
            )
        except Exception as e:
            print(f'[AscendcKvQuantSparseAttnSharedkvMetadata] Fallback to reference implementation: {e}')
            logger.warning("AscendC KvQuantSparseAttnSharedkvMetadata kernel not available, falling back to reference implementation.")
            return super().forward(
                num_heads_q, num_heads_kv, head_dim, kv_quant_mode,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=seqused_q,
                seqused_kv=seqused_kv,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                ori_topk=ori_topk,
                cmp_topk=cmp_topk,
                tile_size=tile_size,
                rope_head_dim=rope_head_dim,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=has_ori_kv,
                has_cmp_kv=has_cmp_kv,
            )


class AscendcQuantLightningIndexerMetadata(MojoQuantLightningIndexerMetadata):
    supported_platforms_list = ["npu", "meta_device"]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_dequant_scale: torch.Tensor,
        key_dequant_scale: torch.Tensor,
        query_quant_mode: int,
        key_quant_mode: int,
        metadata: torch.Tensor = None,
        *,
        actual_seq_lengths_query: torch.Tensor = None,
        actual_seq_lengths_key: torch.Tensor = None,
        block_table: torch.Tensor = None,
        layout_query: str = "BSND",
        layout_key: str = "PA_BSND",
        sparse_count: int = 2048,
        sparse_mode: int = 3,
        pre_tokens: int = 9223372036854775807,
        next_tokens: int = 9223372036854775807,
        cmp_ratio: int = 1,
        return_value: bool = False,
    ):
        # Calculate batch_size and head dimensions from input tensors
        batch_size = query.shape[0]
        num_heads_q = query.shape[2]
        head_dim = query.shape[3]
        
        # Get key heads from key tensor
        if key.dim() == 4:
            num_heads_k = key.shape[2]
        else:
            num_heads_k = 1
        
        # Calculate max sequence lengths
        if actual_seq_lengths_query is not None:
            max_seqlen_q = actual_seq_lengths_query.max().item()
        else:
            max_seqlen_q = query.shape[1]
            
        if actual_seq_lengths_key is not None:
            max_seqlen_k = actual_seq_lengths_key.max().item()
        else:
            if key.dim() == 4 and layout_key == "PA_BSND":
                max_seqlen_k = key.shape[0] * key.shape[1]
            else:
                max_seqlen_k = key.shape[1]

        try:
            import torch_npu

            if hasattr(torch_npu, "npu_quant_lightning_indexer_metadata"):
                print('[AscendcQuantLightningIndexerMetadata] Calling torch_npu.npu_quant_lightning_indexer_metadata')
                return torch_npu.npu_quant_lightning_indexer_metadata(
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    num_heads_q=num_heads_q,
                    num_heads_k=num_heads_k,
                    head_dim=head_dim,
                    query_quant_mode=query_quant_mode,
                    key_quant_mode=key_quant_mode,
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    layout_query=layout_query,
                    layout_key=layout_key,
                    sparse_count=sparse_count,
                    sparse_mode=sparse_mode,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    cmp_ratio=cmp_ratio,
                    device='npu:0'
                )
        except Exception as e:
            print(f'[AscendcQuantLightningIndexerMetadata] torch_npu failed: {e}')
            pass

        try:
            print('[AscendcQuantLightningIndexerMetadata] Calling torch.ops.custom.npu_quant_lightning_indexer_metadata')
            return torch.ops.custom.npu_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                num_heads_q=num_heads_q,
                num_heads_k=num_heads_k,
                head_dim=head_dim,
                query_quant_mode=query_quant_mode,
                key_quant_mode=key_quant_mode,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                layout_query=layout_query,
                layout_key=layout_key,
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                cmp_ratio=cmp_ratio,
                device='npu:0'
            )
        except Exception as e:
            print(f'[AscendcQuantLightningIndexerMetadata] Fallback to reference implementation: {e}')
            logger.warning("AscendC QuantLightningIndexerMetadata kernel not available, falling back to reference implementation.")
            return super().forward(
                query, key, weights, query_dequant_scale, key_dequant_scale,
                query_quant_mode, key_quant_mode, metadata,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                layout_query=layout_query,
                layout_key=layout_key,
                sparse_count=sparse_count,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                cmp_ratio=cmp_ratio,
                return_value=return_value,
            )
