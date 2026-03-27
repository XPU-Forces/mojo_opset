import torch

from mojo_opset.backends.ttx.kernels import indexer_rope

from mojo_opset.core import MojoIndexerRoPE


class TTXIndexerRoPE(MojoIndexerRoPE):
    supported_platforms_list = ["npu"]
    
    def forward(
        self,
        q : torch.Tensor,
        k : torch.Tensor,
        cos : torch.Tensor,
        sin : torch.Tensor,
        rope_head_dim : int = None
    )-> tuple[torch.Tensor, torch.Tensor]:
        assert q.is_contiguous()
        assert k.is_contiguous()
        batch_size, seq_len, n_q_head, head_dim = q.shape
        assert rope_head_dim % 2 == 0, f"rope_head_dim must be even, got {rope_head_dim}"
        assert rope_head_dim <= head_dim, f"rope_head_dim ({rope_head_dim}) must be <= head_dim ({head_dim})"

        cos_batch_size = cos.shape[0]
        sin_batch_size = sin.shape[0]
        assert cos_batch_size == sin_batch_size, "cos and sin must have same batch size"
        assert cos.shape[1] == seq_len, f"cos seq_len mismatch: {cos.shape[1]} vs {seq_len}"
        assert sin.shape[1] == seq_len, f"sin seq_len mismatch: {sin.shape[1]} vs {seq_len}"
        assert cos.shape[2] == rope_head_dim // 2, f"cos dim mismatch: {cos.shape[2]} vs {rope_head_dim // 2}"
        assert sin.shape[2] == rope_head_dim // 2, f"sin dim mismatch: {sin.shape[2]} vs {rope_head_dim // 2}"

        return indexer_rope(q, k, cos, sin, rope_head_dim)
