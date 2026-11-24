import torch

from mojo_opset import functions

from .normalization import LayerNormInfer


class LightningIndexer(torch.nn.Module):
    def __init__(self, *, implementation=None):
        super().__init__()
        self.implementation = implementation

    def forward(self, query, query_scale, key, key_scale=None):
        return functions.lightning_indexer(query, query_scale, key, key_scale, implementation=self.implementation)


class Indexer(torch.nn.Module):
    def __init__(
        self,
        dim=7168,
        n_heads=128,
        head_dim=128,
        qk_rope_head_dim=64,
        topk=2048,
        q_lora_rank=1536,
        max_batch_size=128,
        max_seq_len=32768,
        *,
        implementation=None,
    ):
        super().__init__()
        self.dim, self.n_heads, self.head_dim = dim, n_heads, head_dim
        self.rope_head_dim, self.topk, self.q_lora_rank = qk_rope_head_dim, topk, q_lora_rank
        self.softmax_scale, self.implementation = head_dim**-0.5, implementation
        self.wq_b = torch.nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = torch.nn.Linear(dim, head_dim, bias=False)
        self.k_norm = LayerNormInfer(head_dim, implementation=implementation)
        self.weights_proj = torch.nn.Linear(dim, n_heads, bias=False)
        self.register_buffer(
            "k_cache", torch.zeros(max_batch_size, max_seq_len, head_dim, dtype=torch.int8), persistent=False
        )
        self.register_buffer(
            "k_scale_cache", torch.zeros(max_batch_size, max_seq_len, dtype=torch.float32), persistent=False
        )

    @torch.no_grad()
    def forward(self, x, qr, start_pos, freqs_cis, mask=None):
        return functions.indexer(
            x,
            qr,
            self.wq_b.weight,
            self.wk.weight,
            self.k_norm.weight,
            self.k_norm.bias,
            self.weights_proj.weight,
            self.k_cache,
            self.k_scale_cache,
            start_pos,
            freqs_cis,
            mask,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            topk=self.topk,
            norm_eps=self.k_norm.eps,
            implementation=self.implementation,
        )

    def extra_repr(self):
        return (
            f"dim={self.dim}, n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"rope_head_dim={self.rope_head_dim}, topk={self.topk}, q_lora_rank={self.q_lora_rank}"
        )
