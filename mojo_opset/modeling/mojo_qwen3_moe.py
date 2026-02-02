import torch
from mojo_opset import (
    MojoEmbedding,
    MojoMoEGating,
    MojoMoEDispatch,
    MojoMoECombine,
    MojoLinear,
    MojoNorm,
)
from mojo_opset.core.attn.mojo_prefill_gqa import MojoPrefillGQA
from mojo_opset.core.linear.mojo_linear import MojoGroupLinear


class MojoQwen3MoeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = MojoEmbedding(
            vocab_size=10000,
            embedding_dim=4096,
        )
        self.qkv_proj = MojoLinear()
        self.pre_norm = MojoNorm()
        self.attn = MojoPrefillGQA(
            hidden_size=4096,
            num_heads=32,
            head_dim=128,
        )
        self.post_norm = MojoNorm()
        self.moe_gate = MojoMoEGating(
            in_features=4096,
            num_experts=8,
        )

        self.moe_dispatch = MojoMoEDispatch()
        self.moe_gmm = MojoGroupLinear(
            in_features=4096,
            out_features=4096,
            num_groups=8,
        )
        self.moe_combine = MojoMoECombine()

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.embedding(input_ids)
        hidden_states = self.qkv_proj(hidden_states)
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = self.post_norm(hidden_states)
        gate_logits = self.moe_gate(hidden_states)
        dispatch_indices, dispatch_weights = self.moe_dispatch(gate_logits)
        hidden_states = self.moe_gmm(hidden_states, dispatch_indices, dispatch_weights)
        hidden_states = self.moe_combine(hidden_states, dispatch_indices, dispatch_weights)

        return hidden_states
