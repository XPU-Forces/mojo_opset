import torch
import torch.nn.functional as F

from ..operator import MojoOperator


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)

    return pre, post, comb


class MojoHcPre(MojoOperator):
    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.hidden_size = hidden_size
        mix_size = (2 + hc_mult) * hc_mult
        weight_kwargs = {**self.tensor_factory_kwargs, "dtype": torch.float32}
        self.weight = torch.nn.Parameter(torch.empty(mix_size, hc_mult * hidden_size, **weight_kwargs))
        self.scale = torch.nn.Parameter(torch.empty(3, **weight_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(mix_size, **weight_kwargs))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Project ``hidden_states[..., hc_mult, hidden_size]`` into one branch."""
        hidden_states_fp32 = hidden_states.float()
        flattened_states = hidden_states_fp32.flatten(-2)
        rsqrt = torch.rsqrt(flattened_states.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flattened_states, self.weight.float()) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            self.scale.float(),
            self.bias.float(),
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )
        output = torch.sum(pre.unsqueeze(-1) * hidden_states_fp32, dim=-2)
        return output.to(hidden_states.dtype).contiguous(), post.contiguous(), comb.contiguous()


class MojoHcPost(MojoOperator):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_weights: torch.Tensor,
        combination_weights: torch.Tensor,
    ):
        """Mix the HC branch output back into ``residual[..., hc, D]``."""
        hidden_states_fp32 = hidden_states.float()
        residual_fp = residual.float()
        output = post_weights.float().unsqueeze(-1) * hidden_states_fp32.unsqueeze(-2) + torch.sum(
            combination_weights.float().unsqueeze(-1) * residual_fp.unsqueeze(-2), dim=hidden_states.dim() - 1
        )
        return output.to(hidden_states.dtype).contiguous()
