import torch
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoHcPre(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ):
        """
        HC Pre operator.

        Args:
            x: Input tensor of shape [bs, hc_mult, d] or [b, s, hc_mult, d].
            hc_fn: Projection weight tensor of shape [hc_mix, hc_mult * d].
            hc_scale: Scale tensor of shape [3].
            hc_base: Bias tensor of shape [hc_mix].
            hc_mult: HC multiplier. The AscendC kernel currently supports 4.
            hc_sinkhorn_iters: Number of sinkhorn normalization iterations.
            norm_eps: Epsilon used by the inverse RMS normalization.
            hc_eps: Epsilon used by HC pre/post/comb computation.

        Returns:
            A tuple of (y, post, comb), where y has the same dtype as x and
            post/comb are float tensors.
        """
        shape = x.size()
        dtype = x.dtype
        x_fp = x.float()

        if x.dim() == 4:
            x_flat = x_fp.flatten(2)
        elif x.dim() == 3:
            x_flat = x_fp.flatten(1)
        else:
            raise ValueError(f"Input x should be 3D or 4D, got {x.dim()}D.")

        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            hc_scale.float(),
            hc_base.float(),
            hc_mult=hc_mult,
            sinkhorn_iters=hc_sinkhorn_iters,
            eps=hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * x_fp.view(shape), dim=-2)
        return y.to(dtype).contiguous(), post.contiguous(), comb.contiguous()


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
