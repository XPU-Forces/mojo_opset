import torch

from ..operator import MojoOperator


class MojoHcPost(MojoOperator):
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        """
        HC Post operator: combines x, residual, post, and comb tensors.

        Supports two input modes:
        1. 3D mode: x is [b, s, d], residual is [b, s, hc, d],
           post is [b, s, hc], comb is [b, s, hc, hc]
        2. 2D mode: x is [bs, d], residual is [bs, hc, d],
           post is [bs, hc], comb is [bs, hc, hc]

        Computes:
            out = post.unsqueeze(-1) * x.unsqueeze(-2) +
                  sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=x.dim() - 1)

        Args:
            x: Input tensor of shape [bs, d] (2D) or [b, s, d] (3D).
            residual: Residual tensor of shape [bs, hc, d] (3D) or [b, s, hc, d] (4D).
            post: Post tensor of shape [bs, hc] (2D) or [b, s, hc] (3D).
            comb: Combination tensor of shape [bs, hc, hc] (3D) or [b, s, hc, hc] (4D).

        Returns:
            Output tensor with the same shape as residual.
        """
        print('qqqq')
        data_type = x.dtype
        x_fp = x.float()
        residual_fp = residual.float()
        post_fp = post.float()
        comb_fp = comb.float()
        out = post_fp.unsqueeze(-1) * x_fp.unsqueeze(-2) + torch.sum(
            comb_fp.unsqueeze(-1) * residual_fp.unsqueeze(-2), dim=x.dim() - 1
        )
        out = out.to(data_type).contiguous()
        return out