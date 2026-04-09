import torch

from mojo_opset.backends.ttx.kernels import relative_embedding_fwd_impl
from mojo_opset.core import MojoRelativeEmbedding


class TTXRelativeEmbedding(MojoRelativeEmbedding):
    supported_platforms_list = ["ilu"]

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        if not isinstance(lq, int) or not isinstance(lk, int) or lq <= 0 or lk <= 0:
            raise ValueError("lq and lk must be positive integers")
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        bucket = self._relative_position_bucket(rel_pos)
        return relative_embedding_fwd_impl(bucket, self.embedding.weight)
