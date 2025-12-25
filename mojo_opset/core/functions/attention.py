import torch
import torch.nn.functional as F

from mojo_opset.core.function import MojoFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoSdpaFunction(MojoFunction):
    @staticmethod
    def forward_ref(ctx, query, key, value, mask, scale=1.0, enable_gqa=False):
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa
        ctx.scale = scale
        score = torch.matmul(query, key.transpose(-1, -2)).to(torch.float32) * scale
        score.masked_fill_(mask == 0, float("-inf"))
        p = F.softmax(score - torch.max(score, dim=-1, keepdim=True).values, dim=-1)
        output = torch.matmul(p.to(query.dtype), value)
        ctx.save_for_backward(query, key, value, output, mask)
        return output

    def backward_ref(ctx, do):
        query, key, value, output, mask = ctx.saved_tensors
        score = torch.matmul(query, key.transpose(-1, -2)).to(torch.float32) * ctx.scale
        score.masked_fill_(mask == 0, float("-inf"))
        p = F.softmax(score - torch.max(score, dim=-1, keepdim=True).values, dim=-1)
        dv = torch.matmul(p.transpose(-1, -2).to(query.dtype), do)
        dp = torch.matmul(do, value.transpose(-1, -2))
        ds = p * (dp - torch.sum(do * output, dim=-1, keepdim=True))
        dq = torch.matmul(ds.to(query.dtype), key) * ctx.scale
        dk = torch.matmul(ds.to(query.dtype).transpose(-1, -2), query) * ctx.scale

<<<<<<< HEAD
        return dq, dk, dv, None
=======
        return dq, dk, dv
>>>>>>> d73744c (add sdpa function interface and fix some)
