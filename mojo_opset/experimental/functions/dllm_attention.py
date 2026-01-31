import torch
import torch.nn.functional as F

from mojo_opset.core.function import MojoFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class MojoDllmAttentionFunction(MojoFunction):
    """
    MojoDllmAttentionFunction implements the specific attention for text diffusion.
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlen: torch.Tensor,
        scale: float = 1.0,
        BLOCK_SIZE: int = 8,
    ) -> torch.Tensor:
        """
        Forward pass for diffusion attention.

        Args:
            ctx: Context object for the backward.
            query (torch.Tensor): Query tensor. shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
            key (torch.Tensor): Key tensor. shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]
            value (torch.Tensor): Value tensor. shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]
            cu_seqlen (torch.Tensor): Cumulative sequence lengths tensor without leading zero. shape [BSZ]
            scale (float, optional): Scale factor for attention. Defaults to 1.0.
            BLOCK_SIZE (int, optional): Block size for block diffusion attention. Defaults to 8.

        Returns:
            torch.Tensor: Output tensor after attention.
        """

        q = query
        k = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        v = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
        o = torch.zeros_like(query)
        lse = torch.zeros([query.shape[0], query.shape[1]], device=query.device, dtype=torch.float32)

        idx = torch.arange(8192, device="npu") // BLOCK_SIZE
        mask_ul = idx[:, None] == idx[None, :]
        mask_ur = idx[:, None] > idx[None, :]
        mask_dl = torch.zeros([8192, 8192], device="npu", dtype=torch.bool)
        mask_dr = idx[:, None] >= idx[None, :]
        S = q.shape[0] // 2

        st = 0
        for i in range(cu_seqlen.shape[0]):
            ed = cu_seqlen[i].cpu().item()
            tile_q = torch.concat([q[st:ed, :, :], q[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_k = torch.concat([k[st:ed, :, :], k[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_v = torch.concat([v[st:ed, :, :], v[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_mask_u = torch.concat([mask_ul[: ed - st, : ed - st], mask_ur[: ed - st, : ed - st]], dim=1)
            tile_mask_d = torch.concat([mask_dl[: ed - st, : ed - st], mask_dr[: ed - st, : ed - st]], dim=1)
            tile_mask = torch.concat([tile_mask_u, tile_mask_d], dim=0)

            s = torch.matmul(tile_q, tile_k.transpose(-1, -2)).to(torch.float32) * scale
            s.masked_fill_(tile_mask == 0, float("-inf"))
            tile_lse = torch.logsumexp(s, dim=-1)
            p = F.softmax(s - torch.max(s, dim=-1, keepdim=True).values, dim=-1)
            tile_o = torch.matmul(p.to(torch.bfloat16), tile_v)

            o[st:ed, :, :] = tile_o[:, : ed - st, :].permute(1, 0, 2)
            o[S + st : S + ed, :, :] = tile_o[:, ed - st : 2 * (ed - st), :].permute(1, 0, 2)
            lse[st:ed, :] = tile_lse[:, : ed - st].permute(1, 0)
            lse[S + st : S + ed, :] = tile_lse[:, ed - st : 2 * (ed - st)].permute(1, 0)

            st = ed

        ctx.save_for_backward(query, key, value, o, lse, cu_seqlen, scale, BLOCK_SIZE)

        return o

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        Backward pass for diffusion attention.

        Args:
            ctx: Context object for the backward.
            grad_output (torch.Tensor): Gradient of the output tensor. shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]

        Returns:
            tuple: Gradients of query, key, value, None, None, None.
                grad_query: shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
                grad_key: shape [BSZ, K_HEAD_NUM, SEQ, HEAD_DIM]
                grad_value: shape [BSZ, V_HEAD_NUM, SEQ, HEAD_DIM]
        """
        query, key, value, o, lse, cu_seqlen, scale, BLOCK_SIZE = ctx.saved_tensors

        dq = torch.zeros_like(query)
        dk = torch.zeros_like(key)
        dv = torch.zeros_like(value)
        q = query
        k = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        v = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)

        idx = torch.arange(8192, device="npu:0") // BLOCK_SIZE
        mask_ul = idx[:, None] == idx[None, :]
        mask_ur = idx[:, None] > idx[None, :]
        mask_dl = torch.zeros([8192, 8192], device="npu", dtype=torch.bool)
        mask_dr = idx[:, None] >= idx[None, :]
        S = q.shape[0] // 2

        num_group = key.shape[1]
        group_size = query.shape[1] // num_group
        H = query.shape[2]

        st = 0
        for i in range(cu_seqlen.shape[0]):
            ed = cu_seqlen[i].cpu().item()
            tile_q = torch.concat([q[st:ed, :, :], q[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_lse = torch.concat([lse[st:ed, :], lse[S + st : S + ed, :]], dim=0).permute(1, 0)
            tile_k = torch.concat([k[st:ed, :, :], k[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_v = torch.concat([v[st:ed, :, :], v[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_do = torch.concat([grad_output[st:ed, :, :], grad_output[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_o = torch.concat([o[st:ed, :, :], o[S + st : S + ed, :, :]], dim=0).permute(1, 0, 2)
            tile_mask_u = torch.concat([mask_ul[: ed - st, : ed - st], mask_ur[: ed - st, : ed - st]], dim=1)
            tile_mask_d = torch.concat([mask_dl[: ed - st, : ed - st], mask_dr[: ed - st, : ed - st]], dim=1)
            tile_mask = torch.concat([tile_mask_u, tile_mask_d], dim=0)

            s = torch.matmul(tile_q, tile_k.transpose(-1, -2)).to(torch.float32) * scale
            s.masked_fill_(tile_mask == 0, float("-inf"))
            p = torch.exp(s - tile_lse[:, :, None])
            # p = F.softmax(s - torch.max(s, dim=-1, keepdim=True).values, dim=-1)
            tile_dv = torch.matmul(p.transpose(-1, -2).to(torch.bfloat16), tile_do)
            dp = torch.matmul(tile_do, tile_v.transpose(-1, -2))
            ds = p * (dp - torch.sum(tile_do * tile_o, dim=-1, keepdim=True))
            tile_dq = torch.matmul(ds.to(torch.bfloat16), tile_k) * scale
            tile_dk = torch.matmul(ds.to(torch.bfloat16).transpose(-1, -2), tile_q) * scale

            dq[st:ed, :, :] = tile_dq[:, : ed - st, :].permute(1, 0, 2)
            dk[st:ed, :, :] = (
                tile_dk[:, : ed - st, :].permute(1, 0, 2).view(ed - st, num_group, group_size, H).sum(dim=2)
            )
            dv[st:ed, :, :] = (
                tile_dv[:, : ed - st, :].permute(1, 0, 2).view(ed - st, num_group, group_size, H).sum(dim=2)
            )
            dq[S + st : S + ed, :, :] = tile_dq[:, ed - st : 2 * (ed - st), :].permute(1, 0, 2)
            dk[S + st : S + ed, :, :] = (
                tile_dk[:, ed - st : 2 * (ed - st), :]
                .permute(1, 0, 2)
                .view(ed - st, num_group, group_size, H)
                .sum(dim=2)
            )
            dv[S + st : S + ed, :, :] = (
                tile_dv[:, ed - st : 2 * (ed - st), :]
                .permute(1, 0, 2)
                .view(ed - st, num_group, group_size, H)
                .sum(dim=2)
            )

            st = ed

        return dq, dk, dv, None, None, None


def mojo_dllm_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float = 1.0,
    BLOCK_SIZE: int = 8,
) -> torch.Tensor:
    """
    Applies the diffusion-specific attention mechanism to the input tensors.

    This is a functional wrapper for the `MojoDiffusionAttentionFunction`.

    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        cu_seqlen (torch.Tensor): The cumulative sequence lengths tensor.
        scale (float, optional): The attention scaling factor. Defaults to 1.0.
        BLOCK_SIZE (int, optional): The block size for block diffusion attention. Defaults to 8.

    Returns:
        torch.Tensor: The output of the attention function.
    """
    return MojoDllmAttentionFunction.apply(query, key, value, cu_seqlen, scale, BLOCK_SIZE)
