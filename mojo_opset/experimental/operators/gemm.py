import torch

from mojo_opset.core.operator import MojoOperator


class MojoQuantBatchGemmReduceSum(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor,
        trans_weight: bool = False,
    ):
        super().__init__()

        if not isinstance(trans_weight, bool):
            raise TypeError("trans_weight must be bool.")
        self.trans_weight = trans_weight
        self.weight = weight

    def forward(
        self,
        input: torch.Tensor,
        x1_scale: torch.Tensor,
        x2_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantized batch GEMM with per-token scaling and batch reduction.

        Applies batched GEMM on int8 inputs/weights in float32, scales by
        per-token `x1_scale` and per-output `x2_scale`, then reduces over batch.

        Args:
            input (torch.Tensor): 3D tensor of shape (B, M, K).
            x1_scale (torch.Tensor): Per-token scale of shape (B, M).
            x2_scale (torch.Tensor): Per-output scale of shape (N,), bfloat16 preferred.

        Returns:
            torch.Tensor: Reduced output of shape (M, N) in bfloat16.

        Notes:
            - Expects `self.weight` of shape (B, K, N) if `trans_weight` is False,
              otherwise (B, N, K) and transposed to (B, K, N).
            - The reduction sums outputs across batch dimension B.
            - This operator is experimental because its algorithmic contract is still under review.
        """
        assert input.dim() == 3, "input must be 3D"
        assert self.weight.dim() == 3, "weight must be 3D"

        if self.trans_weight:
            weight = self.weight.transpose(1, 2).contiguous()
        else:
            weight = self.weight

        b, m, k = input.shape
        b_w, k_w, n = weight.shape
        assert b == b_w, "input and weight must have same batch size"
        assert k == k_w, "K of input should be equal to K of weight"

        if x2_scale.dtype != torch.bfloat16:
            x2_scale = x2_scale.to(torch.bfloat16)

        out = torch.bmm(input.float(), weight.float()).to(torch.float32)
        out = x2_scale[None, None, :] * out
        out = x1_scale[:, :, None] * out

        reduced_out = torch.zeros(m, n, dtype=torch.bfloat16, device=out.device)
        for i in range(b):
            reduced_out += out[i, ...].to(torch.bfloat16)

        return reduced_out
