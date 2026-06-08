import torch

from mojo_opset.core import MojoResidualAddRMSNorm

from ._utils import _uc_kernels


_DTYPE_TO_API = {
    torch.bfloat16: "mojo_residual_add_rmsnorm_bf16",
    torch.float16: "mojo_residual_add_rmsnorm_fp16",
}


class UCResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_state.shape != residual.shape:
            raise ValueError(
                f"UC mojo_residual_add_rmsnorm expects matching shapes, "
                f"got hidden_state={hidden_state.shape}, residual={residual.shape}."
            )

        api = _DTYPE_TO_API.get(hidden_state.dtype)
        if api is None:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # P-Wave-6 A1: unified wrapper-cleanup guard (UCNormRoPE template).
        # When the wheel manifest is missing this kernel (e.g. partial-link
        # `_kernels.so` from a `binary.cc:281` abort, see P3-01/P3-05 sev-1),
        # silently fall back to the parent torch path instead of crashing
        # with a raw `KeyError`. Sev-1 3-axis accuracy audit (A1 API existence
        # / A2 direct ABI / A3 fresh iso build) is the out-of-band gate that
        # catches fake-PASS signals produced by this fallback.
        kernels = _uc_kernels()
        if api not in kernels.keys():
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        original_shape = hidden_state.shape
        x = hidden_state.contiguous()
        r = residual.contiguous().to(x.dtype)
        weight = self.weight.contiguous().to(x.dtype)

        x2 = x.reshape(-1, x.shape[-1])
        r2 = r.reshape(-1, r.shape[-1])
        rows, cols = x2.shape

        eps = float(self.variance_epsilon)

        y = torch.empty_like(x2)
        residual_sum = torch.empty_like(x2)

        # New ABI (P3-02): (x, residual, weight, y, residual_out, M, N, eps_scalar)
        kernels[api](x2, r2, weight, y, residual_sum, rows, cols, eps)

        y_out = y.reshape(original_shape)
        residual_sum_out = residual_sum.reshape(original_shape)

        if self.norm_pos == "pre":
            return y_out, residual_sum_out
        return y_out, y_out
