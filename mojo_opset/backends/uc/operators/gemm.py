import torch

from mojo_opset.core import MojoQuantGemm

from ._utils import _uc_kernels


_OUTPUT_DTYPE_SUFFIX = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


def _require_kernel(api: str):
    kernels = _uc_kernels()
    if api not in kernels.keys():
        raise NotImplementedError(
            f"UC backend kernel {api!r} is not available. Rebuild uc-kernel after adding the TileLang source."
        )
    return kernels[api]


class UCQuantGemm(MojoQuantGemm):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"input must be 2D, got shape {tuple(input.shape)}.")
        if input.dtype != torch.int8:
            raise NotImplementedError(f"UC QuantGemm supports int8 input, got {input.dtype}.")
        if self.trans_weight:
            weight = self.weight.t().contiguous()
        else:
            weight = self.weight
        input_scale = input_scale.flatten().float().contiguous()
        weight_scale = self.weight_scale.flatten().float().contiguous()
        if not input.is_contiguous():
            input = input.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        M, K = input.shape
        K_w, N = weight.shape
        if K_w != K:
            raise ValueError(f"input K {K} must match weight K {K_w}.")
        if input_scale.numel() != M:
            raise ValueError(f"input_scale length {input_scale.numel()} must equal M {M}.")
        if weight_scale.numel() != N:
            raise ValueError(f"weight_scale length {weight_scale.numel()} must equal N {N}.")

        output = torch.empty((M, N), device=input.device, dtype=self.output_dtype)
        if output.numel() == 0:
            return output
        suffix = _OUTPUT_DTYPE_SUFFIX.get(self.output_dtype)
        if suffix is None:
            raise NotImplementedError(f"UC QuantGemm does not support output dtype {self.output_dtype}.")

        kernel = _require_kernel(f"mojo_quant_gemm_{suffix}")
        kernel(input, weight, input_scale, weight_scale, output, M, K, N)
        return output
