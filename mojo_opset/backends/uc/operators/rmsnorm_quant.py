import torch

from mojo_opset.core import MojoRMSNormQuant

from ._utils import _matrix_shape
from ._utils import run_kernel


class UCRMSNormQuant(MojoRMSNormQuant):
    """UC backend for fused RMSNorm + dynamic per-token int8 quantization.

    Kernel ABI (per dtype suffix ``bf16`` / ``fp16``)::

        mojo_rmsnorm_quant_<dtype>(
            x        : (M, N) <dtype>,
            weight   : (N,)   fp32,
            y        : (M, N) int8,
            scale    : (M,)   fp32,
            M        : i32,
            N        : i32,
            eps      : fp32,
        )

    Falls back to the parent (torch) reference when the kernel cannot be
    used: non bf16/fp16 input, non int8 quant target, presence of
    ``smooth_scale`` (kernel does not fuse smooth_scale), empty input, or
    when the wheel does not register the typed API.
    """

    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, smooth_scale: torch.Tensor = None):
        # Smooth-scale fusion is not implemented in the kernel — defer to ref.
        if smooth_scale is not None:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        # Only int8 dynamic quant is supported by this kernel.
        if self.quant_dtype != torch.int8:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        # Only bf16/fp16 input is supported by this kernel.
        if hidden_state.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )
        # Empty tensor edge-case: defer to ref to keep shape/dtype semantics.
        if hidden_state.numel() == 0:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        kernel_input = hidden_state.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        if cols != self.norm_size:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        weight_fp32 = self.weight.contiguous().to(dtype=torch.float32)
        kernel_y = torch.empty(
            (rows, cols),
            dtype=torch.int8,
            device=kernel_input.device,
        )
        kernel_scale = torch.empty(
            (rows,),
            dtype=torch.float32,
            device=kernel_input.device,
        )
        eps = float(self.variance_epsilon)

        try:
            run_kernel(
                "mojo_rmsnorm_quant",
                kernel_input.dtype,
                kernel_input,
                weight_fp32,
                kernel_y,
                kernel_scale,
                rows,
                cols,
                eps,
            )
        except NotImplementedError:
            raise NotImplementedError(
                "UC backend cannot service this call (shape/dtype/contract not "
                "honoured by the wheel kernel). Per project rule 'wheel 没实现的就直接给报错' "
                "(2026-06-08), this wrapper does not silently fall back to torch — "
                "use TTX / torch_npu / torch_native backend for unsupported inputs."
            )

        # quant_dtype is int8 here; clone to release storage shared with kernel buf.
        output = kernel_y.reshape(hidden_state.shape).to(dtype=self.quant_dtype)
        # Per-token scale shape is (*, 1) per parent forward contract.
        scale = kernel_scale.reshape(*hidden_state.shape[:-1], 1)
        return output, scale
