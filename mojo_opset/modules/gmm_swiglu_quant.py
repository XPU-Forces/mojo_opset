"""Weight-owning MXFP8 quantization modules."""

from typing import Optional

import torch

from mojo_opset import functions
from mojo_opset.kernels.torch.gmm_swiglu_quant import mxfp8_quantize_weight


class GmmSwigluQuant(torch.nn.Module):
    """MXFP8 gate/up projection + SwiGLU + output quantization.

    Mirrors the ``gmm_swiglu_quant`` function semantics (glm-5.3 lineage;
    ``swiglu_limit=0`` disables the clamp, glm-5.3-flash uses 10.0). Owns
    the fused dense-expert weight as non-trainable buffers — ``weight``
    ``[1, hidden_size, 2*intermediate_size]`` float8_e4m3fn (first N half
    feeds the sigmoid branch) and ``weight_scale`` e8m0 ``[1, 2*intermediate_size,
    ceil(hidden_size/32)]`` (N-major) — quantized at init
    from the ``GlmMoeDsaConfig.initializer_range`` scale. ``forward``
    takes pre-quantized activations ``(x, x_scale)`` and returns
    ``(y, y_scale)`` for the next quantized GEMM.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 0.0,
        *,
        implementation: Optional[str] = None,
        device=None,
    ):
        super().__init__()
        n = 2 * intermediate_size
        # GlmMoeDsaConfig.initializer_range keeps activations in range; MX
        # groups run along K, so the fused weight is quantized per column.
        data, scales = mxfp8_quantize_weight(torch.randn(hidden_size, n, device=device) * 0.02)
        self.register_buffer("weight", data.view(1, hidden_size, n))
        self.register_buffer("weight_scale", scales.view(1, n, -1))
        self.swiglu_limit = swiglu_limit
        self.implementation = implementation

    def forward(self, x: torch.Tensor, x_scale: torch.Tensor, group_list: Optional[torch.Tensor] = None):
        if group_list is None:
            rows = x.numel() // x.shape[-1]
            group_list = torch.tensor([rows], dtype=torch.int64, device=x.device)
        return functions.gmm_swiglu_quant(
            x,
            self.weight,
            self.weight_scale,
            x_scale,
            group_list,
            swiglu_limit=self.swiglu_limit,
            implementation=self.implementation,
        )
