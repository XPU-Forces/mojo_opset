import torch

from mojo_opset.core import MojoGelu
from mojo_opset.core import MojoSilu
from mojo_opset.core import MojoSwiGLU

from ._utils import run_binary_kernel
from ._utils import run_unary_kernel


class UCGelu(MojoGelu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_unary_kernel("mojo_gelu", x)


class UCSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_unary_kernel("mojo_silu", x)


class UCSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu"]

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit > 0:
            raise NotImplementedError(
                f"UCSwiGLU does not implement swiglu_limit > 0 (got {self.swiglu_limit}); "
                "the UC kernel only supports the unclipped variant. "
                "Per project rule 'wheel 没实现的就直接给报错', this wrapper does not "
                "silently fall back to torch — use TTX / torch_npu / torch_native instead."
            )
        return run_binary_kernel("mojo_swiglu", gate_out, up_out)
