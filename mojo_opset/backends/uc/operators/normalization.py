import torch

from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm

from ._utils import _matrix_shape
from ._utils import run_kernel


class UCRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.numel() == 0:
            return torch.empty_like(hidden_state)

        kernel_input = hidden_state.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)
        eps = float(self.variance_epsilon)

        run_kernel(
            "mojo_rmsnorm",
            kernel_input.dtype,
            kernel_input,
            self.weight.contiguous(),
            kernel_output,
            rows,
            cols,
            eps,
        )
        return kernel_output.reshape(hidden_state.shape)


class UCResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            raise ValueError("UC backend MojoResidualAddRMSNorm requires residual.")
        if hidden_state.shape != residual.shape:
            raise ValueError(
                f"UC backend MojoResidualAddRMSNorm expects matching shapes, "
                f"got {hidden_state.shape} and {residual.shape}."
            )
        if hidden_state.dtype != residual.dtype:
            raise ValueError(
                f"UC backend MojoResidualAddRMSNorm expects matching dtypes, "
                f"got {hidden_state.dtype} and {residual.dtype}."
            )
        if hidden_state.numel() == 0:
            empty = torch.empty_like(hidden_state)
            return empty, empty

        kernel_input = hidden_state.contiguous()
        kernel_residual = residual.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)
        kernel_residual_output = torch.empty_like(kernel_input)
        eps = float(self.variance_epsilon)

        run_kernel(
            "mojo_residual_add_rmsnorm",
            kernel_input.dtype,
            kernel_input,
            kernel_residual,
            self.weight.contiguous(),
            kernel_output,
            kernel_residual_output,
            rows,
            cols,
            eps,
        )

        output = kernel_output.reshape(hidden_state.shape)
        updated_residual = kernel_residual_output.reshape(hidden_state.shape)
        if self.norm_pos == "pre":
            return output, updated_residual
        return output, output


class UCResidualAddLayerNorm(MojoResidualAddLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            raise ValueError("UC backend MojoResidualAddLayerNorm requires residual.")
        if hidden_state.shape != residual.shape:
            raise ValueError(
                f"UC backend MojoResidualAddLayerNorm expects matching shapes, "
                f"got {hidden_state.shape} and {residual.shape}."
            )
        if hidden_state.dtype != residual.dtype:
            raise ValueError(
                f"UC backend MojoResidualAddLayerNorm expects matching dtypes, "
                f"got {hidden_state.dtype} and {residual.dtype}."
            )
        if self.weight is None or self.bias is None:
            raise NotImplementedError("UC backend mojo_residual_add_layernorm requires weight and bias.")
        if hidden_state.numel() == 0:
            empty = torch.empty_like(hidden_state)
            return empty, empty

        kernel_input = hidden_state.contiguous()
        kernel_residual = residual.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)
        kernel_residual_output = torch.empty_like(kernel_input)
        eps = float(self.variance_epsilon)

        run_kernel(
            "mojo_residual_add_layernorm",
            kernel_input.dtype,
            kernel_input,
            kernel_residual,
            self.weight.contiguous(),
            self.bias.contiguous(),
            kernel_output,
            kernel_residual_output,
            rows,
            cols,
            eps,
        )

        output = kernel_output.reshape(hidden_state.shape)
        updated_residual = kernel_residual_output.reshape(hidden_state.shape)
        if self.norm_pos == "pre":
            return output, updated_residual
        return output, output


class UCLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.numel() == 0:
            return torch.empty_like(hidden_state)
        if self.weight is None or self.bias is None:
            raise NotImplementedError("UC backend mojo_layernorm requires weight and bias.")

        kernel_input = hidden_state.contiguous()
        rows, cols = _matrix_shape(kernel_input)
        kernel_output = torch.empty_like(kernel_input)
        eps = float(self.variance_epsilon)

        run_kernel(
            "mojo_layernorm",
            kernel_input.dtype,
            kernel_input,
            self.weight.contiguous(),
            self.bias.contiguous(),
            kernel_output,
            rows,
            cols,
            eps,
        )
        return kernel_output.reshape(hidden_state.shape)
