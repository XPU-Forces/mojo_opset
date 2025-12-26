import torch

from mojo_opset.core import MojoNorm
from mojo_opset.core import MojoResidualAddNorm


class AnalysisNorm(MojoNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_bytes: int = 0
        self.write_bytes: int = 0
        self.flops: int = 0

    def forward(self, hidden_state) -> torch.Tensor:
        self.read_bytes = hidden_state.numel() * hidden_state.dtype.element_size()
        self.write_bytes = self.read_bytes

        if self.norm_type == "layernorm":
            comp_intensity = 7
        elif self.norm_type == "rmsnorm":
            comp_intensity = 6

        self.flops = comp_intensity * hidden_state.numel()

        return torch.empty_like(hidden_state)


class AnalysisResidualAddNorm(MojoResidualAddNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_bytes: int = 0
        self.write_bytes: int = 0
        self.flops: int = 0

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        hidden_state_bytes = hidden_state.numel() * hidden_state.dtype.element_size()

        if self.norm_type == "layernorm":
            comp_intensity = 7
        elif self.norm_type == "rmsnorm":
            comp_intensity = 6

        if residual is not None:
            self.read_bytes = hidden_state_bytes * 2
            self.write_byte = hidden_state_bytes * 2
            comp_intensity += 1
        else:
            self.read_bytes = hidden_state_bytes
            self.write_byte = hidden_state_bytes * 2

        self.flops = comp_intensity * hidden_state.numel()

        if residual:
            return torch.empty_like(hidden_state), residual
        else:
            return torch.empty_like(hidden_state), torch.empty_like(hidden_state)
