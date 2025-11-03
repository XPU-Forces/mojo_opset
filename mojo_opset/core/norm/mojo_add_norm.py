from abc import abstractmethod
from typing import Any
from typing import Tuple

import torch
import torch.nn.functional as F

from ..mojo_operator import MojoOperator


class MojoResidualAddNorm(MojoOperator):
    def __init__(
        self,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        eps: float = 1e-05,
        norm_pos: str = "post",
        norm_type: str = "rmsnorm",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.eps = eps

        self.norm_pos = norm_pos
        assert self.norm_pos in ["pre", "post"]

        self.norm_type = norm_type
        assert self.norm_type in ["rmsnorm", "layernorm"]

        self.weight = weight
        self.bias = bias

    @abstractmethod
    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> Tuple[Any]:
        def norm_func(hidden_state: torch.Tensor) -> Tuple[Any]:
            if self.norm_type == "layernorm":
                return F.layer_norm(
                    hidden_state,
                    [hidden_state.shape[-1]],
                    weight=self.weight,
                    bias=self.bias,
                    eps=self.eps,
                )
            elif self.norm_type == "rmsnorm":
                return F.rms_norm(hidden_state, (hidden_state.size(-1),), weight=self.weight, eps=self.eps)

        if self.norm_pos == "pre":
            if residual is not None:
                residual = hidden_state + residual
            else:
                residual = hidden_state
            hidden_state = norm_func(residual)
        else:
            if residual is not None:
                hidden_state = hidden_state + residual
            hidden_state = norm_func(hidden_state)
            residual = hidden_state

        return hidden_state, residual

    def forward_analysis(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> Tuple[Any]:
        """ignore weight and bias"""
        read_bytes = hidden_state.numel() * hidden_state.dtype.element_size()

        if self.norm_type == "layernorm":
            comp_intensity = 7
        elif self.norm_type == "rmsnorm":
            comp_intensity = 6

        if residual is not None:
            read_bytes = read_bytes * 2
            write_byte = read_bytes
            comp_intensity += 1
        else:
            write_byte = read_bytes * 2

        flops = comp_intensity * hidden_state.numel()

        # read_in_bytes, write_out_bytes, flops
        return read_bytes, write_byte, flops


class MojoResidualAddNormQuant(MojoOperator):
    pass


class MojoResidualAddNormCast(MojoOperator):
    pass
