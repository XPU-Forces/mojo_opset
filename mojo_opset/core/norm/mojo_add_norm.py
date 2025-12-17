from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoResidualAddNorm(MojoOperator):
    """
    Common parameter definitions for fusion operator (Residual+LayerNorm/RMSNorm).

    Init parameters:
    - epsilon (float): Numerical stability term, default 1e-5, must be > 0.
    - norm_type (str): Normalization type, enumeration {"rmsnorm", "layernorm"}, default "rmsnorm".
    - gamma (torch.Tensor|None): Affine parameter gamma, optional, 1-D, dtype floating point.
    - beta (torch.Tensor|None): Affine parameter beta (only supported for LayerNorm), optional, 1-D, dtype floating point.
    - is_varlen (bool): When True, prioritize TND (continuous token perspective) normalization; when False, use BSND; default True.
    - op_name (str): Operator name placeholder.
    - layer_idx (int): Layer index placeholder.

    Description: Only covers common parameters and lightweight validation; forward computation body is placeholder, does not include backend or quantization implementation.
    """

    def __init__(
        self,
        epsilon: float = 1e-05,
        norm_type: str = "rmsnorm",
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        norm_pos: str = "pre",
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        if norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError('norm_type should be {"rmsnorm","layernorm"}')

        if norm_type == "rmsnorm" and beta is not None:
            raise ValueError("RMSNorm don't support beta.")
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos should be 'pre' or 'post'")

        self.epsilon = float(epsilon)
        self.gamma = gamma
        self.beta = beta
        self.norm_type = norm_type
        self.norm_pos = norm_pos
        self.affine = gamma is not None and beta is not None
        self.is_varlen = is_varlen

    @abstractmethod
    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> Tuple[Any]:
        raise NotImplementedError

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
