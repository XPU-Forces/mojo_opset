from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Tuple

import torch

from ...utils.mode import get_mojo_exec_mode
from ..mojo_operator import MojoOperator


class MojoNorm(MojoOperator):
    """
    Common parameter definitions for normalization operator (LayerNorm/RMSNorm).

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
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        if norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError('norm_type should be {"rmsnorm","layernorm"}')

        if norm_type == "rmsnorm" and beta is not None:
            raise ValueError("RMSNorm don't support beta.")

        self.epsilon = float(epsilon)
        self.gamma = gamma
        self.beta = beta
        self.norm_type = norm_type
        self.affine = gamma is not None and beta is not None
        self.is_varlen = is_varlen

        mode_str = get_mojo_exec_mode(MojoNorm.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)

    @abstractmethod
    def forward_std(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        raise NotImplementedError

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
        """ignore weight and bias"""
        read_bytes = hidden_state.numel() * hidden_state.dtype.element_size()
        write_bytes = read_bytes

        if self.norm_type == "layernorm":
            comp_intensity = 7
        elif self.norm_type == "rmsnorm":
            comp_intensity = 6

        flops = comp_intensity * hidden_state.numel()

        # read_bytes, write_bytes, flops
        return read_bytes, write_bytes, flops


class MojoNormQuant(MojoOperator):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-05,
        norm_type: str = "rmsnorm",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.variance_epsilon = eps

        self.norm_type = norm_type
        assert self.norm_type in ["rmsnorm", "layernorm"]
