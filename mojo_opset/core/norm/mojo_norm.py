from abc import abstractmethod
from typing import Any
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from ...mojo_utils import get_mojo_exec_mode
from ..mojo_operator import MojoOperator


class MojoNorm(MojoOperator):
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
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self.norm_type = norm_type
        assert self.norm_type in ["rmsnorm", "layernorm"]

        if self.norm_type == "rmsnorm":
            self.bias = None

        mode_str = get_mojo_exec_mode(MojoNorm.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)

    @abstractmethod
    def forward_std(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        if self.norm_type == "layernorm":
            return F.layer_norm(
                hidden_state, hidden_state.shape[-1:], weight=self.weight, bias=self.bias, eps=self.variance_epsilon
            ).to(hidden_state.dtype)
        elif self.norm_type == "rmsnorm":
            return F.rms_norm(hidden_state, hidden_state.shape[-1:], weight=self.weight, eps=self.variance_epsilon).to(
                hidden_state.dtype
            )

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
