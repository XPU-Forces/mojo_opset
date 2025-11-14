from abc import abstractmethod
from typing import Any, Tuple, Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

from ...mojo_utils import get_mojo_exec_mode
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
        only_k_norm: bool = False,
        q_head_num: int = 0,
        kv_head_num: int = 0,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        if norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError('norm_type only support "rmsnorm","layernorm", got {}'.format(norm_type))

        if norm_type == "rmsnorm" and beta is not None:
            raise ValueError("RMSNorm doesn't support beta.")

        self.epsilon = float(epsilon)
        self.gamma = gamma
        self.beta = beta
        self.norm_type = norm_type
        self.affine = gamma is not None and beta is not None
        self.is_varlen = is_varlen
        self.only_k_norm = only_k_norm
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num

        mode_str = get_mojo_exec_mode(MojoNorm.__name__, "FWD", self.layer_idx)
        self._set_forward_mode(mode_str)


    @abstractmethod
    def forward_std(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        input layout: 
        - is_varlen=True(TND): input shape = [T,D] or [T,*,D]
        - is_varlen=False(BNSD): input shape = [B,S,*,D]
        """
        def torch_rms_norm(input, gamma, eps):
            # input: (..., D)
            # gamma: (D,)
            variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = input * torch.rsqrt(variance + eps)
            return (gamma * hidden_states).to(input.dtype)
        
        def torch_layer_norm(input, gamma, beta, eps):
            mu = x.mean(dim=-1, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
            y = (x - mu) / torch.sqrt(var + eps)
            if gamma is not None:
                y = y * gamma
            if beta is not None:
                y = y + beta
            return y

        x = hidden_state
        eps = float(self.epsilon)
        if self.is_varlen:
            if x.ndim not in (2, 3):
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(x.shape)}")
        else:
            if x.ndim < 3:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(x.shape)}")
        if self.norm_type == "layernorm":
            if self.only_k_norm:
                k_slice = slice(self.q_head_num, self.q_head_num + self.kv_head_num)
                k_part = x[:, :, k_slice, :]

                k_part_normed = torch_layer_norm(k_part, self.gamma, self.beta, eps)
                x[:, :, k_slice, :] = k_part_normed
                y = x
            else:
                y = torch_layer_norm(x, self.gamma, self.beta, eps)
        elif self.norm_type == "rmsnorm":
            if self.only_k_norm:
                k_slice = slice(self.q_head_num, self.q_head_num + self.kv_head_num)
                k_part = x[:, :, k_slice, :]

                k_part_normed = torch_rms_norm(k_part, self.gamma, eps)
                x[:, :, k_slice, :] = k_part_normed
                y = x
            else:
                y = torch_rms_norm(x, self.gamma, eps)
        else:
            raise ValueError("norm_type only support 'layernorm' or 'rmsnorm'")
        return y

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
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
