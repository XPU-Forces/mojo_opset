from abc import abstractmethod
from typing import Any, Tuple, Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

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

        if not isinstance(epsilon, (float, int)) or float(epsilon) <= 0:
            raise ValueError("epsilon 需为正数，实际为 {}".format(epsilon))
        if norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError('norm_type 取值需为 "rmsnorm","layernorm"，实际为 {}'.format(norm_type))

        # 类型与形状的轻量校验（无法在此处校验与输入末维完全匹配，留给 forward 校验）
        for name, t in ("gamma", gamma), ("beta", beta):
            if t is None:
                continue
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{name} 需为 torch.Tensor 或 None")
            if t.ndim != 1:
                raise ValueError(f"{name} 需为 1-D 张量，实际为 {tuple(t.shape)}")
            if t.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                raise TypeError(f"{name} 的 dtype 需为浮点类型，实际为 {t.dtype}")

        # RMSNorm 不支持 bias
        if norm_type == "rmsnorm" and beta is not None:
            raise ValueError("RMSNorm 不支持 beta 参数")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")

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

    def forward_ref(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        参考实现（golden）：LayerNorm / RMSNorm，严格区分 TND/BNSD 输入。
        输入布局契约：
        - 当 is_varlen=True（TND）：仅接受 [T,D] 或 [T,*,D]
        - 当 is_varlen=False（BNSD）：仅接受 [B,S,*,D]
        公式：
        - LayerNorm：y = (x-μ)/sqrt(σ^2+ε) · γ + β（μ/σ^2 按最后一维）
        - RMSNorm：y = x / (r + ε) · γ（r = sqrt(mean(x^2))）
        返回：与输入形状一致，dtype 与输入一致。
        """
        x = hidden_state
        eps = float(self.epsilon)
        if self.is_varlen:
            if x.ndim not in (2, 3):
                raise ValueError(f"Expected TND when is_varlen=True; got shape {tuple(x.shape)}")
        else:
            if x.ndim < 3:
                raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(x.shape)}")
        if self.norm_type == "layernorm":
            mu = x.mean(dim=-1, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
            y = (x - mu) / torch.sqrt(var + eps)
            if self.gamma is not None:
                y = y * self.gamma
            if self.beta is not None:
                y = y + self.beta
        elif self.norm_type == "rmsnorm":
            rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
            y = (x / rms.to(x.dtype))
            if self.gamma is not None:
                y = y * self.gamma
        else:
            raise ValueError("norm_type 需为 'layernorm' 或 'rmsnorm'")
        return y

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
