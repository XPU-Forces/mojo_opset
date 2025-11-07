from abc import abstractmethod
from typing import Any, Tuple, Optional, Union

import torch
import torch.nn.functional as F

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

        if not isinstance(epsilon, (float, int)) or float(epsilon) <= 0:
            raise ValueError("epsilon 需为正数")
        if norm_type not in ["rmsnorm", "layernorm"]:
            raise ValueError('norm_type 取值需为 {"rmsnorm","layernorm"}')

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
        if norm_pos not in ["pre", "post"]:
            raise ValueError("norm_pos 需为 'pre' 或 'post'")

        self.epsilon = float(epsilon)
        self.gamma = gamma
        self.beta = beta
        self.norm_type = norm_type
        self.norm_pos = norm_pos
        self.affine = gamma is not None and beta is not None
        self.is_varlen = is_varlen

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        输入：
        - hidden_state：张量，4-D 或更高维，最后一维为特征维；dtype∈{float16,float32,bfloat16}
        - residual：张量，与 hidden_state 形状一致，可选，默认 None。

        输出：与输入形状一致，dtype 与输入相同。
        """

        raise NotImplementedError("MojoNorm forward 当前仅进行通用参数校验，不包含具体实现")

    @abstractmethod
    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(self, hidden_state: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        def norm_func(hidden_state: torch.Tensor) -> Tuple[Any]:
            if self.norm_type == "layernorm":
                return F.layer_norm(
                    hidden_state,
                    [hidden_state.shape[-1]],
                    weight=self.gamma,
                    bias=self.beta,
                    eps=self.epsilon,
                )
            elif self.norm_type == "rmsnorm":
                return F.rms_norm(hidden_state, (hidden_state.size(-1),), weight=self.gamma, eps=self.epsilon)

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
