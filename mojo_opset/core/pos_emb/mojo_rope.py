from typing import Any
from typing import Optional
from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoRoPE(MojoOperator):
    """
    Common parameter definitions for Rotary Position Embedding (RoPE) operator.

    Init parameters:
    - rotary_offset (int): Rotary position offset, default 0.
    - interleaved (bool): Whether to use interleaved mode, default False.
    - dynamic_ntk (bool): Whether to use dynamic NTK scaling, default False.
    - max_seq_len (int|None): Maximum sequence length, optional.
    - is_varlen (bool): When True, prioritize TND (continuous token perspective) processing; when False, use BSND; default True.
    - op_name (str): Operator name placeholder.
    - layer_idx (int): Layer index placeholder.

    Description: Only covers common parameters and lightweight validation; forward computation body is placeholder, does not include backend or quantization implementation.
    """

    def __init__(
        self,
        rotary_offset: int = 0,
        interleaved: bool = False,
        dynamic_ntk: bool = False,
        max_seq_len: Optional[int] = None,
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        # 类型与数值的轻量校验
        if not isinstance(rotary_offset, int) or rotary_offset < 0:
            raise ValueError("rotary_offset 需为非负整数")
        if not isinstance(interleaved, bool):
            raise TypeError("interleaved 必须为 bool 类型")
        if not isinstance(dynamic_ntk, bool):
            raise TypeError("dynamic_ntk 必须为 bool 类型")
        if max_seq_len is not None and (not isinstance(max_seq_len, int) or max_seq_len <= 0):
            raise ValueError("max_seq_len 需为正整数或 None")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen 必须为 bool 类型")

        self.rotary_offset = rotary_offset
        self.interleaved = interleaved
        self.dynamic_ntk = dynamic_ntk
        self.max_seq_len = max_seq_len
        self.is_varlen = is_varlen

    def forward_std(
        self,
        q: torch.Tensor,  # [BNSD]
        k: torch.Tensor,  # [BNSD]
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cum_sum_query_len: Optional[torch.Tensor] = None,
    ) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cum_sum_query_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        print(q_rot.stride(), k_rot.stride())
        return q_rot, k_rot

    def forward_analysis(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[int, int, int]:
        pass


class MojoRoPEStoreKV(MojoOperator):
    pass


class MojoNormRoPE(MojoOperator):
    pass


class MojoNormRoPEStoreKV(MojoOperator):
    pass
