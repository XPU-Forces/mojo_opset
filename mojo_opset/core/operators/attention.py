from typing import Any
from typing import Optional
from typing import Tuple

import torch

from .. import VALID_KV_LAYOUTS
from ..operator import MojoOperator


class MojoDecodeGQA(MojoOperator):
    """
    Paged GQA attention operator.
    Args:
        is_causal (bool): Whether to apply causal masking.
        is_prefill (bool): Whether running in prefill mode.
        page_size (int): Page size for attention computation.
        softmax_scale (float): Scaling factor for the softmax operation.
        gqa_layout (str): Layout for GQA attention.
        window_size (int): Window size for attention computation, -1 means full attention.
        op_name (str): Name of the operator.
    """

    def __init__(self, is_causal, is_prefill, page_size, softmax_scale, gqa_layout, window_size, op_name):
        super().__init__(op_name)
        self.is_causal = is_causal
        self.is_prefill = is_prefill
        self.page_size = page_size


class MojoPagedDecodeGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        q_scale_factor: int = 1,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        kv_layout: str = VALID_KV_LAYOUTS[0],
        tp_size: int = 1,
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize the Paged Decode GQA attention operator with common parameters.
        Parameter descriptions:
        - q_scale_factor (int): Multiplier for q heads (integer, default 1), no scaling applied to query.
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - window_size (int): Attention window length; -1 means full window, or >=1 means sliding window length, default -1.
        - softmax_scale (Optional[float]): Scaling factor for attention scores, must be >0; default None.
        - kv_layout (str): KV storage layout indicator, values defined by VALID_KV_LAYOUTS, default VALID_KV_LAYOUTS[0].
        - tp_size (int): Tensor parallel size, default 1.
        - is_varlen (bool): When True, use TND (variable length) priority path; when False, use BSND; default True.
        - op_name (str): Operator name placeholder for registration and diagnostics.
        """
        super().__init__(op_name, layer_idx)

        # 输入参数校验
        if not isinstance(q_scale_factor, int) or q_scale_factor <= 0:
            raise ValueError(f"q_scale_factor must be a positive integer, got {q_scale_factor}")

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")

        if kv_layout not in VALID_KV_LAYOUTS:
            raise ValueError(f"kv_layout must be one of {VALID_KV_LAYOUTS}, got {kv_layout}")

        if not isinstance(tp_size, int) or tp_size <= 0:
            raise ValueError(f"tp_size must be a positive integer, got {tp_size}")

        if not isinstance(is_varlen, bool):
            raise ValueError(f"is_varlen must be a boolean, got {is_varlen}")

        # 成员变量赋值
        self.is_causal = is_causal
        self.q_scale_factor = q_scale_factor
        self.gqa_layout = gqa_layout
        self.window_size = window_size
        self.kv_layout = kv_layout
        self.tp_size = tp_size
        self.is_varlen = is_varlen

    def forward(self, q, k_cache, v_cache, seqlens, block_tables, softmax_scale: Optional[float] = None) -> Tuple[Any]:
        raise NotImplementedError


class MojoPrefillGQA(MojoOperator):
    pass


class MojoPagedPrefillGQA(MojoOperator):
    def __init__(
        self,
        is_causal: bool = True,
        q_scale_factor: int = 1,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        kv_layout: str = VALID_KV_LAYOUTS[0],
        tp_size: int = 1,
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        Initialize the Paged Prefill GQA attention operator with common parameters.
        Parameter descriptions:
        - q_scale_factor (int): Multiplier for q heads (integer, default 1), no scaling applied to query.
        - gqa_layout (str): GQA head grouping layout, values {"ABAB","AABB"}, default "ABAB".
        - is_causal (bool): Whether to enable causal masking, default True.
        - window_size (int): Attention window length; -1 means full window, or >=1 means sliding window length, default -1.
        - kv_layout (str): KV storage layout indicator, values defined by VALID_KV_LAYOUTS, default VALID_KV_LAYOUTS[0].
        - tp_size (int): Tensor parallel size, default 1.
        - is_varlen (bool): When True, use TND (variable length) priority path; when False, use BSND; default True.
        - op_name (str): Operator name placeholder for registration and diagnostics.
        """
        super().__init__(op_name, layer_idx)

        # 输入参数校验
        if not isinstance(q_scale_factor, int) or q_scale_factor <= 0:
            raise ValueError(f"q_scale_factor must be a positive integer, got {q_scale_factor}")

        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")

        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")

        if kv_layout not in VALID_KV_LAYOUTS:
            raise ValueError(f"kv_layout must be one of {VALID_KV_LAYOUTS}, got {kv_layout}")

        if not isinstance(tp_size, int) or tp_size <= 0:
            raise ValueError(f"tp_size must be a positive integer, got {tp_size}")

        if not isinstance(is_varlen, bool):
            raise ValueError(f"is_varlen must be a boolean, got {is_varlen}")

        # 成员变量赋值
        self.is_causal = is_causal
        self.q_scale_factor = q_scale_factor
        self.gqa_layout = gqa_layout
        self.window_size = window_size
        self.kv_layout = kv_layout
        self.tp_size = tp_size
        self.is_varlen = is_varlen

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[Any]:
        raise NotImplementedError


class MojoQuest(MojoOperator):
    """
    Paged Quest indexing operator for LLM Prefill.
    """

    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

    def forward(self, curr_query_seg, mins, maxs, top_k_page):
        raise NotImplementedError("MojoPagedPrefillQuest forward not implemented")


class MojoPagedPrefillBlockSparseAttention(MojoOperator):
    """
    Paged Block Sparse Attention operator for LLM Prefill.
    """

    def __init__(
        self,
        causal_mask: Optional[torch.Tensor],
        page_size: int,
        q_seg_size: int,
        topk_ratio: float,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.causal_mask = causal_mask
        self.page_size = page_size
        self.q_seg_size = q_seg_size
        self.topk_ratio = topk_ratio
        self.scale = 1.0

    def forward(
        self,
        curr_query_seg,
        key,
        value,
        whole_causal_mask,
        topk_page_indices,
        q_seg_id,
        q_chunk_size,
    ):
        raise NotImplementedError("MojoPagedPrefillBlockSparseAttention forward not implemented")


class MojoDecodeMLA(MojoOperator):
    pass


class MojoPagedDecodeMLA(MojoOperator):
    pass


class MojoDecodeNSA(MojoOperator):
    """
    MojoDecodeNSA operator.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPagedDecodeNSA(MojoOperator):
    """
    Paged MLA attention operator for LLM Decode.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPrefillMLA(MojoOperator):
    pass


class MojoPagedPrefillMLA(MojoOperator):
    pass


class MojoPrefillNSA(MojoOperator):
    """
    MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoPagedPrefillNSA(MojoOperator):
    """
    Paged MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale


class MojoSdpa(MojoOperator):
    def __init__(
        self,
        mask: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        enable_gqa: bool = False,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.mask = mask
        self.scale = scale
        self.enable_gqa = enable_gqa

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        raise NotImplementedError("MojoSdpa forward not implemented")
