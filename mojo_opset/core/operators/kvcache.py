from .. import VALID_KV_LAYOUTS
from ..operator import MojoOperator


class MojoStoreKVCache(MojoOperator):
    pass


class MojoStorePagedKVCache(MojoOperator):
    def __init__(
        self,
        kv_layout: str = VALID_KV_LAYOUTS[0],
        kv_dim: int = None,
        is_varlen: bool = True,
        op_name: str = "",
    ):
        """
        Common parameter definitions for StoreKVPaged.

        Init parameters:
        - kv_layout (str): KV computation layout, values defined by VALID_KV_LAYOUTS, default VALID_KV_LAYOUTS[0].
        - kv_dim (int): KV hidden dimension D_kv, used in scenarios like MLA to distinguish KV compression dimension from K dimension; positive integer.
        - is_varlen (bool): When True, prioritize TND continuous token perspective; when False, use BNSD; default True.
        - op_name (str): Operator name placeholder.

        Scope and description:
        - Only covers common parameters; does not involve paging strategy details and quantization parameters (QuantMode/QuantParam).
        """
        super().__init__(op_name)
        if kv_layout not in VALID_KV_LAYOUTS:
            raise ValueError(f"kv_layout must be one of {VALID_KV_LAYOUTS}, got {kv_layout}")
        if kv_dim is None or not isinstance(kv_dim, int) or kv_dim <= 0:
            raise ValueError("kv_dim must be a positive integer")
        if not isinstance(is_varlen, bool):
            raise TypeError("is_varlen must be a boolean type")
        self.kv_layout = kv_layout
        self.kv_dim = kv_dim
        self.is_varlen = is_varlen


class MojoStoreMLAKVCache(MojoOperator):
    pass


class MojoStorePagedMLAKVCache(MojoOperator):
    pass


class MojoKVCacheCast(MojoOperator):
    pass
