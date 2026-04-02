from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import List

import torch

try:
    from pydantic.v1 import BaseModel
    from pydantic.v1 import validator
except ImportError:
    from pydantic import BaseModel
    from pydantic import validator


dtype_mapping = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# TODO(liuyuan):  get the common configuration fields for all LLM models, add them here as static fields and add conversion functions to convert from different configs.
class MojoDynamicConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class MojoModelConfig(MojoDynamicConfig):
    model_name: str = ""

    hidden_size: int
    embed_dim: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int

    vocab_size: int
    max_position_embeddings: int

    dtype: torch.dtype = torch.bfloat16

    kv_mirror_layers: List[int] = []
    kv_mirror_imitated_layers: List[int] = []

    rope_mode: str = ""
    rope_scale: int
    rope_percentage: float = 1.0

    has_context_layernorm: bool = True
    has_k_layernorm: bool = True
    use_rmsnorm: bool = True
    residual_post_ln_layers: List[int] = []

    has_attn_bias: bool = False
    gqa_weights_layout: str = "AABB"
    q_head_times: int = 1

    moe_expert_num: int = 0
    moe_topk: int = 0
    share_expert_num: int = 0
    moe_ffn_internal_dim: int = 0
    moe_ffn_has_bias: bool = False
    is_exp_moe: bool = False

    has_mlp_gate: bool = True

    is_meta: bool = False

    @validator("dtype", pre=True)
    def validate_dtype(cls, value):
        if isinstance(value, str):
            if value in dtype_mapping:
                return dtype_mapping[value]
            else:
                raise ValueError(f"unsupported dtype: {value}")
        return value


class MojoRunTimeConfig(BaseModel):
    preshard_only: bool = False

    is_deterministic: bool = False

    use_device_graph: bool = False
    use_paged_attention: bool = False
    use_mtp: bool = False
    mtp_draft_recurrent: bool = False

    max_batch_size: int = 16
    max_length: int = 2048
    max_total_tokens: int = 0
    max_num_pred_tokens: int = -1

    num_pages: int = 32
    page_block_size: int = 256

    vanilla_checkpoint_path: str = None
    preshard_checkpoint_path: str = None


class AFDRole(Enum):
    """Enum for AFD roles."""

    ATTN = auto()
    FFN = auto()

    def __str__(self):
        return self.name


@dataclass
class MojoParallelConfig:
    """
    Configuration for distributed parallelism.
    """

    AFD_ENABLED: bool = False
    AFD_ROLE: AFDRole = AFDRole.FFN

    PP_SIZE: int = 1

    ATTN_DP_SIZE: int = 1
    ATTN_SP_SIZE: int = 1
    ATTN_TP_SIZE: int = 1
    ATTN_PP_SIZE: int = 1  # for AFD_ATTN only

    FFN_EP_SIZE: int = 1
    FFN_TP_SIZE: int = 1
    FFN_PP_SIZE: int = 1  # for AFD_FFN only

    USE_ULISSES: bool = True  # Ulysses

    def __post_init__(self):
        """Validate configuration values."""

        if (
            self.PP_SIZE <= 0
            or self.ATTN_DP_SIZE <= 0
            or self.ATTN_SP_SIZE <= 0
            or self.ATTN_TP_SIZE <= 0
            or self.ATTN_PP_SIZE <= 0
            or self.FFN_EP_SIZE <= 0
            or self.FFN_TP_SIZE <= 0
            or self.FFN_PP_SIZE <= 0
        ):
            raise ValueError("All parallel sizes must be positive integers")

        # TODO(minghui): necessary to support FFN TP & EP at the same time?
        # (wens) yes, e.g. m8 shard expert vs MoE.

        # if self.FFN_EP_SIZE > 1 and self.FFN_TP_SIZE > 1:
        #     raise ValueError("FFN TP and FFN EP can not be enabled at the same time")

        # if not self.AFD_ENABLED:
        #     # In non-AFD mode, FFN and Attention share the same resources.
        #     # We assume FFN EP/TP dimensions cover the same total parallelism as Attn DP/SP/TP.
        #     # Note: PP_SIZE is shared in non-AFD mode.
        #     if (
        #         self.FFN_EP_SIZE * self.FFN_TP_SIZE
        #         != self.ATTN_DP_SIZE * self.ATTN_SP_SIZE * self.ATTN_TP_SIZE
        #     ):
        #         raise ValueError(
        #             "FFN EP size * FFN TP size must be equal to ATTN DP size * ATTN SP size * ATTN TP size"
        #         )

    @property
    def world_size(self) -> int:
        """Total number of processes in the distributed system."""
        if not self.AFD_ENABLED:
            return self.ATTN_DP_SIZE * self.ATTN_SP_SIZE * self.ATTN_TP_SIZE * self.PP_SIZE
        else:
            return (
                self.ATTN_DP_SIZE * self.ATTN_SP_SIZE * self.ATTN_TP_SIZE * self.ATTN_PP_SIZE
                + self.FFN_EP_SIZE * self.FFN_TP_SIZE * self.FFN_PP_SIZE
            )

    @property
    def attn_world_size(self) -> int:
        """Total number of processes in non-PP dimensions."""
        if not self.AFD_ENABLED:
            raise ValueError("ATTN world size is not defined when AFD is disabled")
        else:
            return self.ATTN_DP_SIZE * self.ATTN_SP_SIZE * self.ATTN_TP_SIZE * self.ATTN_PP_SIZE

    @property
    def ffn_world_size(self) -> int:
        """Total number of processes in non-PP dimensions."""
        if not self.AFD_ENABLED:
            raise ValueError("FFN world size is not defined when AFD is disabled")
        else:
            return self.FFN_EP_SIZE * self.FFN_TP_SIZE * self.FFN_PP_SIZE


class MojoConfig(BaseModel):
    # TODO(liuyuan): use MojoModelConfig when it is ready for all models.
    model_config: MojoDynamicConfig = None
    parallel_config: MojoParallelConfig = MojoParallelConfig()
    runtime_config: MojoRunTimeConfig = MojoRunTimeConfig()
