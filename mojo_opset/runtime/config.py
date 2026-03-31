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


class MojoParallelConfig(BaseModel):
    dp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    tp_size: int = 1
    dp_rank: int = 0
    pp_rank: int = 0
    ep_rank: int = 0
    tp_rank: int = 0
    dp_group: list = []
    pp_group: list = []
    ep_group: list = []
    tp_group: list = []
    world_size: int = 1


class MojoConfig(BaseModel):
    # TODO(liuyuan): use MojoModelConfig when it is ready for all models.
    model_config: MojoDynamicConfig = None
    parallel_config: MojoParallelConfig = MojoParallelConfig()
    runtime_config: MojoRunTimeConfig = MojoRunTimeConfig()
