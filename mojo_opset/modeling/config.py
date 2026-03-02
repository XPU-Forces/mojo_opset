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


class MojoModelConfig(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)

    # Preshard config
    preshard_only: bool = False

    # Deterministic config
    is_deterministic: bool = False

    # Model definition
    model_name: str = ""
    model_config: "MojoModelConfig" = None
    hidden_size: int = None
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16

    # Runtime settings
    use_npu_graph: bool = False
    npu_graph_capture_range: List[int] = []
    use_paged_attention: bool = False
    use_mtp: bool = False
    mtp_draft_recurrent: bool = False

    # Batching and sequence length
    max_batch_size: int = 16
    max_length: int = 2048
    max_total_tokens: int = 0
    max_num_pred_tokens: int = -1

    # Distributed settings
    world_size: int = 1
    local_world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0
    tp_size: int = 1
    dp_size: int = 1
    tp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0
    tp_group: torch.distributed.ProcessGroup = None
    dp_group: torch.distributed.ProcessGroup = None
    ep_group: torch.distributed.ProcessGroup = None

    # Paged attention specific
    num_pages: int = 32
    page_block_size: int = 256

    # Checkpoint and weights
    vanilla_checkpoint_path: str = None
    preshard_checkpoint_path: str = None

    class Config:
        # Enable support for arbitrary types
        arbitrary_types_allowed = True
        extra = "allow"

    @validator("dtype", pre=True)
    def validate_dtype(cls, value):
        if isinstance(value, str):
            if value in dtype_mapping:
                return dtype_mapping[value]
            else:
                raise ValueError(f"unsupported dtype: {value}")
        return value
