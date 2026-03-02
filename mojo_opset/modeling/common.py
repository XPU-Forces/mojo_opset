from itertools import accumulate
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)

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


def merge_group_and_share_ffn(
    config,
    group_ffn_output: torch.Tensor,
    share_ffn_output: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    if config.dp_size == 1:
        return group_ffn_output + share_ffn_output
    dp_rank = config.dp_rank
    if use_padding:
        raise NotImplementedError("merge_group_ffn not implemented.")
        global_max_batch_size = config.max_batch_size * config.dp_size
        assert group_ffn_output.shape[0] == global_max_batch_size
        merge_group_ffn(group_ffn_output, share_ffn_output, dp_rank_input_len, global_max_batch_size, dp_rank)
    else:
        rank_start = sum(host_dp_rank_input_len[:dp_rank])
        group_ffn_output[rank_start : rank_start + share_ffn_output.shape[0], :] += share_ffn_output
    return group_ffn_output


def dp_allreduce(
    config,
    hidden_states: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    if config.dp_size == 1:
        return hidden_states
    dp_rank = config.dp_rank
    if use_padding:
        raise NotImplementedError("dp_pad not implemented.")
        global_max_batch_size = config.max_batch_size * config.dp_size
        hidden_states = dp_pad(hidden_states, dp_rank_input_len, global_max_batch_size, dp_rank)
    else:
        left_len = sum(host_dp_rank_input_len[:dp_rank])
        right_len = sum(host_dp_rank_input_len[dp_rank + 1 :])
        hidden_states = F.pad(hidden_states, (0, 0, left_len, right_len))
    if config.is_deterministic:
        raise NotImplementedError("all_reduce_with_all_to_all not implemented.")
    else:
        dist.all_reduce(hidden_states, group=config.dp_group)
    return hidden_states


def dp_scatter(
    config,
    ffn_output: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    local_token_num: int,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    dp_rank = config.dp_rank
    if config.dp_size == 1:
        return ffn_output
    if use_padding:
        raise NotImplementedError("dp_unpad not implemented.")
        return dp_unpad(ffn_output, dp_rank_input_len, local_token_num, dp_rank)
    else:
        cu_lens = list(accumulate([0] + host_dp_rank_input_len))
        return ffn_output[cu_lens[dp_rank] : cu_lens[dp_rank + 1]]
