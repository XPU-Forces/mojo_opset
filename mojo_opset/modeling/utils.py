from itertools import accumulate
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


def merge_group_and_share_ffn(
    config,
    group_ffn_output: torch.Tensor,
    share_ffn_output: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    if config.parallel_config.dp_size == 1:
        return group_ffn_output + share_ffn_output
    dp_rank = config.parallel_config.dp_rank
    if use_padding:
        raise NotImplementedError("merge_group_ffn not implemented.")
        global_max_batch_size = config.runtime_config.max_batch_size * config.parallel_config.dp_size
        assert group_ffn_output.shape[0] == global_max_batch_size
        merge_group_ffn(
            group_ffn_output,
            share_ffn_output,
            dp_rank_input_len,
            global_max_batch_size,
            dp_rank,
        )
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
    if config.parallel_config.dp_size == 1:
        return hidden_states
    dp_rank = config.parallel_config.dp_rank
    if use_padding:
        raise NotImplementedError("dp_pad not implemented.")
        global_max_batch_size = config.runtime_config.max_batch_size * config.parallel_config.dp_size
        hidden_states = dp_pad(hidden_states, dp_rank_input_len, global_max_batch_size, dp_rank)
    else:
        left_len = sum(host_dp_rank_input_len[:dp_rank])
        right_len = sum(host_dp_rank_input_len[dp_rank + 1 :])
        hidden_states = F.pad(hidden_states, (0, 0, left_len, right_len))
    if config.runtime_config.is_deterministic:
        raise NotImplementedError("all_reduce_with_all_to_all not implemented.")
    else:
        dist.all_reduce(hidden_states, group=config.parallel_config.dp_group)
    return hidden_states


def dp_scatter(
    config,
    ffn_output: torch.Tensor,
    dp_rank_input_len: torch.Tensor,
    local_token_num: int,
    use_padding: bool,
    host_dp_rank_input_len: List[int],
):
    dp_rank = config.parallel_config.dp_rank
    if config.parallel_config.dp_size == 1:
        return ffn_output
    if use_padding:
        raise NotImplementedError("dp_unpad not implemented.")
        return dp_unpad(ffn_output, dp_rank_input_len, local_token_num, dp_rank)
    else:
        cu_lens = list(accumulate([0] + host_dp_rank_input_len))
        return ffn_output[cu_lens[dp_rank] : cu_lens[dp_rank + 1]]
