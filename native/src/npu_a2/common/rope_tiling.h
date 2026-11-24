#pragma once
#include <cstdint>

// Standalone kernel parameters: deliberately independent of OPP registration.
struct RopeForwardTiling {
  int32_t token_num, batch_size, seqlen_per_block, head_num, head_dim;
  int32_t cut_head_dim, head_num_stride, token_num_stride, chunk_size;
  int32_t blocks_per_core, blocks_per_core_tail, normal_cores, tail_cores;
};
using RopeBackwardTiling = RopeForwardTiling;
