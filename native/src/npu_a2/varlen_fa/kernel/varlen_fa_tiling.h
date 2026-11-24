#pragma once

#include <cstdint>

struct VarlenFaTiling {
  uint32_t batch_size;
  uint32_t total_q_tokens;
  uint32_t total_k_tokens;
  uint32_t head_qo;
  uint32_t head_kv;
  uint32_t gqa_layout;
  uint32_t seq_len_q;
  uint32_t seq_len_k;
  uint32_t max_seq_len_q;
  uint32_t max_seq_len_k;
  uint32_t head_dim;
  uint32_t mask_type;
  uint32_t mask_layout;
  uint32_t num_mask_q_tiles;
  float scale;
  uint32_t aligned_seq_len_k;
  uint32_t tile_qo;
  uint32_t tile_kv;
  uint32_t pipeline_stages;
};
