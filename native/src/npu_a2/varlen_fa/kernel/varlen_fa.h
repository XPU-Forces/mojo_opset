#pragma once

#include "varlen_fa_block_mmad_qk_and_pv.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"
#include "varlen_fa_softmax_and_aggregate.h"

namespace xpu_ops::kernels {

struct CrossCoreEventSet {
  static constexpr uint32_t QK_READY = 0;
  static constexpr uint32_t SCORE_READY = 1;
  static constexpr uint32_t PV_READY = 2;
  static constexpr uint32_t KV_HEAD_PHASE_SYNC = 3;
};

struct FlashAttentionScheduleInfo {
  uint32_t batch_idx, head_idx, start_qo_pos;
};

struct FaSubTaskInfo {
  uint32_t batch_idx, head_idx, kv_head_idx, start_qo_pos, valid_qo_len, sub_core_start_qo_pos, sub_core_valid_qo_len,
      start_kv_pos, valid_kv_len;
  uint32_t task_idx, global_idx;
  bool is_first_task, is_last_task;
};

class FlashAttentionScheduler {
public:
  CATLASS_DEVICE
  FlashAttentionScheduler(uint32_t batch_size, uint32_t head_num, uint32_t seq_len_qo, uint32_t tile_qo,
                          uint32_t num_core)
      : batch_size_(batch_size), head_num_(head_num), seq_len_qo_(seq_len_qo), tile_qo_(tile_qo), num_core_(num_core) {
    num_tile_qo_ = (seq_len_qo_ + tile_qo_ - 1) / tile_qo_;
    task_num_ = batch_size_ * head_num_ * num_tile_qo_;
    stride_ = 2 * num_core_;
  }

  CATLASS_DEVICE
  uint32_t taskNum() const { return task_num_; }

  CATLASS_DEVICE
  FlashAttentionScheduleInfo getTaskInfo(uint32_t task_id) const {
    FlashAttentionScheduleInfo info;
    uint32_t adjusted_task_id = 0;
    if (task_id < task_num_ / stride_ * stride_) {
      uint32_t tail = task_id % stride_;
      if (tail >= num_core_) {
        tail = stride_ - (tail - num_core_) - 1;
      }
      adjusted_task_id = task_id / stride_ * stride_ + tail;
    } else {
      adjusted_task_id = task_id;
    }
    info.batch_idx = adjusted_task_id / (head_num_ * num_tile_qo_);
    info.head_idx = (adjusted_task_id / num_tile_qo_) % head_num_;
    info.start_qo_pos = (adjusted_task_id % num_tile_qo_) * tile_qo_;
    return info;
  }

private:
  uint32_t batch_size_, head_num_, seq_len_qo_;
  uint32_t tile_qo_, num_core_, stride_, num_tile_qo_;
  uint32_t task_num_;
};

class ParallelFlashAttentionScheduler {
public:
  CATLASS_DEVICE
  ParallelFlashAttentionScheduler(uint32_t batch_size, uint32_t head_num, uint32_t seq_len_qo, uint32_t tile_qo,
                                  uint32_t num_core, uint32_t core_idx)
      : batch_size_(batch_size),
        head_num_(head_num),
        seq_len_qo_(seq_len_qo),
        tile_qo_(tile_qo),
        num_core_(num_core),
        core_idx_(core_idx) {
    num_tile_qo_ = (seq_len_qo_ + tile_qo_ - 1) / tile_qo_;
    task_num_ = batch_size_ * head_num_ * num_tile_qo_;
    stride_ = 2 * num_core_;

    this_core_task_num_ = (task_num_ / stride_) * 2;
    uint32_t remainder = task_num_ % stride_;

    if (remainder <= num_core_) {
      if (core_idx_ < remainder) {
        this_core_task_num_ += 1;
      }
    } else {
      this_core_task_num_ += 1;
      remainder -= num_core_;
      if (core_idx_ >= num_core_ - remainder) {
        this_core_task_num_ += 1;
      }
    }
  }

  CATLASS_DEVICE
  uint32_t taskNum() const { return task_num_; }

  CATLASS_DEVICE
  uint32_t thisCoreTaskNum() const { return this_core_task_num_; }

  CATLASS_DEVICE
  FlashAttentionScheduleInfo getTaskInfo(uint32_t this_core_task_id) const {
    FlashAttentionScheduleInfo info;
    uint32_t task_id = (this_core_task_id / 2) * stride_;
    if (this_core_task_id % 2 == 1) {
      task_id += stride_ - core_idx_ - 1;
    } else {
      task_id += core_idx_;
    }
    info.batch_idx = task_id / (head_num_ * num_tile_qo_);
    info.head_idx = (task_id / num_tile_qo_) % head_num_;
    info.start_qo_pos = (task_id % num_tile_qo_) * tile_qo_;
    return info;
  }

private:
  uint32_t batch_size_, head_num_, seq_len_qo_, tile_qo_;
  uint32_t num_core_, core_idx_, stride_, num_tile_qo_;
  uint32_t task_num_, this_core_task_num_;
};

template <class ArchTag_, class ElementInOut_, class ElementCalc_, class ElementMask_, class TileSizeFa_,
          bool IS_CAUSAL_, uint32_t MMAD_L1_STAGE_NUM_ = 2, uint32_t MMAD_L0_STAGE_NUM_ = 2>
class VarlenFa {
public:
  using ArchTag = ArchTag_;
  using ElementInOut = ElementInOut_;
  using ElementCalc = ElementCalc_;
  using ElementMask = ElementMask_;
  using TileSizeFa = TileSizeFa_;

  static constexpr bool IS_CAUSAL = IS_CAUSAL_;
  static constexpr uint32_t MMAD_L1_STAGE_NUM = MMAD_L1_STAGE_NUM_;
  static constexpr uint32_t MMAD_L0_STAGE_NUM = MMAD_L0_STAGE_NUM_;
  static constexpr uint32_t L0_REDUCE_SIZE_QK = 64;
  static constexpr uint32_t L0_REDUCE_SIZE_PV = CUSTOM_FA_L0_REDUCE_SIZE_PV;

  static constexpr uint32_t BLOCK_QO = TileSizeFa::M;
  static constexpr uint32_t BLOCK_KV = TileSizeFa::N;
  static constexpr uint32_t HEAD_DIM = TileSizeFa::K;
  static constexpr uint32_t MAX_PIPELINE_STAGE = 3;
  static constexpr uint32_t MASK_LAYOUT_DENSE = 0;
  static constexpr uint32_t MASK_LAYOUT_PACKED_QTASK = 1;
  static constexpr uint32_t MASK_LAYOUT_CAUSAL_TEMPLATE = 2;
  static constexpr uint32_t PACKED_MASK_TILES_PER_QTASK = 2;
  static constexpr uint32_t GQA_LAYOUT_AABB = 0;
  static constexpr uint32_t GQA_LAYOUT_ABAB = 1;

  static constexpr uint32_t QK_READY = CrossCoreEventSet::QK_READY;
  static constexpr uint32_t SCORE_READY = CrossCoreEventSet::SCORE_READY;
  static constexpr uint32_t PV_READY = CrossCoreEventSet::PV_READY;
  static constexpr uint32_t KV_HEAD_PHASE_SYNC = CrossCoreEventSet::KV_HEAD_PHASE_SYNC;

  using MmadPolicy = QKAndPVPolicy<true, false, MMAD_L1_STAGE_NUM, MMAD_L0_STAGE_NUM, IS_CAUSAL>;
  using L0TileSizeQK = Catlass::GemmShape<BLOCK_QO, BLOCK_KV, L0_REDUCE_SIZE_QK>;
  using L0TileSizePV = Catlass::GemmShape<BLOCK_QO, HEAD_DIM, L0_REDUCE_SIZE_PV>;

  using BlockMmad = BlockMmadQKAndPV<ArchTag, ElementInOut, ElementCalc, MmadPolicy, CrossCoreEventSet, TileSizeFa,
                                     L0TileSizeQK, L0TileSizePV>;
  using SoftmaxAndAggregatePolicy = SoftmaxAndAggregateTileSize<BLOCK_QO, HEAD_DIM, IS_CAUSAL, MAX_PIPELINE_STAGE>;
  using SoftmaxAndAggregateType = SoftmaxAndAggregate<ArchTag, ElementInOut, ElementCalc, ElementMask,
                                                      SoftmaxAndAggregatePolicy, CrossCoreEventSet>;
  using RowMajor = Catlass::layout::RowMajor;
  using ColumnMajor = Catlass::layout::ColumnMajor;

  CATLASS_DEVICE VarlenFa() = default;

  CATLASS_DEVICE
  void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR mask, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k,
            GM_ADDR output, GM_ADDR workspace, const VarlenFaTiling &tiling) {
    batch_size = tiling.batch_size;
    total_q_tokens = tiling.total_q_tokens;
    total_k_tokens = tiling.total_k_tokens;
    head_qo = tiling.head_qo;
    head_kv = tiling.head_kv;
    gqa_layout = tiling.gqa_layout;
    gqa_group_size = head_qo / head_kv;
    gqa_extra_groups = head_qo % head_kv;
    gqa_large_group_end = gqa_extra_groups * (gqa_group_size + 1);
    seq_len_q = tiling.seq_len_q;
    seq_len_k = tiling.seq_len_k;
    max_seq_len_q = tiling.max_seq_len_q;
    max_seq_len_k = tiling.max_seq_len_k;
    seq_len_k_aligned = tiling.aligned_seq_len_k;
    mask_layout = tiling.mask_layout;
    num_mask_q_tiles = tiling.num_mask_q_tiles;
    scale = tiling.scale;

    tile_qo = tiling.tile_qo;
    tile_kv = tiling.tile_kv;
    pipeline_stages = tiling.pipeline_stages;

    XPU_OPS_DEBUG_PRINT(
        "VarlenFa::Init called with batch_size=%u, total_q_tokens=%u, total_k_tokens=%u, head_qo=%u, head_kv=%u, "
        "seq_len_q=%u, seq_len_k=%u, max_seq_len_q=%u, max_seq_len_k=%u, seq_len_k_aligned=%u, gqa_layout=%u, "
        "mask_layout=%u, num_mask_q_tiles=%u, tile_qo=%u, tile_kv=%u, pipeline_stages=%u, scale=%f\n",
        batch_size, total_q_tokens, total_k_tokens, head_qo, head_kv, seq_len_q, seq_len_k, max_seq_len_q,
        max_seq_len_k, seq_len_k_aligned, gqa_layout, mask_layout, num_mask_q_tiles, tile_qo, tile_kv,
        pipeline_stages, scale);

    query_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementInOut *>(query));
    key_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementInOut *>(key));
    value_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementInOut *>(value));
    mask_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMask *>(mask));
    cu_seqlens_q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens_q));
    cu_seqlens_k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens_k));
    output_gm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementInOut *>(output));
#if CUSTOM_FA_L2_BYPASS_QUERY
    query_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
#endif
#if CUSTOM_FA_L2_BYPASS_OUTPUT
    output_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
#endif

    qk_single_size = tile_kv * BLOCK_QO;
    score_single_size = tile_kv * BLOCK_QO;
    pv_single_size = HEAD_DIM * BLOCK_QO;

    uint32_t qk_buf_size = qk_single_size * sizeof(ElementCalc);
    uint32_t score_buf_size = score_single_size * sizeof(ElementInOut);
    uint32_t pv_buf_size = pv_single_size * sizeof(ElementCalc);

    uint32_t workspace_size_per_core = (qk_buf_size + score_buf_size + pv_buf_size) * pipeline_stages;
    workspace += (AscendC::GetBlockIdx() / AscendC::GetSubBlockNum()) * workspace_size_per_core;

    qk_pingpong_buf.SetGlobalBuffer(reinterpret_cast<__gm__ ElementCalc *>(workspace));
    score_pingpong_buf.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementInOut *>(workspace + qk_buf_size * pipeline_stages));
    pv_pingpong_buf.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementCalc *>(workspace + (qk_buf_size + score_buf_size) * pipeline_stages));
  }

  CATLASS_DEVICE bool usePackedMask() const { return mask_layout == MASK_LAYOUT_PACKED_QTASK; }

  CATLASS_DEVICE bool useCausalTemplateMask() const { return mask_layout == MASK_LAYOUT_CAUSAL_TEMPLATE; }

  CATLASS_DEVICE uint32_t kvHeadIdx(uint32_t head_idx) const {
    if (gqa_layout == GQA_LAYOUT_ABAB) {
      return head_idx % head_kv;
    }
    // The first Hq % Hkv contiguous groups each own one extra Q head.
    if (head_idx < gqa_large_group_end) {
      return head_idx / (gqa_group_size + 1);
    }
    return gqa_extra_groups + (head_idx - gqa_large_group_end) / gqa_group_size;
  }

  CATLASS_DEVICE uint32_t batchQueryStart(uint32_t batch_idx) const {
    return static_cast<uint32_t>(cu_seqlens_q_gm.GetValue(batch_idx));
  }

  CATLASS_DEVICE uint32_t batchKeyStart(uint32_t batch_idx) const {
    return static_cast<uint32_t>(cu_seqlens_k_gm.GetValue(batch_idx));
  }

  CATLASS_DEVICE uint32_t batchQueryLen(uint32_t batch_idx) const {
    return static_cast<uint32_t>(cu_seqlens_q_gm.GetValue(batch_idx + 1) - cu_seqlens_q_gm.GetValue(batch_idx));
  }

  CATLASS_DEVICE uint32_t batchKeyLen(uint32_t batch_idx) const {
    return static_cast<uint32_t>(cu_seqlens_k_gm.GetValue(batch_idx + 1) - cu_seqlens_k_gm.GetValue(batch_idx));
  }

  CATLASS_DEVICE uint32_t denseMaskOffset(uint32_t batch_idx, uint32_t start_qo_pos, uint32_t sub_core_start_qo_pos,
                                          uint32_t start_kv_pos) const {
    return (batch_idx * max_seq_len_q + start_qo_pos + sub_core_start_qo_pos) * seq_len_k_aligned + start_kv_pos;
  }

  CATLASS_DEVICE uint32_t packedMaskOffset(uint32_t batch_idx, uint32_t start_qo_pos, uint32_t mask_slot_idx,
                                           uint32_t sub_core_start_qo_pos) const {
    uint32_t q_task_idx = start_qo_pos / tile_qo;
    return ((((batch_idx * num_mask_q_tiles + q_task_idx) * PACKED_MASK_TILES_PER_QTASK + mask_slot_idx) * tile_qo) +
            sub_core_start_qo_pos) *
           tile_kv;
  }

  CATLASS_DEVICE uint32_t causalTemplateMaskOffset(uint32_t start_qo_pos, uint32_t sub_core_start_qo_pos) const {
    return ((start_qo_pos + sub_core_start_qo_pos) % tile_kv) * tile_kv;
  }

  CATLASS_DEVICE uint32_t firstPackedBoundaryTileIdx(uint32_t start_qo_pos, uint32_t batch_seq_len_q,
                                                     uint32_t batch_seq_len_k) const {
    int64_t diff = static_cast<int64_t>(batch_seq_len_k) - static_cast<int64_t>(batch_seq_len_q);
    int64_t first_tile = (static_cast<int64_t>(start_qo_pos) + diff) / static_cast<int64_t>(tile_kv);
    return first_tile > 0 ? static_cast<uint32_t>(first_tile) : 0;
  }

  CATLASS_DEVICE uint32_t lastPackedBoundaryTileIdx(uint32_t start_qo_pos, uint32_t valid_qo_len,
                                                    uint32_t batch_seq_len_q, uint32_t batch_seq_len_k) const {
    int64_t diff = static_cast<int64_t>(batch_seq_len_k) - static_cast<int64_t>(batch_seq_len_q);
    uint32_t q_end = start_qo_pos + valid_qo_len;
    int64_t last_tile = (static_cast<int64_t>(q_end) - 1 + diff) / static_cast<int64_t>(tile_kv);
    return last_tile > 0 ? static_cast<uint32_t>(last_tile) : 0;
  }

  CATLASS_DEVICE int32_t packedMaskSlotIdx(uint32_t start_qo_pos, uint32_t valid_qo_len, uint32_t start_kv_pos,
                                           uint32_t batch_seq_len_q, uint32_t batch_seq_len_k) const {
    if (!IS_CAUSAL || !usePackedMask()) {
      return -1;
    }
    uint32_t tile_kv_idx = start_kv_pos / tile_kv;
    uint32_t first_tile_idx = firstPackedBoundaryTileIdx(start_qo_pos, batch_seq_len_q, batch_seq_len_k);
    if (tile_kv_idx == first_tile_idx) {
      return 0;
    }
    uint32_t last_tile_idx = lastPackedBoundaryTileIdx(start_qo_pos, valid_qo_len, batch_seq_len_q, batch_seq_len_k);
    if (tile_kv_idx == last_tile_idx && last_tile_idx != first_tile_idx) {
      return 1;
    }
    return -1;
  }

  template <int32_t CORE_TYPE = g_coreType>
  CATLASS_DEVICE void Process() {
    uint32_t num_core = AscendC::GetBlockNum();
    uint32_t core_idx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    ParallelFlashAttentionScheduler scheduler(batch_size, head_qo, seq_len_q, tile_qo, num_core, core_idx);
    uint32_t this_core_task_num = scheduler.thisCoreTaskNum();

    if constexpr (CORE_TYPE == AscendC::AIC) {
      typename BlockMmad::Params params = {tile_kv, pipeline_stages};
      BlockMmad block_mmad(params, resource);

      for (uint32_t this_core_task_idx = 0; this_core_task_idx < this_core_task_num; this_core_task_idx++) {
        auto task_info = scheduler.getTaskInfo(this_core_task_idx);
        uint32_t batch_idx = task_info.batch_idx;
        uint32_t head_idx = task_info.head_idx;
        uint32_t kv_head_idx = kvHeadIdx(head_idx);
        uint32_t start_qo_pos = task_info.start_qo_pos;
        uint32_t batch_q_start = batchQueryStart(batch_idx);
        uint32_t batch_k_start = batchKeyStart(batch_idx);
        uint32_t batch_seq_len_q = batchQueryLen(batch_idx);
        uint32_t batch_seq_len_k = batchKeyLen(batch_idx);
        if (start_qo_pos >= batch_seq_len_q) {
          continue;
        }

        uint32_t qo_end = (start_qo_pos + tile_qo <= batch_seq_len_q) ? (start_qo_pos + tile_qo) : batch_seq_len_q;
        uint32_t valid_len_qo = qo_end - start_qo_pos;
        int64_t diff = static_cast<int64_t>(batch_seq_len_k) - static_cast<int64_t>(batch_seq_len_q);
        int64_t causal_valid_len_kv = static_cast<int64_t>(qo_end) + diff;
        uint32_t valid_len_kv = IS_CAUSAL
                                    ? static_cast<uint32_t>(causal_valid_len_kv > 0
                                                                ? ((causal_valid_len_kv < batch_seq_len_k)
                                                                       ? causal_valid_len_kv
                                                                       : batch_seq_len_k)
                                                                : 0)
                                    : batch_seq_len_k;
        if (valid_len_kv == 0) {
          continue;
        }

        XPU_OPS_DEBUG_PRINT(
            "start process task: batch_idx=%u, head_idx=%u, start_qo_pos=%u, valid_len_qo=%u, valid_len_kv=%u\n",
            batch_idx, head_idx, start_qo_pos, valid_len_qo, valid_len_kv);

        uint32_t q_offset = ((batch_q_start + start_qo_pos) * head_qo + head_idx) * HEAD_DIM;
        uint32_t k_offset = (batch_k_start * head_kv + kv_head_idx) * HEAD_DIM;
        uint32_t v_offset = (batch_k_start * head_kv + kv_head_idx) * HEAD_DIM;

        RowMajor layout_q(valid_len_qo, HEAD_DIM, head_qo * HEAD_DIM);
        ColumnMajor layout_k(HEAD_DIM, valid_len_kv, head_kv * HEAD_DIM);
        RowMajor layout_v(valid_len_kv, HEAD_DIM, head_kv * HEAD_DIM);

        block_mmad(query_gm[q_offset], layout_q, key_gm[k_offset], layout_k, value_gm[v_offset], layout_v,
                   qk_pingpong_buf, score_pingpong_buf, pv_pingpong_buf);
      }
    } else if constexpr (CORE_TYPE == AscendC::AIV) {
      typename SoftmaxAndAggregateType::Params params = {tile_kv, pipeline_stages, scale};
      SoftmaxAndAggregateType softmax_and_aggregate(params, resource);

      for (uint32_t this_core_task_idx = 0; this_core_task_idx < this_core_task_num; this_core_task_idx++) {
        auto task_info = scheduler.getTaskInfo(this_core_task_idx);
        uint32_t batch_idx = task_info.batch_idx;
        uint32_t head_idx = task_info.head_idx;
        uint32_t kv_head_idx = kvHeadIdx(head_idx);
        uint32_t start_qo_pos = task_info.start_qo_pos;
        uint32_t batch_q_start = batchQueryStart(batch_idx);
        uint32_t batch_seq_len_q = batchQueryLen(batch_idx);
        uint32_t batch_seq_len_k = batchKeyLen(batch_idx);
        if (start_qo_pos >= batch_seq_len_q) {
          continue;
        }

        uint32_t qo_end = (start_qo_pos + tile_qo <= batch_seq_len_q) ? (start_qo_pos + tile_qo) : batch_seq_len_q;
        uint32_t valid_len_qo = qo_end - start_qo_pos;
        int64_t diff = static_cast<int64_t>(batch_seq_len_k) - static_cast<int64_t>(batch_seq_len_q);
        int64_t causal_valid_len_kv = static_cast<int64_t>(qo_end) + diff;
        uint32_t valid_len_kv = IS_CAUSAL
                                    ? static_cast<uint32_t>(causal_valid_len_kv > 0
                                                                ? ((causal_valid_len_kv < batch_seq_len_k)
                                                                       ? causal_valid_len_kv
                                                                       : batch_seq_len_k)
                                                                : 0)
                                    : batch_seq_len_k;
        if (valid_len_kv == 0) {
          continue;
        }

        XPU_OPS_DEBUG_PRINT(
            "start process task: batch_idx=%u, head_idx=%u, start_qo_pos=%u, valid_len_qo=%u, valid_len_kv=%u\n",
            batch_idx, head_idx, start_qo_pos, valid_len_qo, valid_len_kv);

        uint32_t m_offset = denseMaskOffset(batch_idx, start_qo_pos, 0, 0);
        uint32_t o_offset = ((batch_q_start + start_qo_pos) * head_qo + head_idx) * HEAD_DIM;

        RowMajor layout_mask(valid_len_qo, valid_len_kv, seq_len_k_aligned);
        RowMajor layout_o(valid_len_qo, HEAD_DIM, head_qo * HEAD_DIM);

        softmax_and_aggregate(qk_pingpong_buf, mask_gm[m_offset], layout_mask, score_pingpong_buf, pv_pingpong_buf,
                              output_gm[o_offset], layout_o, start_qo_pos);
      }
    }

    AscendC::PipeBarrier<PIPE_ALL>();
  }

  template <int32_t CORE_TYPE = g_coreType>
  CATLASS_DEVICE void pipelinedProcess() {
    uint32_t num_core = AscendC::GetBlockNum();
    uint32_t core_idx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    ParallelFlashAttentionScheduler scheduler(batch_size, head_qo, seq_len_q, tile_qo, num_core, core_idx);
    uint32_t this_core_task_num = scheduler.thisCoreTaskNum();

    if constexpr (CORE_TYPE == AscendC::AIC) {
      typename BlockMmad::Params params = {tile_kv, pipeline_stages};
      BlockMmad block_mmad(params, resource);

      FaSubTaskInfo sub_tasks[MAX_PIPELINE_STAGE];

      uint32_t current_kv_head_idx = 0;
      uint32_t num_tile_qo = (seq_len_q + tile_qo - 1) / tile_qo;
      bool enable_kv_head_sync = false;

      uint32_t global_task_idx = 0;
      uint32_t valid_task_idx = 0;
      for (uint32_t this_core_task_idx = 0; this_core_task_idx < this_core_task_num; this_core_task_idx++) {
        auto task_info = scheduler.getTaskInfo(this_core_task_idx);
        uint32_t batch_idx = task_info.batch_idx;
        uint32_t head_idx = task_info.head_idx;
        uint32_t kv_head_idx = kvHeadIdx(head_idx);
        if (enable_kv_head_sync && kv_head_idx != current_kv_head_idx) {
          AscendC::CrossCoreSetFlag<0x0, PIPE_MTE2>(KV_HEAD_PHASE_SYNC);
          AscendC::CrossCoreWaitFlag(KV_HEAD_PHASE_SYNC);
          current_kv_head_idx = kv_head_idx;
        }
        uint32_t start_qo_pos = task_info.start_qo_pos;
        uint32_t batch_q_start = batchQueryStart(batch_idx);
        uint32_t batch_k_start = batchKeyStart(batch_idx);
        uint32_t batch_seq_len_q = batchQueryLen(batch_idx);
        uint32_t batch_seq_len_k = batchKeyLen(batch_idx);
        if (start_qo_pos >= batch_seq_len_q) {
          continue;
        }

        uint32_t qo_end = (start_qo_pos + tile_qo <= batch_seq_len_q) ? (start_qo_pos + tile_qo) : batch_seq_len_q;
        uint32_t valid_len_qo = qo_end - start_qo_pos;
        int64_t diff = static_cast<int64_t>(batch_seq_len_k) - static_cast<int64_t>(batch_seq_len_q);
        int64_t causal_valid_len_kv = static_cast<int64_t>(qo_end) + diff;
        uint32_t valid_len_kv = IS_CAUSAL
                                    ? static_cast<uint32_t>(causal_valid_len_kv > 0
                                                                ? ((causal_valid_len_kv < batch_seq_len_k)
                                                                       ? causal_valid_len_kv
                                                                       : batch_seq_len_k)
                                                                : 0)
                                    : batch_seq_len_k;
        if (valid_len_kv == 0) {
          continue;
        }
        uint32_t num_tile_kv = (valid_len_kv + tile_kv - 1) / tile_kv;

        for (uint32_t tile_kv_idx = 0; tile_kv_idx < num_tile_kv; tile_kv_idx++) {
          uint32_t pingpong_idx = global_task_idx % pipeline_stages;
          sub_tasks[pingpong_idx] = {
              batch_idx,
              head_idx,
              kv_head_idx,
              start_qo_pos,
              valid_len_qo,
              0,
              0,
              tile_kv_idx * tile_kv,
              (tile_kv_idx == num_tile_kv - 1) ? (valid_len_kv - tile_kv_idx * tile_kv) : tile_kv,
              valid_task_idx,
              global_task_idx,
              (tile_kv_idx == 0),
              (tile_kv_idx == num_tile_kv - 1)};

          {
            auto &sub_task = sub_tasks[pingpong_idx];
            uint32_t q_offset =
                ((batch_q_start + sub_task.start_qo_pos) * head_qo + sub_task.head_idx) * HEAD_DIM;
            uint32_t k_offset =
                ((batch_k_start + sub_task.start_kv_pos) * head_kv + sub_task.kv_head_idx) * HEAD_DIM;

            RowMajor layout_q(sub_task.valid_qo_len, HEAD_DIM, head_qo * HEAD_DIM);
            ColumnMajor layout_k(HEAD_DIM, sub_task.valid_kv_len, head_kv * HEAD_DIM);

            XPU_OPS_DEBUG_PRINT(
                "start QK[%u-%u], batch_idx=%u, head_idx=%u, kv_head_idx=%u, start_qo_pos=%u, valid_len_qo=%u, "
                "start_kv_pos=%u, "
                "valid_len_kv=%u, is_first_tile=%u, is_last_tile=%u\n",
                sub_task.global_idx, sub_task.task_idx, sub_task.batch_idx, sub_task.head_idx, sub_task.kv_head_idx,
                sub_task.start_qo_pos, sub_task.valid_qo_len, sub_task.start_kv_pos, sub_task.valid_kv_len,
                sub_task.is_first_task, sub_task.is_last_task);
            block_mmad.partialMmadQK(query_gm[q_offset], layout_q, key_gm[k_offset], layout_k,
                                     qk_pingpong_buf[pingpong_idx * qk_single_size], sub_task.is_first_task,
                                     sub_task.is_last_task);
            XPU_OPS_DEBUG_PRINT("  QK[%u-%u] done.\n", sub_task.global_idx, sub_task.task_idx);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(QK_READY);
            XPU_OPS_DEBUG_PRINT("  QK[%u-%u] ready.\n", sub_task.global_idx, sub_task.task_idx);
          }

          if (global_task_idx >= pipeline_stages - 1) {
            uint32_t pv_pingpong_id = (global_task_idx - (pipeline_stages - 1)) % pipeline_stages;
            auto &sub_task = sub_tasks[pv_pingpong_id];

            uint32_t sub_task_batch_k_start = batchKeyStart(sub_task.batch_idx);
            uint32_t sub_task_batch_q_start = batchQueryStart(sub_task.batch_idx);
            uint32_t v_offset =
                ((sub_task_batch_k_start + sub_task.start_kv_pos) * head_kv + sub_task.kv_head_idx) * HEAD_DIM;
            uint32_t o_offset =
                ((sub_task_batch_q_start + sub_task.start_qo_pos) * head_qo + sub_task.head_idx) * HEAD_DIM;

            uint32_t padded_len_kv = (sub_task.valid_kv_len + BLOCK_KV - 1) / BLOCK_KV * BLOCK_KV;
            RowMajor layout_p(BLOCK_QO, padded_len_kv, padded_len_kv);
            RowMajor layout_v(sub_task.valid_kv_len, HEAD_DIM, head_kv * HEAD_DIM);
            RowMajor layout_o(sub_task.valid_qo_len, HEAD_DIM, head_qo * HEAD_DIM);

            XPU_OPS_DEBUG_PRINT(
                "start PV[%u-%u], batch_idx=%u, head_idx=%u, kv_head_idx=%u, start_qo_pos=%u, valid_len_qo=%u, "
                "start_kv_pos=%u, "
                "valid_len_kv=%u, is_first_tile=%u, is_last_tile=%u\n",
                sub_task.global_idx, sub_task.task_idx, sub_task.batch_idx, sub_task.head_idx, sub_task.kv_head_idx,
                sub_task.start_qo_pos, sub_task.valid_qo_len, sub_task.start_kv_pos, sub_task.valid_kv_len,
                sub_task.is_first_task, sub_task.is_last_task);
            // AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(SCORE_READY);
            block_mmad.partialMmadPV(score_pingpong_buf[pv_pingpong_id * score_single_size], layout_p,
                                     value_gm[v_offset], layout_v, pv_pingpong_buf[pv_pingpong_id * pv_single_size]);
            XPU_OPS_DEBUG_PRINT("  PV[%u-%u] done.\n", sub_task.global_idx, sub_task.task_idx);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PV_READY);
            XPU_OPS_DEBUG_PRINT("  PV[%u-%u] ready.\n", sub_task.global_idx, sub_task.task_idx);
          }
          ++global_task_idx;
        }
        ++valid_task_idx;
      }
      uint32_t remaining_tasks = pipeline_stages - 1 < global_task_idx ? pipeline_stages - 1 : global_task_idx;
      for (uint32_t i = 0; i < remaining_tasks; i++) {
        uint32_t pingpong_id = (global_task_idx - remaining_tasks) % pipeline_stages;
        auto &sub_task = sub_tasks[pingpong_id];

        uint32_t sub_task_batch_k_start = batchKeyStart(sub_task.batch_idx);
        uint32_t sub_task_batch_q_start = batchQueryStart(sub_task.batch_idx);
        uint32_t v_offset =
            ((sub_task_batch_k_start + sub_task.start_kv_pos) * head_kv + sub_task.kv_head_idx) * HEAD_DIM;
        uint32_t o_offset =
            ((sub_task_batch_q_start + sub_task.start_qo_pos) * head_qo + sub_task.head_idx) * HEAD_DIM;

        uint32_t padded_len_kv = (sub_task.valid_kv_len + BLOCK_KV - 1) / BLOCK_KV * BLOCK_KV;
        RowMajor layout_p(BLOCK_QO, padded_len_kv, padded_len_kv);
        RowMajor layout_v(sub_task.valid_kv_len, HEAD_DIM, head_kv * HEAD_DIM);
        RowMajor layout_o(sub_task.valid_qo_len, HEAD_DIM, head_qo * HEAD_DIM);

        XPU_OPS_DEBUG_PRINT(
            "start PV[%u-%u], batch_idx=%u, head_idx=%u, kv_head_idx=%u, start_qo_pos=%u, valid_len_qo=%u, "
            "start_kv_pos=%u, "
            "valid_len_kv=%u, is_first_tile=%u, is_last_tile=%u\n",
            sub_task.global_idx, sub_task.task_idx, sub_task.batch_idx, sub_task.head_idx, sub_task.kv_head_idx,
            sub_task.start_qo_pos, sub_task.valid_qo_len, sub_task.start_kv_pos, sub_task.valid_kv_len,
            sub_task.is_first_task, sub_task.is_last_task);
        // AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(SCORE_READY);
        block_mmad.partialMmadPV(score_pingpong_buf[pingpong_id * score_single_size], layout_p, value_gm[v_offset],
                                 layout_v, pv_pingpong_buf[pingpong_id * pv_single_size]);
        XPU_OPS_DEBUG_PRINT("  PV[%u-%u] done.\n", sub_task.global_idx, sub_task.task_idx);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PV_READY);
        XPU_OPS_DEBUG_PRINT("  PV[%u-%u] ready.\n", sub_task.global_idx, sub_task.task_idx);
        ++global_task_idx;
      }
    } else if constexpr (CORE_TYPE == AscendC::AIV) {
      uint32_t sub_core_num = AscendC::GetSubBlockNum();
      uint32_t sub_core_id = AscendC::GetSubBlockIdx();
      uint32_t sub_core_tile_qo = BLOCK_QO / sub_core_num;

      typename SoftmaxAndAggregateType::Params params = {tile_kv, pipeline_stages, scale};
      SoftmaxAndAggregateType softmax_and_aggregate(params, resource);

      FaSubTaskInfo sub_tasks[MAX_PIPELINE_STAGE];

      uint32_t global_task_idx = 0;
      uint32_t valid_task_idx = 0;
      for (uint32_t this_core_task_idx = 0; this_core_task_idx < this_core_task_num; this_core_task_idx++) {
        auto task_info = scheduler.getTaskInfo(this_core_task_idx);
        uint32_t batch_idx = task_info.batch_idx;
        uint32_t head_idx = task_info.head_idx;
        uint32_t kv_head_idx = kvHeadIdx(head_idx);
        uint32_t start_qo_pos = task_info.start_qo_pos;
        uint32_t batch_q_start = batchQueryStart(batch_idx);
        uint32_t batch_seq_len_q = batchQueryLen(batch_idx);
        uint32_t batch_seq_len_k = batchKeyLen(batch_idx);
        if (start_qo_pos >= batch_seq_len_q) {
          continue;
        }

        uint32_t qo_end = (start_qo_pos + tile_qo <= batch_seq_len_q) ? (start_qo_pos + tile_qo) : batch_seq_len_q;
        uint32_t valid_len_qo = qo_end - start_qo_pos;
        int64_t diff = static_cast<int64_t>(batch_seq_len_k) - static_cast<int64_t>(batch_seq_len_q);
        int64_t causal_valid_len_kv = static_cast<int64_t>(qo_end) + diff;
        uint32_t valid_len_kv = IS_CAUSAL
                                    ? static_cast<uint32_t>(causal_valid_len_kv > 0
                                                                ? ((causal_valid_len_kv < batch_seq_len_k)
                                                                       ? causal_valid_len_kv
                                                                       : batch_seq_len_k)
                                                                : 0)
                                    : batch_seq_len_k;
        if (valid_len_kv == 0) {
          continue;
        }
        uint32_t num_tile_kv = (valid_len_kv + tile_kv - 1) / tile_kv;

        // Each AIV owns one fixed half of the padded 128-row Q tile.
        uint32_t sub_core_start_qo_pos = sub_core_id * sub_core_tile_qo;
        uint32_t sub_core_valid_qo_len =
            sub_core_start_qo_pos < valid_len_qo
                ? ((valid_len_qo - sub_core_start_qo_pos < sub_core_tile_qo)
                       ? valid_len_qo - sub_core_start_qo_pos
                       : sub_core_tile_qo)
                : 0;

        for (uint32_t tile_kv_idx = 0; tile_kv_idx < num_tile_kv; tile_kv_idx++) {
          uint32_t pingpong_idx = global_task_idx % pipeline_stages;
          sub_tasks[pingpong_idx] = {
              batch_idx,
              head_idx,
              kv_head_idx,
              start_qo_pos,
              valid_len_qo,
              sub_core_start_qo_pos,
              sub_core_valid_qo_len,
              tile_kv_idx * tile_kv,
              (tile_kv_idx == num_tile_kv - 1) ? (valid_len_kv - tile_kv_idx * tile_kv) : tile_kv,
              valid_task_idx,
              global_task_idx,
              (tile_kv_idx == 0),
              (tile_kv_idx == num_tile_kv - 1)};

          {
            auto &sub_task = sub_tasks[pingpong_idx];

            uint32_t padded_len_kv = (sub_task.valid_kv_len + BLOCK_KV - 1) / BLOCK_KV * BLOCK_KV;
            RowMajor layout_qk(sub_core_tile_qo, padded_len_kv, padded_len_kv);
            uint32_t offset_qk =
                pingpong_idx * qk_single_size + sub_task.sub_core_start_qo_pos * layout_qk.stride(0);

            RowMajor layout_score(sub_core_tile_qo, padded_len_kv, padded_len_kv);
            uint32_t offset_score =
                pingpong_idx * score_single_size + sub_task.sub_core_start_qo_pos * layout_score.stride(0);

            XPU_OPS_DEBUG_PRINT(
                "Softmax[%u-%u], batch_idx=%u, head_idx=%u, kv_head_idx=%u, start_qo_pos=%u, "
                "valid_len_qo=%u, sub_core_start_qo_pos=%u, sub_core_valid_qo_len=%u, start_kv_pos=%u, "
                "valid_len_kv=%u, is_first_tile=%u, "
                "is_last_tile=%u\n",
                sub_task.global_idx, sub_task.task_idx, sub_task.batch_idx, sub_task.head_idx, sub_task.kv_head_idx,
                sub_task.start_qo_pos, sub_task.valid_qo_len, sub_task.sub_core_start_qo_pos,
                sub_task.sub_core_valid_qo_len, sub_task.start_kv_pos, sub_task.valid_kv_len, sub_task.is_first_task,
                sub_task.is_last_task);
            // AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(QK_READY);
            if constexpr (IS_CAUSAL) {
              bool tile_has_partial_mask = !usePackedMask() && !useCausalTemplateMask();
              uint32_t mask_stride = seq_len_k_aligned;
              uint32_t offset_mask = denseMaskOffset(batch_idx, sub_task.start_qo_pos, sub_task.sub_core_start_qo_pos,
                                                     sub_task.start_kv_pos);
              if (usePackedMask()) {
                int32_t mask_slot_idx =
                    packedMaskSlotIdx(sub_task.start_qo_pos, sub_task.valid_qo_len, sub_task.start_kv_pos,
                                      batch_seq_len_q, batch_seq_len_k);
                tile_has_partial_mask = mask_slot_idx >= 0;
                mask_stride = tile_kv;
                offset_mask = packedMaskOffset(batch_idx, sub_task.start_qo_pos,
                                               tile_has_partial_mask ? static_cast<uint32_t>(mask_slot_idx) : 0,
                                               sub_task.sub_core_start_qo_pos);
              } else if (useCausalTemplateMask()) {
                uint32_t tile_kv_idx = sub_task.start_kv_pos / tile_kv;
                uint32_t first_tile_idx =
                    firstPackedBoundaryTileIdx(sub_task.start_qo_pos, batch_seq_len_q, batch_seq_len_k);
                uint32_t last_tile_idx = lastPackedBoundaryTileIdx(
                    sub_task.start_qo_pos, sub_task.valid_qo_len, batch_seq_len_q, batch_seq_len_k);
                tile_has_partial_mask =
                    (tile_kv_idx == first_tile_idx) || (tile_kv_idx == last_tile_idx && last_tile_idx != first_tile_idx);
                mask_stride = tile_kv;
                offset_mask = causalTemplateMaskOffset(sub_task.start_qo_pos, sub_task.sub_core_start_qo_pos);
              }
              if (tile_has_partial_mask) {
                RowMajor layout_mask(sub_task.sub_core_valid_qo_len, sub_task.valid_kv_len, mask_stride);
                softmax_and_aggregate.subCoreOnlineSoftmax(
                    qk_pingpong_buf[offset_qk], layout_qk, mask_gm[offset_mask], layout_mask,
                    score_pingpong_buf[offset_score], layout_score,
                    sub_task.sub_core_start_qo_pos + sub_task.start_qo_pos, sub_task.start_kv_pos, pingpong_idx,
                    sub_task.task_idx % pipeline_stages, sub_task.is_first_task, sub_task.is_last_task,
                    tile_has_partial_mask);
              } else {
                softmax_and_aggregate.subCoreOnlineSoftmaxNoMask(
                    qk_pingpong_buf[offset_qk], layout_qk, score_pingpong_buf[offset_score], layout_score,
                    sub_task.sub_core_start_qo_pos + sub_task.start_qo_pos, sub_task.start_kv_pos, pingpong_idx,
                    sub_task.task_idx % pipeline_stages, sub_task.is_first_task, sub_task.is_last_task,
                    sub_task.valid_kv_len);
              }
            } else {
              softmax_and_aggregate.subCoreOnlineSoftmaxNoMask(
                  qk_pingpong_buf[offset_qk], layout_qk, score_pingpong_buf[offset_score], layout_score,
                  sub_task.sub_core_start_qo_pos + sub_task.start_qo_pos, sub_task.start_kv_pos, pingpong_idx,
                  sub_task.task_idx % pipeline_stages, sub_task.is_first_task, sub_task.is_last_task,
                  sub_task.valid_kv_len);
            }
            XPU_OPS_DEBUG_PRINT("  Softmax[%u-%u] done.\n", sub_task.global_idx, sub_task.task_idx);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SCORE_READY);
            XPU_OPS_DEBUG_PRINT("  Softmax[%u-%u] ready.\n", sub_task.global_idx, sub_task.task_idx);
          }

          if (global_task_idx >= pipeline_stages - 1) {
            uint32_t agg_pingpong_id = (global_task_idx - (pipeline_stages - 1)) % pipeline_stages;

            auto &sub_task = sub_tasks[agg_pingpong_id];

            RowMajor layout_pv(sub_core_tile_qo, HEAD_DIM, HEAD_DIM);
            uint32_t offset_pv =
                agg_pingpong_id * pv_single_size + sub_task.sub_core_start_qo_pos * layout_pv.stride(0);

            uint32_t sub_task_batch_q_start = batchQueryStart(sub_task.batch_idx);
            RowMajor layout_o(sub_task.sub_core_valid_qo_len, HEAD_DIM, head_qo * HEAD_DIM);
            uint32_t o_offset =
                ((sub_task_batch_q_start + sub_task.start_qo_pos + sub_task.sub_core_start_qo_pos) * head_qo +
                 sub_task.head_idx) *
                HEAD_DIM;

            XPU_OPS_DEBUG_PRINT(
                "Aggregate[%u-%u], batch_idx=%u, head_idx=%u, kv_head_idx=%u, start_qo_pos=%u, "
                "valid_len_qo=%u, sub_core_start_qo_pos=%u, sub_core_valid_qo_len=%u, start_kv_pos=%u, "
                "valid_len_kv=%u, is_first_tile=%u, "
                "is_last_tile=%u\n",
                sub_task.global_idx, sub_task.task_idx, sub_task.batch_idx, sub_task.head_idx, sub_task.kv_head_idx,
                sub_task.start_qo_pos, sub_task.valid_qo_len, sub_task.sub_core_start_qo_pos,
                sub_task.sub_core_valid_qo_len, sub_task.start_kv_pos, sub_task.valid_kv_len, sub_task.is_first_task,
                sub_task.is_last_task);
            // AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(PV_READY);
            softmax_and_aggregate.subCoreAggregate(pv_pingpong_buf[offset_pv], layout_pv, output_gm[o_offset], layout_o,
                                                   agg_pingpong_id, sub_task.task_idx % pipeline_stages,
                                                   sub_task.is_first_task, sub_task.is_last_task);
            XPU_OPS_DEBUG_PRINT("  Aggregate[%u-%u] done.\n", sub_task.global_idx, sub_task.task_idx);
          }
          ++global_task_idx;
        }
        ++valid_task_idx;
      }

      uint32_t remaining_tasks = pipeline_stages - 1 < global_task_idx ? pipeline_stages - 1 : global_task_idx;
      for (uint32_t i = 0; i < remaining_tasks; i++) {
        uint32_t pingpong_id = (global_task_idx - remaining_tasks) % pipeline_stages;

        auto &sub_task = sub_tasks[pingpong_id];

        RowMajor layout_pv(sub_core_tile_qo, HEAD_DIM, HEAD_DIM);
        uint32_t offset_pv = pingpong_id * pv_single_size + sub_task.sub_core_start_qo_pos * layout_pv.stride(0);

        uint32_t sub_task_batch_q_start = batchQueryStart(sub_task.batch_idx);
        RowMajor layout_o(sub_task.sub_core_valid_qo_len, HEAD_DIM, head_qo * HEAD_DIM);
        uint32_t o_offset =
            ((sub_task_batch_q_start + sub_task.start_qo_pos + sub_task.sub_core_start_qo_pos) * head_qo +
             sub_task.head_idx) *
            HEAD_DIM;

        XPU_OPS_DEBUG_PRINT(
            "Aggregate[%u-%u], batch_idx=%u, head_idx=%u, kv_head_idx=%u, start_qo_pos=%u, "
            "valid_len_qo=%u, sub_core_start_qo_pos=%u, sub_core_valid_qo_len=%u, start_kv_pos=%u, "
            "valid_len_kv=%u, is_first_tile=%u, "
            "is_last_tile=%u\n",
            sub_task.global_idx, sub_task.task_idx, sub_task.batch_idx, sub_task.head_idx, sub_task.kv_head_idx,
            sub_task.start_qo_pos, sub_task.valid_qo_len, sub_task.sub_core_start_qo_pos,
            sub_task.sub_core_valid_qo_len, sub_task.start_kv_pos, sub_task.valid_kv_len, sub_task.is_first_task,
            sub_task.is_last_task);
        // AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(PV_READY);
        softmax_and_aggregate.subCoreAggregate(pv_pingpong_buf[offset_pv], layout_pv, output_gm[o_offset], layout_o,
                                               pingpong_id, sub_task.task_idx % pipeline_stages, sub_task.is_first_task,
                                               sub_task.is_last_task);
        XPU_OPS_DEBUG_PRINT("  Aggregate[%u-%u] done.\n", sub_task.global_idx, sub_task.task_idx);

        ++global_task_idx;
      }
    }

    AscendC::PipeBarrier<PIPE_ALL>();
  }

private:
  uint32_t batch_size, total_q_tokens, total_k_tokens, head_qo, head_kv, gqa_layout, seq_len_q, seq_len_k, max_seq_len_q,
      max_seq_len_k, seq_len_k_aligned, mask_layout, num_mask_q_tiles;
  uint32_t gqa_group_size, gqa_extra_groups, gqa_large_group_end;
  float scale;
  uint32_t tile_qo, tile_kv, pipeline_stages;

  uint32_t qk_single_size, score_single_size, pv_single_size;

  GM_ADDR workspace_gm;

  AscendC::GlobalTensor<ElementInOut> query_gm;
  AscendC::GlobalTensor<ElementInOut> key_gm;
  AscendC::GlobalTensor<ElementInOut> value_gm;
  AscendC::GlobalTensor<ElementMask> mask_gm;
  AscendC::GlobalTensor<int64_t> cu_seqlens_q_gm;
  AscendC::GlobalTensor<int64_t> cu_seqlens_k_gm;
  AscendC::GlobalTensor<ElementInOut> output_gm;
  AscendC::GlobalTensor<ElementCalc> qk_pingpong_buf;
  AscendC::GlobalTensor<ElementInOut> score_pingpong_buf;
  AscendC::GlobalTensor<ElementCalc> pv_pingpong_buf;

  Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace xpu_ops::kernels
