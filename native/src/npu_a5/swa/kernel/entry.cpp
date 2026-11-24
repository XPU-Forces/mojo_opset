#include "kernel_operator.h"
#include "kernel/mask_helper.h"
#include "kernel/aic.cpp"
#include "kernel/aiv.cpp"

#ifndef SWA_V5_ENTRY_KERNEL
#define SWA_V5_ENTRY_KERNEL swa_v5_full_path
#endif

#ifndef SWA_V5_ENTRY_BATCH
#define SWA_V5_ENTRY_BATCH 5
#endif

#ifndef SWA_V5_ENTRY_NUM_Q_HEADS
#define SWA_V5_ENTRY_NUM_Q_HEADS 8
#endif

#ifndef SWA_V5_ENTRY_NUM_KV_HEADS
#define SWA_V5_ENTRY_NUM_KV_HEADS 2
#endif

#ifndef SWA_V5_ENTRY_MAX_Q_LEN
#define SWA_V5_ENTRY_MAX_Q_LEN 384
#endif

#ifndef SWA_V5_ENTRY_MAX_KV_LEN
#define SWA_V5_ENTRY_MAX_KV_LEN 512
#endif

#ifndef SWA_V5_ENTRY_DIM
#define SWA_V5_ENTRY_DIM 128
#endif

#ifndef SWA_V5_ENTRY_WINDOW_LEFT
#define SWA_V5_ENTRY_WINDOW_LEFT 128
#endif

#ifndef SWA_V5_ENTRY_WINDOW_RIGHT
#define SWA_V5_ENTRY_WINDOW_RIGHT 0
#endif

#ifndef SWA_V5_ENTRY_GLOBAL_WINDOW_SIZE
#define SWA_V5_ENTRY_GLOBAL_WINDOW_SIZE 0
#endif

#ifndef SWA_V5_ENTRY_IS_CAUSAL
#define SWA_V5_ENTRY_IS_CAUSAL 1
#endif

#ifndef SWA_V5_ENTRY_GQA_LAYOUT_ABAB
#define SWA_V5_ENTRY_GQA_LAYOUT_ABAB 0
#endif

#ifndef SWA_V5_ENTRY_OUTPUT_FP16
#define SWA_V5_ENTRY_OUTPUT_FP16 0
#endif
#ifndef SWA_V5_ENTRY_INPUT_BF16
#define SWA_V5_ENTRY_INPUT_BF16 0
#endif
#ifndef SWA_V5_ENTRY_OUTPUT_BF16
#define SWA_V5_ENTRY_OUTPUT_BF16 0
#endif

namespace SWA_v5::Pipeline::FullPath {

using SWA_v5::GMTensor_ND;

static constexpr uint32_t TILE_Q = 128;
static constexpr uint32_t TILE_L0 = 128;
static constexpr uint32_t TILE_L1 = 256;
static constexpr uint32_t L2_SEQ_WAVE_TILES = 55U;
static constexpr uint32_t L2_SEQ_WAVE_PROMOTION_TILES = 8U;
static constexpr uint32_t DIM = SWA_V5_ENTRY_DIM;
static_assert(DIM == 128, "v5 full_path currently fixes head_dim=128");
static_assert(TILE_L1 % TILE_L0 == 0 && TILE_L1 >= TILE_L0,
    "v5 full_path expects TILE_L1 to be a multiple of TILE_L0");
static_assert(CUBE2_QK_CONSUMED_BASE + CUBE2_QK_SLOTS - 1U < CUBE2_P_READY_BASE,
    "QK consumed events must not overlap P-ready events");
static_assert(CUBE2_BMM2_READY_BASE + CUBE2_BMM2_SLOTS - 1U <= 10U,
    "cube2 event ids must stay in 0-10 and 16-26");

struct TaskInfo {
    uint32_t linear_q_tile;
    uint32_t batch_idx;
    uint32_t q_head;
    uint32_t kv_head;
    uint32_t q_tile;
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t kv_computed_len;
    uint32_t q_global_begin;
    uint32_t kv_global_begin;
    uint32_t q_rows;
    uint32_t start_l0_tile;
    uint32_t second_l0_tile;
    uint32_t last_l0_tile;
    uint32_t first_l0_count;
    uint32_t valid_l0_count;
};

struct Vec2TaskInfo {
    uint32_t q_global_begin;
    uint32_t q_head;
    uint32_t q_rows;
    uint32_t valid_l0_count;
};

template<uint32_t CAPACITY>
struct TaskMetadata {
    uint32_t batch_begin;
    uint32_t q_lens[CAPACITY + 1U];
    uint32_t kv_lens[CAPACITY + 1U];
};

__aicore__ inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

__aicore__ inline uint32_t map_kv_head(
    uint32_t q_head, uint32_t num_kv_heads, uint32_t share_q_heads, bool is_abab) {
    return is_abab ? q_head % num_kv_heads : q_head / share_q_heads;
}

__aicore__ inline uint32_t l0_tile_at(const TaskInfo& task, uint32_t ordinal) {
    return MaskHelper::l0_tile_at(
        task.start_l0_tile, task.second_l0_tile,
        task.first_l0_count, ordinal);
}

__aicore__ inline bool has_l0_gap(const TaskInfo& task) {
    return !MaskHelper::is_l0_range_contiguous(
        task.start_l0_tile, task.second_l0_tile,
        task.first_l0_count, task.valid_l0_count);
}

__aicore__ inline uint32_t local_l1_idx(const TaskInfo& task, uint32_t ordinal) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (!has_l0_gap(task) || ordinal < task.first_l0_count) {
        return ordinal / L1_PARTS;
    }
    return ceil_div(task.first_l0_count, L1_PARTS) +
        (ordinal - task.first_l0_count) / L1_PARTS;
}

__aicore__ inline uint32_t part_idx(const TaskInfo& task, uint32_t local_l0_idx) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (has_l0_gap(task) && local_l0_idx >= task.first_l0_count) {
        return (local_l0_idx - task.first_l0_count) % L1_PARTS;
    }
    return local_l0_idx % L1_PARTS;
}

__aicore__ inline uint32_t l1_group_start_local_l0_idx(const TaskInfo& task, uint32_t local_l0_idx) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (has_l0_gap(task) && local_l0_idx >= task.first_l0_count) {
        return task.first_l0_count +
            ((local_l0_idx - task.first_l0_count) / L1_PARTS) * L1_PARTS;
    }
    return (local_l0_idx / L1_PARTS) * L1_PARTS;
}

__aicore__ inline uint32_t num_parts(const TaskInfo& task, uint32_t local_l0_idx) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (has_l0_gap(task) && local_l0_idx == 0U) {
        return task.first_l0_count < L1_PARTS ?
            task.first_l0_count : L1_PARTS;
    }
    const uint32_t group_start = l1_group_start_local_l0_idx(task, local_l0_idx);
    const uint32_t segment_end =
        has_l0_gap(task) && group_start < task.first_l0_count ?
            task.first_l0_count : task.valid_l0_count;
    const uint32_t remaining_l0_parts = segment_end - group_start;
    return remaining_l0_parts < L1_PARTS ? remaining_l0_parts : L1_PARTS;
}

__aicore__ inline uint32_t l1_group_count(const TaskInfo& task) {
    if (task.valid_l0_count == 0U) {
        return 0U;
    }
    if (has_l0_gap(task)) {
        return ceil_div(task.first_l0_count, TILE_L1 / TILE_L0) +
            ceil_div(task.valid_l0_count - task.first_l0_count, TILE_L1 / TILE_L0);
    }
    return ceil_div(task.valid_l0_count, TILE_L1 / TILE_L0);
}

__aicore__ inline AicQTaskScalars make_aic_q_task_scalars(
    const TaskInfo& task, uint32_t q_offset, uint32_t kv_offset,
    uint32_t global_window_size) {
    return {
        task.linear_q_tile,
        q_offset,
        kv_offset,
        task.q_rows,
        task.kv_len,
        global_window_size,
        task.start_l0_tile,
        task.second_l0_tile,
        task.first_l0_count,
        task.valid_l0_count,
    };
}

__aicore__ inline AicL0TileScalars make_aic_l0_tile_scalars_contiguous(
    const TaskInfo& task, uint32_t local_l0_idx, uint32_t global_l0_idx, uint32_t global_l1_base_idx) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (local_l0_idx >= task.valid_l0_count) {
        return {local_l0_idx, global_l0_idx, global_l1_base_idx, 0U, 0U, 0U};
    }
    const uint32_t group_idx = local_l0_idx / L1_PARTS;
    const uint32_t group_start = group_idx * L1_PARTS;
    const uint32_t remaining_l0_parts = task.valid_l0_count - group_start;
    return {
        local_l0_idx,
        global_l0_idx,
        global_l1_base_idx + group_idx,
        group_idx,
        local_l0_idx % L1_PARTS,
        remaining_l0_parts < L1_PARTS ? remaining_l0_parts : L1_PARTS,
    };
}

__aicore__ inline AicL0TileScalars make_aic_l0_tile_scalars_sparse(
    const TaskInfo& task, uint32_t local_l0_idx, uint32_t global_l0_idx, uint32_t global_l1_base_idx) {
    if (local_l0_idx >= task.valid_l0_count) {
        return {local_l0_idx, global_l0_idx, global_l1_base_idx, 0U, 0U, 0U};
    }
    const uint32_t group_idx = local_l1_idx(task, local_l0_idx);
    return {
        local_l0_idx,
        global_l0_idx,
        global_l1_base_idx + group_idx,
        group_idx,
        part_idx(task, local_l0_idx),
        num_parts(task, local_l0_idx),
    };
}

__aicore__ inline AicL0TileScalars make_aic_l0_tile_scalars(
    const TaskInfo& task, uint32_t local_l0_idx, uint32_t global_l0_idx, uint32_t global_l1_base_idx) {
    return has_l0_gap(task) ?
        make_aic_l0_tile_scalars_sparse(task, local_l0_idx, global_l0_idx, global_l1_base_idx) :
        make_aic_l0_tile_scalars_contiguous(task, local_l0_idx, global_l0_idx, global_l1_base_idx);
}

__aicore__ inline bool fill_task_in_batch(
    uint32_t linear_q_tile, uint32_t wave_linear_begin,
    uint32_t q_wave_begin, uint32_t q_wave_count, uint32_t batch_idx,
    uint32_t q_begin, uint32_t q_len, uint32_t kv_begin, uint32_t kv_len,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size, bool is_abab,
    TaskInfo& task) {
    const uint32_t remaining = linear_q_tile - wave_linear_begin;
    const uint32_t share_q_heads = num_q_heads / num_kv_heads;
    const uint32_t q_head = remaining / q_wave_count;
    const uint32_t head_local_q_tile = remaining - q_head * q_wave_count;
    const uint32_t scheduled_q_tile = (q_head & 1U) == 0U ?
        head_local_q_tile : q_wave_count - 1U - head_local_q_tile;
    const uint32_t local_q_tile = scheduled_q_tile;
    const uint32_t q_tile = q_wave_begin + local_q_tile;
    const uint32_t q_tile_begin = q_tile * TILE_Q;
    const uint32_t q_rows = min_u32(TILE_Q, q_len - q_tile_begin);
    const uint32_t kv_computed_len = kv_len - q_len;
    const uint32_t q_abs_start = kv_computed_len + q_tile_begin;
    const L0TileRange l0_range =
        MaskHelper::make_l0_tile_range<TILE_L0, (SWA_V5_ENTRY_IS_CAUSAL != 0)>(
            q_abs_start, q_rows, kv_len, window_left, window_right, global_window_size);
    if (l0_range.valid_l0_count == 0U) {
        return false;
    }
    task.linear_q_tile = linear_q_tile;
    task.batch_idx = batch_idx;
    task.q_head = q_head;
    task.kv_head = map_kv_head(q_head, num_kv_heads, share_q_heads, is_abab);
    task.q_tile = q_tile;
    task.q_len = q_len;
    task.kv_len = kv_len;
    task.kv_computed_len = kv_computed_len;
    task.q_global_begin = q_begin + q_tile_begin;
    task.kv_global_begin = kv_begin;
    task.q_rows = q_rows;
    task.start_l0_tile = l0_range.start_l0_tile;
    task.second_l0_tile = l0_range.second_l0_tile;
    task.last_l0_tile = l0_range.last_l0_tile;
    task.first_l0_count = l0_range.first_l0_count;
    task.valid_l0_count = l0_range.valid_l0_count;
    return true;
}

template<uint32_t CAPACITY>
class TaskCursor {
public:
    __aicore__ inline TaskCursor(
        const TaskMetadata<CAPACITY>& metadata, uint32_t batch, uint32_t core_idx,
        uint32_t core_count, uint32_t num_q_heads, uint32_t num_kv_heads,
        uint32_t window_left, uint32_t window_right,
        uint32_t global_window_size, bool is_abab)
        : metadata_(metadata),
          batch_(batch),
          core_idx_(core_idx),
          core_count_(core_count),
          num_q_heads_(num_q_heads),
          num_kv_heads_(num_kv_heads),
          window_left_(window_left),
          window_right_(window_right),
          global_window_size_(global_window_size),
          is_abab_(is_abab) {}

    __aicore__ inline bool next(TaskInfo& task) {
        while (true) {
            if (next_linear_ < wave_linear_end_) {
                const uint32_t linear = next_linear_;
                next_linear_ += core_count_;
                if (fill_task_in_batch(
                        linear, wave_linear_begin_, q_wave_begin_,
                        q_wave_count_, current_batch_, q_begin_, q_len_,
                        kv_begin_, kv_len_, num_q_heads_, num_kv_heads_,
                        window_left_, window_right_, global_window_size_,
                        is_abab_, task)) {
                    return true;
                }
                continue;
            }
            if (next_q_tile_ < num_q_tiles_) {
                prepare_wave();
                continue;
            }
            if (next_batch_ >= batch_) {
                return false;
            }
            prepare_batch();
        }
    }

private:
    __aicore__ inline void prepare_batch() {
        const uint32_t local_batch = next_batch_;
        current_batch_ = metadata_.batch_begin + local_batch;
        ++next_batch_;
        q_begin_ = metadata_.q_lens[local_batch];
        q_len_ = metadata_.q_lens[local_batch + 1U] - q_begin_;
        kv_begin_ = metadata_.kv_lens[local_batch];
        kv_len_ = metadata_.kv_lens[local_batch + 1U] - kv_begin_;
        num_q_tiles_ = q_len_ > 0U && kv_len_ >= q_len_ ?
            ceil_div(q_len_, TILE_Q) : 0U;
        next_q_tile_ = 0U;
        q_wave_capacity_ = L2_SEQ_WAVE_TILES;
        const uint32_t wave_count =
            ceil_div(num_q_tiles_, L2_SEQ_WAVE_TILES);
        if (wave_count > 1U && wave_count <= 4U) {
            const uint32_t tail_tiles =
                num_q_tiles_ - (wave_count - 1U) * L2_SEQ_WAVE_TILES;
            const uint32_t promotion_limit =
                L2_SEQ_WAVE_TILES + L2_SEQ_WAVE_PROMOTION_TILES;
            if (tail_tiles <= L2_SEQ_WAVE_PROMOTION_TILES) {
                const uint32_t promoted_tiles =
                    ceil_div(num_q_tiles_, wave_count - 1U);
                if (promoted_tiles <= promotion_limit) {
                    q_wave_capacity_ = promoted_tiles;
                }
            }
        }
    }

    __aicore__ inline void prepare_wave() {
        q_wave_begin_ = next_q_tile_;
        q_wave_count_ = min_u32(
            q_wave_capacity_, num_q_tiles_ - next_q_tile_);
        next_q_tile_ += q_wave_count_;
        wave_linear_begin_ = total_linear_;
        wave_linear_end_ =
            wave_linear_begin_ + q_wave_count_ * num_q_heads_;
        const uint32_t owner_delta =
            (core_idx_ + core_count_ -
             wave_linear_begin_ % core_count_) % core_count_;
        next_linear_ = wave_linear_begin_ + owner_delta;
        total_linear_ = wave_linear_end_;
    }

    const TaskMetadata<CAPACITY>& metadata_;
    uint32_t batch_;
    uint32_t core_idx_;
    uint32_t core_count_;
    uint32_t num_q_heads_;
    uint32_t num_kv_heads_;
    uint32_t window_left_;
    uint32_t window_right_;
    uint32_t global_window_size_;
    bool is_abab_;
    uint32_t next_batch_ = 0U;
    uint32_t current_batch_ = 0U;
    uint32_t num_q_tiles_ = 0U;
    uint32_t next_q_tile_ = 0U;
    uint32_t q_wave_capacity_ = L2_SEQ_WAVE_TILES;
    uint32_t q_wave_begin_ = 0U;
    uint32_t q_wave_count_ = 0U;
    uint32_t total_linear_ = 0U;
    uint32_t wave_linear_begin_ = 0U;
    uint32_t wave_linear_end_ = 0U;
    uint32_t next_linear_ = 0U;
    uint32_t q_begin_ = 0U;
    uint32_t q_len_ = 0U;
    uint32_t kv_begin_ = 0U;
    uint32_t kv_len_ = 0U;
};

}  // namespace SWA_v5::Pipeline::FullPath

template<uint32_t TASK_METADATA_BATCHES, uint32_t VEC2_QUEUE_DEPTH, bool WRITE_OUTPUT_CACHE>
__global__ __aicore__ void SWA_V5_ENTRY_KERNEL(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR output_cache, GM_ADDR softmax_lse,
    SWA_v5::Pipeline::FullPath::TaskMetadata<TASK_METADATA_BATCHES> task_metadata,
    uint32_t batch, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t total_q_tokens,
    uint32_t total_kv_tokens, uint32_t max_q_len, uint32_t max_kv_len, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size, uint32_t gqa_layout,
    uint32_t num_aic_cores, uint32_t dim, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace SWA_v5::Pipeline::FullPath;
    if (num_aic_cores == 0U) {
        return;
    }
    static_assert(VEC2_QUEUE_DEPTH == 2U || VEC2_QUEUE_DEPTH == 3U, "unsupported Vec2 queue depth");

    if (num_q_heads == 0U || num_kv_heads == 0U || num_q_heads % num_kv_heads != 0U ||
        dim != DIM || max_q_len == 0U || max_kv_len == 0U ||
        (window_left == 0U && global_window_size == 0U) ||
        gqa_layout > 1U) {
        return;
    }
#if SWA_V5_ENTRY_INPUT_BF16
    using InputT = bfloat16_t;
#else
    using InputT = half;
#endif
#if SWA_V5_ENTRY_OUTPUT_BF16
    using OutputT = bfloat16_t;
#elif SWA_V5_ENTRY_OUTPUT_FP16
    using OutputT = half;
#else
    using OutputT = float;
#endif
    AICore_RT<TILE_Q, TILE_L0, TILE_L1, DIM, InputT> rt;
    static_assert(decltype(rt)::VEC_RT::EXPMAX_SLOTS >= VEC2_QUEUE_DEPTH + 2U,
        "paired Vec2 must retain both expmax values until consumption");
    GMTensor_ND<InputT> q_gm;
    GMTensor_ND<InputT> k_gm;
    GMTensor_ND<InputT> v_gm;
    GMTensor_ND<OutputT> out_gm;
    GMTensor_ND<float> output_cache_gm;
    GMTensor_ND<float> softmax_lse_gm;
    GMTensor_ND<uint8_t> atten_mask_gm;
    q_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(q));
    k_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(k));
    v_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(v));
    out_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(out));
    output_cache_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output_cache));
    softmax_lse_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmax_lse));
    if constexpr (g_coreType == AscendC::AIC) {
        const uint32_t aic_idx = AscendC::GetBlockIdx();
        if (aic_idx >= num_aic_cores) {
            return;
        }
        rt.init();
        FullPathAIC aic;
        const uint32_t q_stride = num_q_heads * DIM;
        const uint32_t kv_stride = num_kv_heads * DIM;
        // These are loop-carried scheduling counters, not async completion state.
        // Buffer lifetime is still expressed only by producer/consumer events.
        uint32_t global_l0_idx = 0U;
        uint32_t stream_global_l1_idx = 0U;
        AicQTaskScalars pending_q_tasks[2] = {};
        AicL0TileScalars pending_l0_tiles[2] = {};
        TaskCursor<TASK_METADATA_BATCHES> cursor(
            task_metadata, batch, aic_idx, num_aic_cores, num_q_heads,
            num_kv_heads, window_left, window_right, global_window_size,
            gqa_layout != 0U);
        TaskInfo task;
        while (cursor.next(task)) {
            const uint32_t q_offset = task.q_global_begin * q_stride + task.q_head * DIM;
            const uint32_t kv_offset = task.kv_global_begin * kv_stride + task.kv_head * DIM;
            const uint32_t task_global_l1_base_idx = stream_global_l1_idx;
            const AicQTaskScalars q_task =
                make_aic_q_task_scalars(task, q_offset, kv_offset, global_window_size);
            if (!has_l0_gap(task)) {
                const AicL0TileScalars first_tile =
                    make_aic_l0_tile_scalars_contiguous(task, 0U, global_l0_idx, task_global_l1_base_idx);
                aic.template init<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                    rt, q_gm, k_gm, v_gm, q_stride, kv_stride, q_task, first_tile);
                for (uint32_t ordinal = 0U; ordinal < task.valid_l0_count; ++ordinal) {
                    const uint32_t next_ordinal = ordinal + 1U;
                    const AicL0TileScalars cur_l0_tile =
                        make_aic_l0_tile_scalars_contiguous(task, ordinal, global_l0_idx, task_global_l1_base_idx);
                    const AicL0TileScalars next_l0_tile = make_aic_l0_tile_scalars_contiguous(
                        task, next_ordinal, global_l0_idx + 1U, task_global_l1_base_idx);
                    const bool has_next_l1 = next_ordinal < task.valid_l0_count &&
                        next_l0_tile.local_l1_idx != cur_l0_tile.local_l1_idx;
                    aic.template run_cube1<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                        rt, k_gm, kv_stride, q_task, cur_l0_tile, has_next_l1, next_l0_tile);
                    if (global_l0_idx >= 2U) {
                        const uint32_t consume_slot = global_l0_idx % 2U;
                        const uint32_t future_slot = (global_l0_idx - 1U) % 2U;
                        const bool preload_next_v_l1 =
                            pending_q_tasks[consume_slot].linear_q_tile ==
                                pending_q_tasks[future_slot].linear_q_tile &&
                            pending_l0_tiles[future_slot].global_l1_idx !=
                                pending_l0_tiles[consume_slot].global_l1_idx;
                        aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                            rt, v_gm, kv_stride,
                            pending_q_tasks[consume_slot], pending_l0_tiles[consume_slot],
                            preload_next_v_l1,
                            pending_q_tasks[future_slot], pending_l0_tiles[future_slot]);
                    }
                    const uint32_t pending_slot = global_l0_idx % 2U;
                    pending_q_tasks[pending_slot] = q_task;
                    pending_l0_tiles[pending_slot] = cur_l0_tile;
                    ++global_l0_idx;
                }
            } else {
                const AicL0TileScalars first_tile =
                    make_aic_l0_tile_scalars_sparse(task, 0U, global_l0_idx, task_global_l1_base_idx);
                aic.template init<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                    rt, q_gm, k_gm, v_gm, q_stride, kv_stride, q_task, first_tile);
                for (uint32_t ordinal = 0U; ordinal < task.valid_l0_count; ++ordinal) {
                    const uint32_t next_ordinal = ordinal + 1U;
                    const AicL0TileScalars cur_l0_tile =
                        make_aic_l0_tile_scalars_sparse(task, ordinal, global_l0_idx, task_global_l1_base_idx);
                    const AicL0TileScalars next_l0_tile = make_aic_l0_tile_scalars_sparse(
                        task, next_ordinal, global_l0_idx + 1U, task_global_l1_base_idx);
                    const bool has_next_l1 = next_ordinal < task.valid_l0_count &&
                        next_l0_tile.local_l1_idx != cur_l0_tile.local_l1_idx;
                    aic.template run_cube1<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                        rt, k_gm, kv_stride, q_task, cur_l0_tile, has_next_l1, next_l0_tile);
                    if (global_l0_idx >= 2U) {
                        const uint32_t consume_slot = global_l0_idx % 2U;
                        const uint32_t future_slot = (global_l0_idx - 1U) % 2U;
                        const bool preload_next_v_l1 =
                            pending_q_tasks[consume_slot].linear_q_tile ==
                                pending_q_tasks[future_slot].linear_q_tile &&
                            pending_l0_tiles[future_slot].global_l1_idx !=
                                pending_l0_tiles[consume_slot].global_l1_idx;
                        aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                            rt, v_gm, kv_stride,
                            pending_q_tasks[consume_slot], pending_l0_tiles[consume_slot],
                            preload_next_v_l1,
                            pending_q_tasks[future_slot], pending_l0_tiles[future_slot]);
                    }
                    const uint32_t pending_slot = global_l0_idx % 2U;
                    pending_q_tasks[pending_slot] = q_task;
                    pending_l0_tiles[pending_slot] = cur_l0_tile;
                    ++global_l0_idx;
                }
            }
            stream_global_l1_idx = task_global_l1_base_idx + l1_group_count(task);
        }
        if (global_l0_idx >= 2U) {
            const uint32_t first_slot = (global_l0_idx - 2U) % 2U;
            const uint32_t second_slot = (global_l0_idx - 1U) % 2U;
            const bool preload_next_v_l1 =
                pending_q_tasks[first_slot].linear_q_tile ==
                    pending_q_tasks[second_slot].linear_q_tile &&
                pending_l0_tiles[second_slot].global_l1_idx !=
                    pending_l0_tiles[first_slot].global_l1_idx;
            aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                rt, v_gm, kv_stride,
                pending_q_tasks[first_slot], pending_l0_tiles[first_slot],
                preload_next_v_l1,
                pending_q_tasks[second_slot], pending_l0_tiles[second_slot]);
            const AicQTaskScalars empty_future_q_task = {};
            const AicL0TileScalars empty_future_l0_tile = {};
            aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                rt, v_gm, kv_stride,
                pending_q_tasks[second_slot], pending_l0_tiles[second_slot], false,
                empty_future_q_task, empty_future_l0_tile);
        } else if (global_l0_idx == 1U) {
            const AicQTaskScalars empty_future_q_task = {};
            const AicL0TileScalars empty_future_l0_tile = {};
            aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                rt, v_gm, kv_stride,
                pending_q_tasks[0], pending_l0_tiles[0], false,
                empty_future_q_task, empty_future_l0_tile);
        }
    } else if constexpr (g_coreType == AscendC::AIV) {
        const uint32_t aiv_idx = AscendC::GetBlockIdx();
        const uint32_t aic_idx = FullPathAIV::aic_index(aiv_idx, num_aic_cores);
        if (aic_idx >= num_aic_cores) {
            return;
        }
        rt.init();
        FullPathAIV aiv;
        aiv.init_backward_tokens(rt);
        // Loop-carried scheduling counters only; they do not describe event completion.
        uint32_t issue_idx = 0U;
        uint32_t q_owner_idx = 0U;
        Vec2TaskInfo pending_issue_tasks[VEC2_QUEUE_DEPTH];
        uint32_t pending_issue_ordinals[VEC2_QUEUE_DEPTH] = {};
        uint32_t pending_issue_indices[VEC2_QUEUE_DEPTH] = {};
        uint32_t pending_issue_q_owners[VEC2_QUEUE_DEPTH] = {};
        TaskCursor<TASK_METADATA_BATCHES> cursor(
            task_metadata, batch, aic_idx, num_aic_cores, num_q_heads,
            num_kv_heads, window_left, window_right, global_window_size,
            gqa_layout != 0U);
        TaskInfo task;
        while (cursor.next(task)) {
            const uint32_t task_q_owner = q_owner_idx++;
            const uint32_t q_abs_tile_base = task.kv_computed_len + task.q_tile * TILE_Q;
            if (!has_l0_gap(task)) {
                for (uint32_t ordinal = 0U; ordinal < task.valid_l0_count; ++ordinal) {
                    const uint32_t k_tile = task.start_l0_tile + ordinal;
                    aiv.template run_vec1<decltype(rt), TILE_Q, TILE_L0, DIM, (SWA_V5_ENTRY_IS_CAUSAL != 0)>(
                        rt, atten_mask_gm,
                        ordinal, k_tile, q_abs_tile_base, task.q_rows, task.kv_len,
                        window_left, window_right, global_window_size, scale, issue_idx,
                        task_q_owner);
                    if (issue_idx >= VEC2_QUEUE_DEPTH) {
                        const uint32_t consume_slot = issue_idx % VEC2_QUEUE_DEPTH;
                        aiv.template run_vec2<decltype(rt), TILE_Q, TILE_L0, DIM,
                            (SWA_V5_ENTRY_IS_FORWARD != 0), (SWA_V5_ENTRY_OUTPUT_FP16 != 0), (SWA_V5_ENTRY_OUTPUT_BF16 != 0)>(rt,
                            out_gm, output_cache_gm, softmax_lse_gm, WRITE_OUTPUT_CACHE,
                            pending_issue_tasks[consume_slot].q_global_begin,
                            pending_issue_tasks[consume_slot].q_head,
                            pending_issue_tasks[consume_slot].q_rows, total_q_tokens,
                            pending_issue_ordinals[consume_slot],
                            pending_issue_tasks[consume_slot].valid_l0_count,
                            num_q_heads * DIM, pending_issue_indices[consume_slot],
                            pending_issue_q_owners[consume_slot]);
                    }
                    const uint32_t pending_slot = issue_idx % VEC2_QUEUE_DEPTH;
                    pending_issue_tasks[pending_slot] = {
                        task.q_global_begin, task.q_head, task.q_rows, task.valid_l0_count};
                    pending_issue_ordinals[pending_slot] = ordinal;
                    pending_issue_indices[pending_slot] = issue_idx;
                    pending_issue_q_owners[pending_slot] = task_q_owner;
                    ++issue_idx;
                }
            } else {
                for (uint32_t ordinal = 0U; ordinal < task.valid_l0_count; ++ordinal) {
                    const uint32_t k_tile = l0_tile_at(task, ordinal);
                    aiv.template run_vec1<decltype(rt), TILE_Q, TILE_L0, DIM, (SWA_V5_ENTRY_IS_CAUSAL != 0)>(
                        rt, atten_mask_gm,
                        ordinal, k_tile, q_abs_tile_base, task.q_rows, task.kv_len,
                        window_left, window_right, global_window_size, scale, issue_idx,
                        task_q_owner);
                    if (issue_idx >= VEC2_QUEUE_DEPTH) {
                        const uint32_t consume_slot = issue_idx % VEC2_QUEUE_DEPTH;
                        aiv.template run_vec2<decltype(rt), TILE_Q, TILE_L0, DIM,
                            (SWA_V5_ENTRY_IS_FORWARD != 0), (SWA_V5_ENTRY_OUTPUT_FP16 != 0), (SWA_V5_ENTRY_OUTPUT_BF16 != 0)>(rt,
                            out_gm, output_cache_gm, softmax_lse_gm, WRITE_OUTPUT_CACHE,
                            pending_issue_tasks[consume_slot].q_global_begin,
                            pending_issue_tasks[consume_slot].q_head,
                            pending_issue_tasks[consume_slot].q_rows, total_q_tokens,
                            pending_issue_ordinals[consume_slot],
                            pending_issue_tasks[consume_slot].valid_l0_count,
                            num_q_heads * DIM, pending_issue_indices[consume_slot],
                            pending_issue_q_owners[consume_slot]);
                    }
                    const uint32_t pending_slot = issue_idx % VEC2_QUEUE_DEPTH;
                    pending_issue_tasks[pending_slot] = {
                        task.q_global_begin, task.q_head, task.q_rows, task.valid_l0_count};
                    pending_issue_ordinals[pending_slot] = ordinal;
                    pending_issue_indices[pending_slot] = issue_idx;
                    pending_issue_q_owners[pending_slot] = task_q_owner;
                    ++issue_idx;
                }
            }
        }
        const uint32_t pending_count = issue_idx < VEC2_QUEUE_DEPTH ? issue_idx : VEC2_QUEUE_DEPTH;
        for (uint32_t pending = issue_idx - pending_count; pending < issue_idx; ++pending) {
            const uint32_t consume_slot = pending % VEC2_QUEUE_DEPTH;
            aiv.template run_vec2<decltype(rt), TILE_Q, TILE_L0, DIM,
                                (SWA_V5_ENTRY_IS_FORWARD != 0), (SWA_V5_ENTRY_OUTPUT_FP16 != 0), (SWA_V5_ENTRY_OUTPUT_BF16 != 0)>(rt,
                    out_gm, output_cache_gm, softmax_lse_gm, WRITE_OUTPUT_CACHE,
                    pending_issue_tasks[consume_slot].q_global_begin,
                    pending_issue_tasks[consume_slot].q_head,
                    pending_issue_tasks[consume_slot].q_rows, total_q_tokens,
                    pending_issue_ordinals[consume_slot],
                    pending_issue_tasks[consume_slot].valid_l0_count,
                    num_q_heads * DIM, pending_issue_indices[consume_slot],
                    pending_issue_q_owners[consume_slot]);
        }
        aiv.finish_vec2_writes(rt);
    }
}
