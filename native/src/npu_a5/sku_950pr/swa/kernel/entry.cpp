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

#ifndef SWA_V5_ENTRY_NUM_AIC_CORES
#define SWA_V5_ENTRY_NUM_AIC_CORES 28
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
static constexpr uint32_t DIM = SWA_V5_ENTRY_DIM;
static_assert(DIM == 128, "v5 full_path currently fixes head_dim=128");
static_assert(TILE_L1 % TILE_L0 == 0 && TILE_L1 >= TILE_L0,
    "v5 full_path expects TILE_L1 to be a multiple of TILE_L0");
static_assert(CUBE2_BMM2_CONSUMED_BASE + 1U <= 10U, "cube2 event ids must stay in 0-10 and 16-26");

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
    uint32_t valid_l0_count;
};

__aicore__ inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

template<bool IS_ABAB>
__aicore__ inline uint32_t map_kv_head(uint32_t q_head, uint32_t num_kv_heads, uint32_t share_q_heads) {
    if constexpr (IS_ABAB) {
        return q_head % num_kv_heads;
    }
    return q_head / share_q_heads;
}

__aicore__ inline uint32_t l0_tile_at(const TaskInfo& task, uint32_t ordinal) {
    return MaskHelper::l0_tile_at(task.start_l0_tile, task.second_l0_tile, ordinal);
}

__aicore__ inline bool has_l0_gap(const TaskInfo& task) {
    return !MaskHelper::is_l0_range_contiguous(task.start_l0_tile, task.second_l0_tile, task.valid_l0_count);
}

__aicore__ inline uint32_t local_l1_idx(const TaskInfo& task, uint32_t ordinal) {
    if (has_l0_gap(task)) {
        return ordinal == 0U ? 0U : 1U + (ordinal - 1U) / (TILE_L1 / TILE_L0);
    }
    return ordinal / (TILE_L1 / TILE_L0);
}

__aicore__ inline uint32_t part_idx(const TaskInfo& task, uint32_t local_l0_idx) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (has_l0_gap(task)) {
        return local_l0_idx == 0U ? 0U : (local_l0_idx - 1U) % L1_PARTS;
    }
    return local_l0_idx % L1_PARTS;
}

__aicore__ inline uint32_t l1_group_start_local_l0_idx(const TaskInfo& task, uint32_t local_l0_idx) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (has_l0_gap(task)) {
        return local_l0_idx == 0U ? 0U : 1U + ((local_l0_idx - 1U) / L1_PARTS) * L1_PARTS;
    }
    return (local_l0_idx / L1_PARTS) * L1_PARTS;
}

__aicore__ inline uint32_t num_parts(const TaskInfo& task, uint32_t local_l0_idx) {
    constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
    if (has_l0_gap(task) && local_l0_idx == 0U) {
        return 1U;
    }
    const uint32_t group_start = l1_group_start_local_l0_idx(task, local_l0_idx);
    const uint32_t remaining_l0_parts = task.valid_l0_count - group_start;
    return remaining_l0_parts < L1_PARTS ? remaining_l0_parts : L1_PARTS;
}

__aicore__ inline uint32_t l1_group_count(const TaskInfo& task) {
    if (task.valid_l0_count == 0U) {
        return 0U;
    }
    if (has_l0_gap(task)) {
        return 1U + ceil_div(task.valid_l0_count - 1U, TILE_L1 / TILE_L0);
    }
    return ceil_div(task.valid_l0_count, TILE_L1 / TILE_L0);
}

__aicore__ inline AicQTaskScalars make_aic_q_task_scalars(
    const TaskInfo& task, uint32_t q_offset, uint32_t kv_offset) {
    return {
        task.linear_q_tile,
        q_offset,
        kv_offset,
        task.q_rows,
        task.kv_len,
        task.start_l0_tile,
        task.second_l0_tile,
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

__aicore__ inline bool fill_task(uint32_t linear_q_tile, AscendC::GlobalTensor<uint32_t>& cu_q_lens_gm,
    AscendC::GlobalTensor<uint32_t>& cu_kv_lens_gm, uint32_t batch, uint32_t num_q_heads,
    uint32_t num_kv_heads, TaskInfo& task) {
    if (num_kv_heads == 0U || num_q_heads % num_kv_heads != 0U) {
        return false;
    }
    uint32_t remaining = linear_q_tile;
    const uint32_t share_q_heads = num_q_heads / num_kv_heads;
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t q_begin = cu_q_lens_gm.GetValue(b);
        const uint32_t q_end = cu_q_lens_gm.GetValue(b + 1U);
        const uint32_t q_len = q_end - q_begin;
        const uint32_t q_tiles = ceil_div(q_len, TILE_Q);
        const uint32_t batch_q_tile_tasks = q_tiles * num_q_heads;
        if (remaining >= batch_q_tile_tasks) {
            remaining -= batch_q_tile_tasks;
            continue;
        }
        if (q_tiles == 0U) {
            return false;
        }
        const uint32_t q_tile = remaining / num_q_heads;
        const uint32_t q_head = remaining - q_tile * num_q_heads;
        const uint32_t kv_begin = cu_kv_lens_gm.GetValue(b);
        const uint32_t kv_end = cu_kv_lens_gm.GetValue(b + 1U);
        const uint32_t kv_len = kv_end - kv_begin;
        if (kv_len < q_len) {
            return false;
        }
        const uint32_t q_tile_begin = q_tile * TILE_Q;
        const uint32_t q_rows = min_u32(TILE_Q, q_len - q_tile_begin);
        if (q_rows == 0U) {
            return false;
        }
        const uint32_t kv_computed_len = kv_len - q_len;
        const uint32_t q_abs_start = kv_computed_len + q_tile_begin;
        const L0TileRange l0_range =
            MaskHelper::make_l0_tile_range<TILE_L0, SWA_V5_ENTRY_WINDOW_LEFT, SWA_V5_ENTRY_WINDOW_RIGHT,
                SWA_V5_ENTRY_GLOBAL_WINDOW_SIZE, (SWA_V5_ENTRY_IS_CAUSAL != 0)>(q_abs_start, q_rows, kv_len);
        if (l0_range.valid_l0_count == 0U) {
            return false;
        }
        task.linear_q_tile = linear_q_tile;
        task.batch_idx = b;
        task.q_head = q_head;
        task.kv_head = map_kv_head<(SWA_V5_ENTRY_GQA_LAYOUT_ABAB != 0)>(q_head, num_kv_heads, share_q_heads);
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
        task.valid_l0_count = l0_range.valid_l0_count;
        return true;
    }
    return false;
}

__aicore__ inline uint32_t count_linear_q_tiles(AscendC::GlobalTensor<uint32_t>& cu_q_lens_gm,
    uint32_t batch, uint32_t num_q_heads) {
    uint32_t total = 0U;
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t q_len = cu_q_lens_gm.GetValue(b + 1U) - cu_q_lens_gm.GetValue(b);
        total += ceil_div(q_len, TILE_Q) * num_q_heads;
    }
    return total;
}

}  // namespace SWA_v5::Pipeline::FullPath

extern "C" __global__ __aicore__ void SWA_V5_ENTRY_KERNEL(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR softmax_lse, GM_ADDR cu_q_lens, GM_ADDR cu_kv_lens,
    uint32_t batch, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t total_q_tokens,
    uint32_t total_kv_tokens, uint32_t max_q_len, uint32_t max_kv_len, uint32_t window_left,
    uint32_t window_right, uint32_t dim) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace SWA_v5::Pipeline::FullPath;

    if (num_q_heads != SWA_V5_ENTRY_NUM_Q_HEADS || num_kv_heads != SWA_V5_ENTRY_NUM_KV_HEADS ||
        dim != DIM || max_q_len != SWA_V5_ENTRY_MAX_Q_LEN || max_kv_len != SWA_V5_ENTRY_MAX_KV_LEN ||
        window_left != SWA_V5_ENTRY_WINDOW_LEFT || window_right != SWA_V5_ENTRY_WINDOW_RIGHT) {
        return;
    }
    if (total_q_tokens > batch * max_q_len || total_kv_tokens > batch * max_kv_len) {
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
    GMTensor_ND<InputT> q_gm;
    GMTensor_ND<InputT> k_gm;
    GMTensor_ND<InputT> v_gm;
    GMTensor_ND<OutputT> out_gm;
    GMTensor_ND<float> softmax_lse_gm;
    GMTensor_ND<uint8_t> atten_mask_gm;
    AscendC::GlobalTensor<uint32_t> cu_q_lens_gm;
    AscendC::GlobalTensor<uint32_t> cu_kv_lens_gm;
    q_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(q));
    k_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(k));
    v_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(v));
    out_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(out));
    softmax_lse_gm.tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmax_lse));
    cu_q_lens_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cu_q_lens));
    cu_kv_lens_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cu_kv_lens));

    const uint32_t total_q_tile_tasks = count_linear_q_tiles(cu_q_lens_gm, batch, num_q_heads);

    if constexpr (g_coreType == AscendC::AIC) {
        const uint32_t aic_idx = AscendC::GetBlockIdx();
        if (aic_idx >= SWA_V5_ENTRY_NUM_AIC_CORES) {
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
        AicQTaskScalars prior_q_task = {};
        AicL0TileScalars prior_l0_tile = {};
        for (uint32_t linear = aic_idx; linear < total_q_tile_tasks; linear += SWA_V5_ENTRY_NUM_AIC_CORES) {
            TaskInfo task;
            if (!fill_task(linear, cu_q_lens_gm, cu_kv_lens_gm, batch, num_q_heads, num_kv_heads, task)) {
                continue;
            }
            const uint32_t q_offset = task.q_global_begin * q_stride + task.q_head * DIM;
            const uint32_t kv_offset = task.kv_global_begin * kv_stride + task.kv_head * DIM;
            const uint32_t task_global_l1_base_idx = stream_global_l1_idx;
            const AicQTaskScalars q_task = make_aic_q_task_scalars(task, q_offset, kv_offset);
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
                    if (global_l0_idx != 0U) {
                        const bool preload_next_v_l1 =
                            prior_q_task.linear_q_tile == q_task.linear_q_tile &&
                            cur_l0_tile.global_l1_idx != prior_l0_tile.global_l1_idx;
                        aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                            rt, v_gm, kv_stride, prior_q_task, prior_l0_tile, preload_next_v_l1,
                            q_task, cur_l0_tile);
                    }
                    prior_q_task = q_task;
                    prior_l0_tile = cur_l0_tile;
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
                    if (global_l0_idx != 0U) {
                        const bool preload_next_v_l1 =
                            prior_q_task.linear_q_tile == q_task.linear_q_tile &&
                            cur_l0_tile.global_l1_idx != prior_l0_tile.global_l1_idx;
                        aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                            rt, v_gm, kv_stride, prior_q_task, prior_l0_tile, preload_next_v_l1,
                            q_task, cur_l0_tile);
                    }
                    prior_q_task = q_task;
                    prior_l0_tile = cur_l0_tile;
                    ++global_l0_idx;
                }
            }
            stream_global_l1_idx = task_global_l1_base_idx + l1_group_count(task);
        }
        if (global_l0_idx != 0U) {
            const AicQTaskScalars empty_future_q_task = {};
            const AicL0TileScalars empty_future_l0_tile = {};
            aic.template run_cube2<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(
                rt, v_gm, kv_stride, prior_q_task, prior_l0_tile, false,
                empty_future_q_task, empty_future_l0_tile);
        }
    } else if constexpr (g_coreType == AscendC::AIV) {
        const uint32_t aiv_idx = AscendC::GetBlockIdx();
        const uint32_t aic_idx = FullPathAIV::aic_index(aiv_idx, SWA_V5_ENTRY_NUM_AIC_CORES);
        if (aic_idx >= SWA_V5_ENTRY_NUM_AIC_CORES) {
            return;
        }
        rt.init();
        FullPathAIV aiv;
        aiv.init_backward_tokens(rt);
        // Loop-carried scheduling counters only; they do not describe event completion.
        uint32_t issue_idx = 0U;
        TaskInfo prior_issue_task;
        uint32_t prior_issue_ordinal = 0U;
        uint32_t prior_issue_idx = 0U;
        for (uint32_t linear = aic_idx; linear < total_q_tile_tasks; linear += SWA_V5_ENTRY_NUM_AIC_CORES) {
            TaskInfo task;
            if (!fill_task(linear, cu_q_lens_gm, cu_kv_lens_gm, batch, num_q_heads, num_kv_heads, task)) {
                continue;
            }
            const uint32_t q_abs_tile_base = task.kv_computed_len + task.q_tile * TILE_Q;
            if (!has_l0_gap(task)) {
                for (uint32_t ordinal = 0U; ordinal < task.valid_l0_count; ++ordinal) {
                    const uint32_t k_tile = task.start_l0_tile + ordinal;
                    aiv.template run_vec1<decltype(rt), TILE_Q, TILE_L0, DIM, (SWA_V5_ENTRY_IS_CAUSAL != 0),
                        SWA_V5_ENTRY_WINDOW_LEFT, SWA_V5_ENTRY_WINDOW_RIGHT, SWA_V5_ENTRY_GLOBAL_WINDOW_SIZE>(
                        rt, atten_mask_gm,
                        ordinal, k_tile, q_abs_tile_base, task.q_rows, task.kv_len, issue_idx,
                        task.linear_q_tile);
                    if (issue_idx != 0U) {
                        aiv.template run_vec2<decltype(rt), TILE_Q, TILE_L0, DIM,
                            (SWA_V5_ENTRY_IS_FORWARD != 0), (SWA_V5_ENTRY_OUTPUT_FP16 != 0), (SWA_V5_ENTRY_OUTPUT_BF16 != 0)>(rt,
                            out_gm, softmax_lse_gm, prior_issue_task.q_global_begin, prior_issue_task.q_head, prior_issue_task.q_rows, total_q_tokens,
                            prior_issue_ordinal, prior_issue_task.valid_l0_count, num_q_heads * DIM, prior_issue_idx,
                            prior_issue_task.linear_q_tile);
                    }
                    prior_issue_task = task;
                    prior_issue_ordinal = ordinal;
                    prior_issue_idx = issue_idx;
                    ++issue_idx;
                }
            } else {
                for (uint32_t ordinal = 0U; ordinal < task.valid_l0_count; ++ordinal) {
                    const uint32_t k_tile = l0_tile_at(task, ordinal);
                    aiv.template run_vec1<decltype(rt), TILE_Q, TILE_L0, DIM, (SWA_V5_ENTRY_IS_CAUSAL != 0),
                        SWA_V5_ENTRY_WINDOW_LEFT, SWA_V5_ENTRY_WINDOW_RIGHT, SWA_V5_ENTRY_GLOBAL_WINDOW_SIZE>(
                        rt, atten_mask_gm,
                        ordinal, k_tile, q_abs_tile_base, task.q_rows, task.kv_len, issue_idx,
                        task.linear_q_tile);
                    if (issue_idx != 0U) {
                        aiv.template run_vec2<decltype(rt), TILE_Q, TILE_L0, DIM,
                            (SWA_V5_ENTRY_IS_FORWARD != 0), (SWA_V5_ENTRY_OUTPUT_FP16 != 0), (SWA_V5_ENTRY_OUTPUT_BF16 != 0)>(rt,
                            out_gm, softmax_lse_gm, prior_issue_task.q_global_begin, prior_issue_task.q_head, prior_issue_task.q_rows, total_q_tokens,
                            prior_issue_ordinal, prior_issue_task.valid_l0_count, num_q_heads * DIM, prior_issue_idx,
                            prior_issue_task.linear_q_tile);
                    }
                    prior_issue_task = task;
                    prior_issue_ordinal = ordinal;
                    prior_issue_idx = issue_idx;
                    ++issue_idx;
                }
            }
        }
        if (issue_idx != 0U) {
            aiv.template run_vec2<decltype(rt), TILE_Q, TILE_L0, DIM,
                            (SWA_V5_ENTRY_IS_FORWARD != 0), (SWA_V5_ENTRY_OUTPUT_FP16 != 0), (SWA_V5_ENTRY_OUTPUT_BF16 != 0)>(rt,
                out_gm, softmax_lse_gm, prior_issue_task.q_global_begin, prior_issue_task.q_head, prior_issue_task.q_rows, total_q_tokens,
                prior_issue_ordinal, prior_issue_task.valid_l0_count, num_q_heads * DIM, prior_issue_idx,
                prior_issue_task.linear_q_tile);
        }
    }
}
