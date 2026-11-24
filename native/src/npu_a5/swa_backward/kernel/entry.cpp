#include "kernel_operator.h"
#include "kernel/aic.cpp"
#include "kernel/aiv.cpp"
#include "kernel/finalize.h"
#include "kernel/task_scheduler.h"

#ifndef SWA_BWD_ENTRY_KERNEL
#define SWA_BWD_ENTRY_KERNEL swa_v5_backward_schedule
#endif

#ifndef SWA_BWD_ENTRY_IS_CAUSAL
#define SWA_BWD_ENTRY_IS_CAUSAL 1
#endif

namespace SWA_v5::Backward {

static constexpr uint32_t TILE_Q = 128U;
static constexpr uint32_t TILE_K = 128U;
static constexpr uint32_t DIM = 128U;
static constexpr uint32_t PIPELINE_DISTANCE = 2U;
static constexpr uint32_t MM_SLOTS = 2U;
static constexpr uint32_t P_DS_SLOTS = 2U;
static constexpr uint32_t AIVS_PER_AIC = 2U;
static constexpr uint64_t DKV_COLOR_WORKSPACE_LIMIT_ELEMS =
    96ULL * 1024ULL * 1024ULL / sizeof(float);
static constexpr uint64_t DKV_COLOR_MQA_WORKSPACE_LIMIT_ELEMS =
    88ULL * 1024ULL * 1024ULL / sizeof(float);
static constexpr uint64_t DKV_COLOR_TWO_HEAD_WORKSPACE_LIMIT_ELEMS =
    84ULL * 1024ULL * 1024ULL / sizeof(float);
static constexpr uint64_t DKV_COLOR_LARGE_GQA_WORKSPACE_LIMIT_ELEMS =
    80ULL * 1024ULL * 1024ULL / sizeof(float);
static constexpr uint64_t DKV_COLOR_THREE_HEAD_PACK_LIMIT_ELEMS =
    64ULL * 1024ULL * 1024ULL / sizeof(float);
static constexpr uint64_t DKV_COLOR_SINGLE_HEAD_WORKSPACE_LIMIT_ELEMS =
    512ULL * 1024ULL * 1024ULL / sizeof(float);

__aicore__ inline AicStageTask make_aic_stage(
    const TaskInfo& task, uint32_t issue_idx, uint32_t group_tile_count,
    uint32_t q_l1_slot, bool direct_atomic = false) {
    return {
        task,
        issue_idx,
        issue_idx % MM_SLOTS,
        issue_idx % P_DS_SLOTS,
        q_l1_slot,
        direct_atomic || task.kv_tile_ordinal == 0U,
        direct_atomic || task.kv_tile_ordinal + 1U == group_tile_count,
        direct_atomic,
        direct_atomic,
        false,
        false,
        false,
        false,
        false,
        false,
        0U,
        0U,
        0U,
    };
}

__aicore__ inline void configure_dkv_color_stage(
    AicStageTask& stage, uint32_t head_in_group, int32_t delta,
    int32_t color_min_delta, uint32_t color_count,
    uint64_t color_dk_storage_elems,
    bool batch_local, uint32_t batch_kv_begin,
    uint64_t batch_padded_begin, uint32_t kv_stride,
    bool chunk_local, uint32_t chunk_kv_begin,
    bool use_bf16_partials, bool use_fp16_partials) {
    stage.dkv_color_partial = true;
    stage.dkv_color_atomic = head_in_group != 0U;
    stage.dkv_color_bf16 = use_bf16_partials;
    stage.dkv_color_fp16 = use_fp16_partials;
    const uint64_t color =
        static_cast<uint32_t>(delta - color_min_delta);
    stage.dkv_color_offset =
        color * (batch_local &&
                !use_bf16_partials && !use_fp16_partials ?
            color_dk_storage_elems :
            2U * color_dk_storage_elems);
    uint64_t color_row =
        static_cast<uint64_t>(stage.task.kv_global_begin) +
        static_cast<uint64_t>(stage.task.batch_idx) *
            (TILE_K - 1U);
    if (batch_local) {
        if (chunk_local) {
            color_row = static_cast<uint64_t>(
                stage.task.kv_global_begin - batch_kv_begin -
                chunk_kv_begin);
        } else {
            color_row =
                static_cast<uint64_t>(stage.task.kv_global_begin) +
                static_cast<uint64_t>(stage.task.batch_idx) *
                    (TILE_K - 1U) -
                batch_padded_begin;
        }
    }
    stage.dkv_color_dst_offset =
        color_row * kv_stride +
        static_cast<uint64_t>(stage.task.kv_head) * DIM;
    stage.dkv_color_storage_elems =
        batch_local && !use_bf16_partials && !use_fp16_partials ?
            static_cast<uint64_t>(color_count) *
                color_dk_storage_elems :
            color_dk_storage_elems;
}

template<class CuLens>
__aicore__ inline void configure_global_dkv_stage(
    AicStageTask& stage, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t core_idx, uint64_t dkv_color_elems,
    uint64_t global_dkv_plane_elems, uint32_t kv_stride,
    uint32_t window_left, uint32_t global_window_size) {
    stage.dkv_color_partial = true;
    stage.dkv_color_atomic = true;
    stage.dkv_color_offset =
        dkv_color_elems + 2U * core_idx * global_dkv_plane_elems;
    const uint64_t partial_row_base =
        global_partial_row_base<TILE_K>(
            cu_q_lens, cu_kv_lens, stage.task.batch_idx,
            window_left, global_window_size);
    stage.dkv_color_dst_offset =
        (partial_row_base +
            static_cast<uint64_t>(stage.task.kv_tile) * TILE_K) *
            kv_stride +
        static_cast<uint64_t>(stage.task.kv_head) * DIM;
    stage.dkv_color_storage_elems = global_dkv_plane_elems;
}

__aicore__ inline bool task_in_dkv_color_chunk(
    const TaskInfo& task, uint32_t batch_kv_begin,
    uint32_t chunk_kv_begin, uint32_t chunk_kv_rows) {
    const uint32_t task_kv_begin =
        task.kv_global_begin - batch_kv_begin;
    return task_kv_begin >= chunk_kv_begin &&
        task_kv_begin < chunk_kv_begin + chunk_kv_rows;
}

__aicore__ inline AivStageTask make_aiv_stage(const TaskInfo& task, uint32_t issue_idx) {
    return {task, issue_idx, issue_idx % MM_SLOTS, issue_idx % P_DS_SLOTS};
}

__aicore__ inline bool same_q_task(
    const TaskInfo& lhs, const TaskInfo& rhs) {
    return lhs.q_global_begin == rhs.q_global_begin &&
        lhs.q_head == rhs.q_head && lhs.q_rows == rhs.q_rows;
}

template<class CuLens>
__aicore__ inline bool first_packed_wave_task_for_core(
    uint32_t core_idx, uint32_t wave,
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t active_aic_cores,
    uint32_t packed_color_storage_rows, uint32_t window_left,
    uint32_t window_right, uint32_t gqa_layout,
    uint32_t uniform_q_len, uint32_t uniform_kv_len,
    int32_t min_delta, int32_t max_delta, bool use_bf16_partials,
    TaskInfo& task) {
    uint32_t batch_begin = 0U;
    uint32_t batch_end = batch;
    if (!decode_packed_batch_color_wave<TILE_K>(
            wave, cu_q_lens, cu_kv_lens, batch,
            packed_color_storage_rows, batch_begin, batch_end)) {
        return false;
    }
    uint32_t candidate_begin = 0U;
    uint32_t candidate_count = 0U;
    local_round_candidate_range_span<TILE_Q>(
        cu_q_lens, cu_kv_lens, batch_begin, batch_end,
        num_kv_heads, candidate_begin, candidate_count);
    constexpr uint32_t candidate_local_begin = 0U;
    const bool fold_candidate_boundary =
        candidate_count > active_aic_cores;
    const uint32_t gqa_ratio = num_q_heads / num_kv_heads;
    const bool collaborate_tail_candidates =
        !use_bf16_partials && gqa_ratio > 1U &&
        fold_candidate_boundary &&
        use_tail_candidate_collaboration(
            candidate_count, active_aic_cores);
    const int32_t collaborative_split_delta = min_delta + 2;
    for (uint32_t candidate_ordinal = 0U;
         ; ++candidate_ordinal) {
        uint32_t linear_candidate = 0U;
        bool helper_task = false;
        bool tail_owner_task = false;
        if (!scheduled_local_round_collaborative_candidate(
                core_idx, candidate_ordinal, candidate_count,
                active_aic_cores, fold_candidate_boundary,
                collaborate_tail_candidates, linear_candidate,
                helper_task, tail_owner_task)) {
            return false;
        }
        for (int32_t delta = min_delta; delta <= max_delta; ++delta) {
            if (helper_task && delta < collaborative_split_delta) {
                continue;
            }
            if (fill_local_round_task_span<
                    TILE_Q, TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                    candidate_local_begin + linear_candidate,
                    candidate_begin, delta, 0U,
                    cu_q_lens, cu_kv_lens, batch_begin, batch_end,
                    num_q_heads, num_kv_heads, window_left,
                    window_right, gqa_layout != 0U,
                    uniform_q_len, uniform_kv_len, task)) {
                return true;
            }
        }
    }
}

template<class CuLens>
__aicore__ inline uint32_t total_tasks(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size) {
    return count_tasks<TILE_Q, TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
        cu_q_lens, cu_kv_lens, batch, num_q_heads, num_kv_heads,
        window_left, window_right, global_window_size);
}

template<class CuLens>
__aicore__ inline uint32_t total_q_groups(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads) {
    return count_q_groups<TILE_Q>(
        cu_q_lens, cu_kv_lens, batch, num_q_heads, num_kv_heads);
}

template<class CuLens>
__aicore__ inline bool decode_task(uint32_t linear_task, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t window_left, uint32_t window_right, uint32_t global_window_size,
    bool gqa_layout_abab, TaskInfo& task) {
    return fill_task<TILE_Q, TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
        linear_task, cu_q_lens, cu_kv_lens, batch, num_q_heads, num_kv_heads,
        window_left, window_right, global_window_size, gqa_layout_abab, task);
}

template<class CuLens>
__aicore__ inline bool decode_q_group(uint32_t linear_group,
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size, bool gqa_layout_abab,
    QTaskGroup& group) {
    return fill_q_group<TILE_Q, TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
        linear_group, cu_q_lens, cu_kv_lens, batch, num_q_heads, num_kv_heads,
        window_left, window_right, global_window_size, gqa_layout_abab, group);
}

template<class CuLens>
__aicore__ inline bool decode_global_q_group(
    uint32_t linear_group, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t window_left, uint32_t window_right,
    uint32_t global_window_size, bool gqa_layout_abab,
    QTaskGroup& group) {
    if (!decode_q_group(
            linear_group, cu_q_lens, cu_kv_lens, batch,
            num_q_heads, num_kv_heads, window_left, window_right,
            global_window_size, gqa_layout_abab, group)) {
        return false;
    }
    const uint32_t global_tile_count =
        group.kv_range.tile_count - group.kv_range.local_tile_count;
    group.kv_range = {0U, 0U, 0U, global_tile_count};
    return global_tile_count > 0U;
}

__aicore__ inline bool scheduled_q_group(
    uint32_t core_idx, uint32_t ordinal, uint32_t q_group_count,
    uint32_t active_aic_cores, uint32_t& linear_group) {
    linear_group = core_idx + ordinal * active_aic_cores;
    return linear_group < q_group_count;
}

__aicore__ inline bool scheduled_global_q_group(
    uint32_t core_idx, uint32_t ordinal, uint32_t q_group_count,
    uint32_t parallel_core_count, uint32_t& linear_group) {
    if (parallel_core_count > 0U) {
        if (core_idx >= parallel_core_count) {
            return false;
        }
        return scheduled_q_group(
            core_idx, ordinal, q_group_count,
            parallel_core_count, linear_group);
    }
    if (core_idx != 0U) {
        return false;
    }
    linear_group = ordinal;
    return linear_group < q_group_count;
}

__aicore__ inline bool scheduled_contiguous_group(
    uint32_t core_idx, uint32_t ordinal, uint32_t group_count,
    uint32_t core_count, uint32_t& linear_group) {
    if (core_count == 0U || core_idx >= core_count) {
        return false;
    }
    const uint32_t chunk_size = ceil_div(group_count, core_count);
    if (chunk_size == 0U || ordinal >= chunk_size) {
        return false;
    }
    linear_group = core_idx * chunk_size + ordinal;
    return linear_group < group_count;
}

}  // namespace SWA_v5::Backward

extern "C" __global__ __aicore__ void SWA_BWD_ENTRY_KERNEL(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR dout, GM_ADDR softmax_lse,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, GM_ADDR cu_q_lens, GM_ADDR cu_kv_lens,
    uint32_t batch, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t total_q_tokens,
    uint32_t total_kv_tokens, uint32_t max_q_len, uint32_t max_kv_len, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size, uint32_t gqa_layout,
    uint32_t active_aic_cores, uint32_t dim, float scale,
    uint32_t output_is_f32, uint32_t color_chunk_rows) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace SWA_v5::Backward;

    if (num_q_heads == 0U || num_kv_heads == 0U || num_q_heads % num_kv_heads != 0U ||
        (window_left == 0U && global_window_size == 0U) ||
        gqa_layout > 1U || active_aic_cores == 0U ||
        dim != DIM) {
        return;
    }
    const uint32_t active_aiv_cores =
        active_aic_cores * AIVS_PER_AIC;
    const uint64_t dkv_scratch_elems =
        bwd_dkv_scratch_elems(active_aic_cores);
    const bool uniform_local_batches =
        batch > 0U && max_q_len > 0U && max_kv_len >= max_q_len &&
        static_cast<uint64_t>(total_q_tokens) ==
            static_cast<uint64_t>(batch) * max_q_len &&
        static_cast<uint64_t>(total_kv_tokens) ==
            static_cast<uint64_t>(batch) * max_kv_len;
    const uint32_t uniform_q_len =
        uniform_local_batches ? max_q_len : 0U;
    const uint32_t uniform_kv_len =
        uniform_local_batches ? max_kv_len : 0U;
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR user_workspace = AscendC::GetUserWorkspace(workspace);

    AscendC::GlobalTensor<uint32_t> cu_q_lens_gm;
    AscendC::GlobalTensor<uint32_t> cu_kv_lens_gm;
    cu_q_lens_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cu_q_lens));
    cu_kv_lens_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cu_kv_lens));
    const uint32_t q_group_count =
        total_q_groups(cu_q_lens_gm, cu_kv_lens_gm, batch, num_q_heads, num_kv_heads);
    const uint32_t q_stride = num_q_heads * DIM;
    const uint32_t kv_stride = num_kv_heads * DIM;
    const uint64_t dq_storage_elems =
        (static_cast<uint64_t>(total_q_tokens) +
            static_cast<uint64_t>(batch) * (TILE_Q - 1U)) * q_stride;
    const uint64_t dk_storage_elems =
        (static_cast<uint64_t>(total_kv_tokens) +
            static_cast<uint64_t>(batch) * (TILE_K - 1U)) * kv_stride;
    const uint64_t dv_storage_elems =
        (static_cast<uint64_t>(total_kv_tokens) +
            static_cast<uint64_t>(batch) * (TILE_K - 1U)) * kv_stride;
    const uint64_t workspace_elems =
        dq_storage_elems + dk_storage_elems + dv_storage_elems;
    const uint64_t dkv_color_workspace_limit_elems =
        num_q_heads == 1U && num_kv_heads == 1U ?
            DKV_COLOR_SINGLE_HEAD_WORKSPACE_LIMIT_ELEMS :
        num_kv_heads == 1U ?
            DKV_COLOR_MQA_WORKSPACE_LIMIT_ELEMS :
        num_kv_heads == 2U ?
            DKV_COLOR_TWO_HEAD_WORKSPACE_LIMIT_ELEMS :
        num_q_heads > num_kv_heads && num_kv_heads >= 4U &&
            total_q_tokens == total_kv_tokens ?
            DKV_COLOR_LARGE_GQA_WORKSPACE_LIMIT_ELEMS :
            DKV_COLOR_WORKSPACE_LIMIT_ELEMS;
    const int32_t color_min_delta =
        local_round_min_delta<TILE_Q, TILE_K>(
            cu_q_lens_gm, cu_kv_lens_gm, batch);
    const int32_t color_max_delta =
        local_round_max_delta<TILE_K>(window_left);
    const uint64_t color_count =
        static_cast<uint32_t>(
            color_max_delta - color_min_delta + 1);
    const uint64_t batch_dk_storage_elems =
        (static_cast<uint64_t>(max_kv_len) + TILE_K - 1U) *
        kv_stride;
    const uint64_t color_chunk_storage_rows =
        dkv_color_workspace_limit_elems /
        (2U * color_count * kv_stride);
    const uint64_t packed_color_storage_rows =
        num_kv_heads == 3U ?
              DKV_COLOR_THREE_HEAD_PACK_LIMIT_ELEMS /
                  (2U * color_count * kv_stride) :
            color_chunk_storage_rows;
    const uint64_t chunk_dk_storage_elems =
        (static_cast<uint64_t>(color_chunk_rows) + TILE_K - 1U) *
        kv_stride;
    const uint64_t batch_color_dk_storage_elems =
        packed_color_storage_rows * kv_stride;
    const bool use_dkv_color_partials =
        window_left > 0U && global_window_size == 0U && q_group_count > 0U &&
        color_count > 1U &&
        dk_storage_elems <=
            dkv_color_workspace_limit_elems / (2U * color_count);
    const bool use_batch_dkv_color_partials =
        window_left > 0U && q_group_count > 0U && color_count > 1U &&
        !use_dkv_color_partials &&
        batch_dk_storage_elems <=
            dkv_color_workspace_limit_elems / (2U * color_count);
    const bool pack_batch_dkv_color_partials =
        use_batch_dkv_color_partials &&
        num_kv_heads <= 3U &&
        batch_dk_storage_elems <=
            packed_color_storage_rows * kv_stride;
    const bool use_chunked_dkv_color_partials =
        window_left > 0U && q_group_count > 0U && color_count > 1U &&
        !use_dkv_color_partials && !use_batch_dkv_color_partials &&
        color_chunk_rows >= TILE_K;
    const bool use_dkv_color_path =
        use_dkv_color_partials || use_batch_dkv_color_partials ||
        use_chunked_dkv_color_partials;
    const bool use_bf16_dkv_color_partials =
        use_dkv_color_partials &&
        num_q_heads == num_kv_heads;
    const bool use_fp16_dkv_color_partials =
        (use_batch_dkv_color_partials ||
            use_chunked_dkv_color_partials) &&
        num_q_heads > num_kv_heads &&
        num_q_heads <= 8U * num_kv_heads;
    const bool use_kv_owned_mha =
        use_bf16_dkv_color_partials &&
        total_q_tokens == total_kv_tokens &&
        kv_stride == DIM;
    const uint32_t kv_group_count =
        use_kv_owned_mha ?
            count_kv_groups<TILE_K>(
                cu_q_lens_gm, cu_kv_lens_gm, batch,
                num_kv_heads) :
            0U;
    const uint32_t kv_core_chunk_size =
        ceil_div(kv_group_count, active_aic_cores);
    const uint32_t dq_core_plane_count =
        kv_core_chunk_size == 0U ?
            0U :
            ceil_div(
                static_cast<uint32_t>(color_count) +
                    kv_core_chunk_size - 1U,
                kv_core_chunk_size);
    const bool use_kv_owned_dq_core_planes =
        use_kv_owned_mha && dq_core_plane_count > 0U &&
        dq_core_plane_count < color_count &&
        static_cast<uint64_t>(dq_core_plane_count) *
                dq_storage_elems <=
            2U * color_count * dk_storage_elems;
    const bool direct_dkv_output =
        global_window_size <= TILE_K &&
        (use_batch_dkv_color_partials ||
            use_chunked_dkv_color_partials);
    const bool use_compact_chunk_dq =
        global_window_size == 0U && direct_dkv_output &&
        use_chunked_dkv_color_partials;
    const bool use_compact_batch_dq =
        global_window_size == 0U && direct_dkv_output &&
        use_batch_dkv_color_partials;
    const uint32_t dq_ring_rows =
        use_compact_chunk_dq ?
            color_chunk_rows +
                ceil_div(local_window_radius(window_left), TILE_Q) * TILE_Q :
        use_compact_batch_dq ?
            batch * TILE_Q : 0U;
    const uint64_t physical_dq_workspace_elems =
        use_compact_chunk_dq || use_compact_batch_dq ?
            static_cast<uint64_t>(dq_ring_rows) * q_stride :
            dq_storage_elems;
    const uint64_t physical_workspace_elems =
        direct_dkv_output && global_window_size == 0U ?
            physical_dq_workspace_elems : workspace_elems;
    const bool use_direct_global_dq =
        global_window_size > 0U &&
        global_window_size <= TILE_K &&
        max_q_len > local_window_radius(window_left) + TILE_Q &&
        batch * 2U >= active_aic_cores &&
        use_dkv_color_path && !use_chunked_dkv_color_partials;
    const bool direct_scaled_dq =
        (global_window_size == 0U || use_direct_global_dq) &&
        use_dkv_color_path && !use_chunked_dkv_color_partials;
    const uint64_t color_dk_storage_elems =
        use_chunked_dkv_color_partials ?
            chunk_dk_storage_elems :
        pack_batch_dkv_color_partials ?
            batch_color_dk_storage_elems :
        use_batch_dkv_color_partials ?
            batch_dk_storage_elems : dk_storage_elems;
    const uint64_t dkv_color_elems =
        use_dkv_color_partials ?
            color_count * 2U * dk_storage_elems :
        pack_batch_dkv_color_partials ?
            color_count * 2U * batch_color_dk_storage_elems :
        use_batch_dkv_color_partials ?
            color_count * 2U * batch_dk_storage_elems :
        use_chunked_dkv_color_partials ?
            color_count * 2U * chunk_dk_storage_elems : 0U;
    const uint64_t global_partial_rows =
        global_partial_row_base<TILE_K>(
            cu_q_lens_gm, cu_kv_lens_gm, batch,
            window_left, global_window_size);
    const uint64_t global_dkv_plane_elems =
        global_partial_rows * kv_stride;
    const uint32_t global_dkv_partial_core_count =
        global_dkv_plane_elems > 0U ? active_aic_cores : 0U;
    const bool use_global_dkv_partials =
        global_dkv_partial_core_count > 0U;
    const uint64_t global_dkv_partial_offset =
        physical_workspace_elems + dkv_scratch_elems +
        dkv_color_elems;
    const bool zero_workspace =
        !direct_scaled_dq || use_chunked_dkv_color_partials ||
        use_direct_global_dq;
    if constexpr (g_coreType == AscendC::AIV) {
        if (zero_workspace || use_global_dkv_partials) {
            BackwardWorkspaceZero zero;
            zero.init();
            if (zero_workspace) {
                const uint64_t zero_start =
                    use_direct_global_dq ? dq_storage_elems : 0U;
                zero.zero(
                    user_workspace, zero_start,
                    physical_workspace_elems - zero_start,
                    active_aiv_cores);
            }
            if (use_global_dkv_partials) {
                zero.zero_global_dkv_partials(
                    user_workspace, global_dkv_partial_offset,
                    global_dkv_plane_elems,
                    global_dkv_partial_core_count,
                    cu_q_lens_gm, cu_kv_lens_gm, batch,
                    num_kv_heads, kv_stride, window_left,
                    global_window_size, active_aiv_cores);
            }
            AscendC::SyncAll<false>();
            zero.destroy();
        }
    } else {
        if (zero_workspace || use_global_dkv_partials) {
            AscendC::SyncAll<false>();
        }
    }

    if constexpr (g_coreType == AscendC::AIC) {
        const uint32_t core_idx = AscendC::GetBlockIdx();

        BackwardAIC aic;
        aic.init(q, k, v, dout, dq, dk, dv, user_workspace, batch,
            total_q_tokens,
            total_kv_tokens, q_stride, kv_stride, physical_workspace_elems,
            dkv_scratch_elems, dq_ring_rows, direct_scaled_dq);
        AicStageTask pending_grad[PIPELINE_DISTANCE] = {};
        uint32_t issue_idx = 0U;
        if (window_left > 0U && use_kv_owned_mha) {
            AscendC::SyncAll<false>();
            for (uint32_t group_ordinal = 0U,
                          linear_group = 0U;
                 use_kv_owned_dq_core_planes ?
                     scheduled_contiguous_group(
                         core_idx, group_ordinal, kv_group_count,
                         active_aic_cores,
                         linear_group) :
                     scheduled_local_round_candidate(
                         core_idx, group_ordinal, kv_group_count,
                         active_aic_cores, false,
                         linear_group);
                 ++group_ordinal) {
                TaskInfo kv_group;
                if (!fill_kv_group<TILE_K>(
                        linear_group, cu_q_lens_gm, cu_kv_lens_gm,
                        batch, num_kv_heads, kv_group)) {
                    continue;
                }
                const uint32_t q_tiles =
                    ceil_div(kv_group.q_len, TILE_Q);
                uint32_t first_q_tile = q_tiles;
                uint32_t last_q_tile = 0U;
                for (uint32_t q_tile = 0U; q_tile < q_tiles; ++q_tile) {
                    TaskInfo task;
                    if (fill_kv_owned_task<
                            TILE_Q, TILE_K,
                            (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                            kv_group, q_tile, num_kv_heads,
                            window_left, window_right, task)) {
                        if (first_q_tile == q_tiles) {
                            first_q_tile = q_tile;
                        }
                        last_q_tile = q_tile;
                    }
                }
                if (first_q_tile == q_tiles) {
                    continue;
                }
                aic.prefetch_kv(kv_group);
                aic.activate_kv();
                const uint32_t group_issue_begin = issue_idx;
                for (uint32_t q_tile = first_q_tile;
                     q_tile <= last_q_tile; ++q_tile) {
                    TaskInfo task;
                    if (!fill_kv_owned_task<
                            TILE_Q, TILE_K,
                            (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                            kv_group, q_tile, num_kv_heads,
                            window_left, window_right, task)) {
                        continue;
                    }
                    const uint32_t slot =
                        issue_idx % PIPELINE_DISTANCE;
                    if (issue_idx - group_issue_begin >=
                        PIPELINE_DISTANCE) {
                        AicStageTask& pending = pending_grad[slot];
                        aic.run_kv_owned_grads(
                            pending,
                            pending.task.q_tile == first_q_tile,
                            pending.task.q_tile == last_q_tile,
                            use_kv_owned_dq_core_planes);
                        aic.finish_q(pending.q_l1_slot);
                    }
                    aic.prefetch_q(task, slot);
                    aic.activate_q(slot);
                    AicStageTask stage =
                        make_aic_stage(task, issue_idx, 1U, slot, true);
                    stage.dkv_color_offset =
                        (use_kv_owned_dq_core_planes ?
                            core_idx % dq_core_plane_count :
                            static_cast<uint32_t>(
                                static_cast<int32_t>(task.q_tile) -
                                static_cast<int32_t>(task.kv_tile) -
                                color_min_delta)) *
                        dq_storage_elems;
                    if (use_kv_owned_dq_core_planes) {
                        const KvTileRange dq_range =
                            make_kv_tile_range<TILE_K, true>(
                                task.q_tile * TILE_Q, task.q_rows,
                                task.kv_len, window_left, 0U, 0U);
                        const uint32_t batch_group_begin =
                            linear_group - task.kv_tile;
                        const uint32_t range_begin =
                            batch_group_begin + dq_range.first_tile;
                        const uint32_t chunk_begin =
                            core_idx * kv_core_chunk_size;
                        const uint32_t first_group =
                            range_begin > chunk_begin ?
                                range_begin : chunk_begin;
                        stage.dq_atomic =
                            linear_group != first_group;
                    }
                    stage.dkv_color_dst_offset =
                        (static_cast<uint64_t>(task.q_global_begin) +
                            static_cast<uint64_t>(task.batch_idx) *
                                (TILE_Q - 1U)) *
                            q_stride +
                        static_cast<uint64_t>(task.q_head) * DIM;
                    aic.run_kv_owned_c1_c2(stage);
                    pending_grad[slot] = stage;
                    ++issue_idx;
                }
                const uint32_t group_issue_count =
                    issue_idx - group_issue_begin;
                if (group_issue_count >= PIPELINE_DISTANCE) {
                    AicStageTask& pending = pending_grad[
                        (issue_idx - PIPELINE_DISTANCE) %
                        PIPELINE_DISTANCE];
                    aic.run_kv_owned_grads(
                        pending,
                        pending.task.q_tile == first_q_tile,
                        pending.task.q_tile == last_q_tile,
                        use_kv_owned_dq_core_planes);
                    aic.finish_q(pending.q_l1_slot);
                }
                if (group_issue_count >= 1U) {
                    AicStageTask& pending = pending_grad[
                        (issue_idx - 1U) % PIPELINE_DISTANCE];
                    aic.run_kv_owned_grads(
                        pending,
                        pending.task.q_tile == first_q_tile,
                        pending.task.q_tile == last_q_tile,
                        use_kv_owned_dq_core_planes);
                    aic.finish_q(pending.q_l1_slot);
                }
                aic.finish_kv();
            }
        } else if (window_left > 0U && use_dkv_color_path) {
            const uint32_t gqa_ratio = num_q_heads / num_kv_heads;
            const int32_t min_delta =
                local_round_min_delta<TILE_Q, TILE_K>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch);
            const int32_t max_delta =
                local_round_max_delta<TILE_K>(window_left);
            const uint32_t wave_count =
                use_chunked_dkv_color_partials ?
                    count_local_color_chunk_waves(
                        cu_q_lens_gm, cu_kv_lens_gm, batch,
                        color_chunk_rows) :
                use_batch_dkv_color_partials ?
                    (pack_batch_dkv_color_partials ?
                        count_packed_batch_color_waves<TILE_K>(
                            cu_q_lens_gm, cu_kv_lens_gm, batch,
                            static_cast<uint32_t>(
                                packed_color_storage_rows)) :
                        batch) :
                    1U;
            bool has_preissued_wave_c1_c2 = false;
            TaskInfo preissued_wave_task = {};
            for (uint32_t wave = 0U; wave < wave_count; ++wave) {
            uint32_t batch_wave =
                use_batch_dkv_color_partials &&
                    !pack_batch_dkv_color_partials ?
                wave : 0U;
            uint32_t batch_wave_end =
                use_batch_dkv_color_partials &&
                    !pack_batch_dkv_color_partials ?
                wave + 1U : batch;
            uint32_t chunk_wave = 0U;
            if (use_chunked_dkv_color_partials &&
                !decode_local_color_chunk_wave(
                    wave, cu_q_lens_gm, cu_kv_lens_gm, batch,
                    color_chunk_rows, batch_wave, chunk_wave)) {
                continue;
            }
            if (use_chunked_dkv_color_partials) {
                batch_wave_end = batch_wave + 1U;
            } else if (pack_batch_dkv_color_partials &&
                !decode_packed_batch_color_wave<TILE_K>(
                    wave, cu_q_lens_gm, cu_kv_lens_gm, batch,
                    static_cast<uint32_t>(packed_color_storage_rows),
                    batch_wave, batch_wave_end)) {
                continue;
            }
            const uint64_t wave_color_dk_storage_elems =
                use_batch_dkv_color_partials && kv_stride == DIM ?
                    (static_cast<uint64_t>(
                        cu_kv_lens_gm.GetValue(batch_wave_end) -
                        cu_kv_lens_gm.GetValue(batch_wave)) +
                        static_cast<uint64_t>(
                            batch_wave_end - batch_wave) *
                            (TILE_K - 1U)) *
                        kv_stride :
                    color_dk_storage_elems;
            uint32_t candidate_begin = 0U;
            uint32_t candidate_count =
                count_local_round_candidates<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch, num_kv_heads);
            if (pack_batch_dkv_color_partials) {
                local_round_candidate_range_span<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm,
                    batch_wave, batch_wave_end, num_kv_heads,
                    candidate_begin, candidate_count);
            } else if (use_batch_dkv_color_partials ||
                use_chunked_dkv_color_partials) {
                local_round_candidate_range<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch_wave, num_kv_heads,
                    candidate_begin, candidate_count);
            }
            const uint32_t batch_kv_begin =
                use_batch_dkv_color_partials ||
                    use_chunked_dkv_color_partials ?
                cu_kv_lens_gm.GetValue(batch_wave) : 0U;
            const uint64_t batch_padded_begin =
                use_batch_dkv_color_partials ?
                    static_cast<uint64_t>(batch_kv_begin) +
                    static_cast<uint64_t>(batch_wave) *
                        (TILE_K - 1U) :
                    0U;
            const uint32_t chunk_kv_begin =
                use_chunked_dkv_color_partials ?
                    chunk_wave * color_chunk_rows : 0U;
            const uint32_t batch_kv_len =
                use_chunked_dkv_color_partials ?
                    cu_kv_lens_gm.GetValue(batch_wave + 1U) -
                        batch_kv_begin :
                    max_kv_len;
            const uint32_t chunk_kv_rows =
                use_chunked_dkv_color_partials ?
                    min_u32(
                        color_chunk_rows,
                        batch_kv_len - chunk_kv_begin) : 0U;
            uint32_t candidate_local_begin = 0U;
            if (use_chunked_dkv_color_partials) {
                const uint32_t batch_q_len =
                    cu_q_lens_gm.GetValue(batch_wave + 1U) -
                    cu_q_lens_gm.GetValue(batch_wave);
                local_round_chunk_candidate_range<TILE_Q, TILE_K>(
                    batch_q_len, batch_kv_len, num_kv_heads,
                    chunk_kv_begin, chunk_kv_rows, min_delta, max_delta,
                    candidate_local_begin, candidate_count);
            }
            const bool fold_candidate_boundary =
                candidate_count > active_aic_cores;
            const bool collaborate_tail_candidates =
                (use_dkv_color_partials ||
                    use_batch_dkv_color_partials) &&
                !use_bf16_dkv_color_partials && gqa_ratio > 1U &&
                fold_candidate_boundary &&
                use_tail_candidate_collaboration(
                    candidate_count, active_aic_cores);
            const int32_t collaborative_split_delta =
                min_delta + 2;
            const bool cache_q_d = false;
            if (cache_q_d) {
                AscendC::SyncAll<false>();
            }
            const uint32_t assigned_candidate_count =
                count_local_round_candidates_for_core(
                    core_idx, candidate_count, active_aic_cores,
                    fold_candidate_boundary);
            const bool retain_q_inputs = false;
            AicStageTask group_carried[PIPELINE_DISTANCE] = {};
            uint32_t group_carried_count = 0U;
            uint32_t group_next_q_l1_slot = 0U;
            for (uint32_t head_in_group = 0U;
                 head_in_group < gqa_ratio; ++head_in_group) {
                uint32_t resident_count = 0U;
                if (retain_q_inputs) {
                    for (uint32_t candidate_ordinal = 0U,
                                  linear_candidate = 0U;
                         scheduled_local_round_candidate(
                             core_idx, candidate_ordinal,
                             candidate_count, active_aic_cores,
                             fold_candidate_boundary, linear_candidate);
                         ++candidate_ordinal) {
                        TaskInfo task;
                        if (fill_local_round_task_span<
                                TILE_Q, TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                candidate_local_begin + linear_candidate,
                                candidate_begin, 0,
                                head_in_group,
                                cu_q_lens_gm, cu_kv_lens_gm,
                                batch_wave, batch_wave_end,
                                num_q_heads, num_kv_heads, window_left,
                                window_right, gqa_layout != 0U,
                                uniform_q_len, uniform_kv_len, task)) {
                            aic.prefetch_q(task, resident_count);
                            aic.activate_q(resident_count);
                            ++resident_count;
                        }
                    }
                }
                bool resident_dq_started = false;
                if (retain_q_inputs) {
                    AicStageTask carried[PIPELINE_DISTANCE] = {};
                    uint32_t carried_count = 0U;
                    for (int32_t delta = min_delta; delta <= max_delta; ++delta) {
                        AicStageTask current[PIPELINE_DISTANCE] = {};
                        uint32_t current_count = 0U;
                        uint32_t carried_done = 0U;
                        uint32_t candidate_slot = 0U;
                        for (uint32_t candidate_ordinal = 0U,
                                      linear_candidate = 0U;
                             scheduled_local_round_candidate(
                                 core_idx, candidate_ordinal,
                                 candidate_count, active_aic_cores,
                                 fold_candidate_boundary, linear_candidate);
                             ++candidate_ordinal, ++candidate_slot) {
                            TaskInfo task;
                            if (!fill_local_round_task_span<
                                    TILE_Q, TILE_K,
                                    (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                    candidate_local_begin + linear_candidate,
                                    candidate_begin,
                                    delta, head_in_group,
                                    cu_q_lens_gm, cu_kv_lens_gm,
                                    batch_wave, batch_wave_end,
                                    num_q_heads, num_kv_heads, window_left,
                                    window_right, gqa_layout != 0U,
                                    uniform_q_len, uniform_kv_len, task)) {
                                continue;
                            }
                            if (use_chunked_dkv_color_partials &&
                                !task_in_dkv_color_chunk(
                                    task, batch_kv_begin, chunk_kv_begin,
                                    chunk_kv_rows)) {
                                continue;
                            }
                            AicStageTask stage = make_aic_stage(
                                task, issue_idx, 1U, candidate_slot, true);
                            if (use_dkv_color_path) {
                                configure_dkv_color_stage(
                                    stage, head_in_group, delta,
                                    color_min_delta, color_count,
                                    wave_color_dk_storage_elems,
                                    use_batch_dkv_color_partials ||
                                        use_chunked_dkv_color_partials,
                                    batch_kv_begin, batch_padded_begin,
                                    kv_stride,
                                    use_chunked_dkv_color_partials,
                                    chunk_kv_begin,
                                    use_bf16_dkv_color_partials,
                                    use_fp16_dkv_color_partials);
                            }
                            if (resident_count == 1U) {
                                stage.dq_first = !resident_dq_started;
                                stage.dq_last = task.kv_tile_ordinal == 0U;
                                stage.dq_atomic = false;
                                resident_dq_started = true;
                            }
                            aic.run_c1_c2(stage);
                            if (carried_done < carried_count) {
                                aic.run_c3_c4_c5(carried[carried_done++]);
                            }
                            current[current_count++] = stage;
                            ++issue_idx;
                        }
                        while (carried_done < carried_count) {
                            aic.run_c3_c4_c5(carried[carried_done++]);
                        }
                        carried_count = current_count;
                        for (uint32_t slot = 0U; slot < current_count; ++slot) {
                            carried[slot] = current[slot];
                        }
                    }
                    for (uint32_t slot = 0U; slot < carried_count; ++slot) {
                        aic.run_c3_c4_c5(carried[slot]);
                    }
                } else if (!retain_q_inputs) {
                    uint32_t candidate_ordinal = 0U;
                    uint32_t linear_candidate = 0U;
                    bool helper_task = false;
                    bool tail_owner_task = false;
                    while (scheduled_local_round_collaborative_candidate(
                            core_idx, candidate_ordinal,
                            candidate_count, active_aic_cores,
                            fold_candidate_boundary,
                            collaborate_tail_candidates, linear_candidate,
                            helper_task, tail_owner_task)) {
                        TaskInfo first_task;
                        bool has_task = false;
                        int32_t last_delta = min_delta;
                        for (int32_t delta = min_delta;
                             delta <= max_delta; ++delta) {
                            if (helper_task &&
                                delta < collaborative_split_delta) {
                                continue;
                            }
                            TaskInfo task;
                            if (fill_local_round_task_span<
                                    TILE_Q, TILE_K,
                                    (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                    candidate_local_begin + linear_candidate,
                                    candidate_begin,
                                    delta, head_in_group,
                                    cu_q_lens_gm, cu_kv_lens_gm,
                                    batch_wave, batch_wave_end,
                                    num_q_heads, num_kv_heads, window_left,
                                    window_right, gqa_layout != 0U,
                                    uniform_q_len, uniform_kv_len,
                                    task) &&
                                (!use_chunked_dkv_color_partials ||
                                    task_in_dkv_color_chunk(
                                        task, batch_kv_begin,
                                        chunk_kv_begin, chunk_kv_rows))) {
                                if (!has_task) {
                                    first_task = task;
                                    has_task = true;
                                }
                                last_delta = delta;
                            }
                        }
                        if (!has_task) {
                            ++candidate_ordinal;
                            continue;
                        }
                        QTaskGroup global_group;
                        const bool has_fused_global_task =
                            use_direct_global_dq && !helper_task &&
                            fill_q_tile_wave_group<
                                TILE_Q, TILE_K,
                                (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                first_task.batch_idx, first_task.q_tile,
                                first_task.kv_head, head_in_group,
                                cu_q_lens_gm, cu_kv_lens_gm,
                                num_q_heads, num_kv_heads, window_left,
                                window_right, global_window_size, true,
                                gqa_layout != 0U, global_group);
                        const uint32_t q_l1_slot = group_next_q_l1_slot;
                        const bool use_preissued_wave_c1_c2 =
                            has_preissued_wave_c1_c2 &&
                            same_q_task(preissued_wave_task, first_task);
                        if (!use_preissued_wave_c1_c2) {
                            aic.prefetch_q(first_task, q_l1_slot);
                            aic.activate_q(q_l1_slot);
                        } else {
                            has_preissued_wave_c1_c2 = false;
                        }
                        const uint32_t candidate_issue_begin = issue_idx;
                        uint32_t carried_done = 0U;
                        bool dq_started = false;
                        for (int32_t delta = min_delta;
                             delta <= max_delta; ++delta) {
                            if (helper_task &&
                                delta < collaborative_split_delta) {
                                continue;
                            }
                            TaskInfo task;
                            if (!fill_local_round_task_span<
                                    TILE_Q, TILE_K,
                                    (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                    candidate_local_begin + linear_candidate,
                                    candidate_begin,
                                    delta, head_in_group,
                                    cu_q_lens_gm, cu_kv_lens_gm,
                                    batch_wave, batch_wave_end,
                                    num_q_heads, num_kv_heads, window_left,
                                    window_right, gqa_layout != 0U,
                                    uniform_q_len, uniform_kv_len, task)) {
                                continue;
                            }
                            if (use_chunked_dkv_color_partials &&
                                !task_in_dkv_color_chunk(
                                    task, batch_kv_begin, chunk_kv_begin,
                                    chunk_kv_rows)) {
                                continue;
                            }
                            const uint32_t pipeline_slot =
                                issue_idx % PIPELINE_DISTANCE;
                            if (issue_idx - candidate_issue_begin >=
                                PIPELINE_DISTANCE) {
                                aic.run_c3_c4_c5(
                                    pending_grad[pipeline_slot]);
                            }
                            AicStageTask current = make_aic_stage(
                                task, issue_idx, 1U, q_l1_slot, true);
                            configure_dkv_color_stage(
                                current, head_in_group, delta,
                                color_min_delta, color_count,
                                wave_color_dk_storage_elems,
                                use_batch_dkv_color_partials ||
                                    use_chunked_dkv_color_partials,
                                batch_kv_begin, batch_padded_begin,
                                kv_stride,
                                use_chunked_dkv_color_partials,
                                chunk_kv_begin,
                                use_bf16_dkv_color_partials,
                                use_fp16_dkv_color_partials);
                            current.dq_first = !dq_started;
                            current.dq_last =
                                !has_fused_global_task &&
                                (use_chunked_dkv_color_partials ?
                                    delta == last_delta :
                                    task.kv_tile_ordinal == 0U);
                            current.dq_atomic =
                                use_chunked_dkv_color_partials;
                            current.skip_dq = helper_task;
                            current.skip_dkv =
                                tail_owner_task &&
                                delta >= collaborative_split_delta;
                            dq_started = true;
                            aic.run_c1_c2(current);
                            if (carried_done < group_carried_count) {
                                aic.run_c3_c4_c5(
                                    group_carried[carried_done++]);
                            }
                            pending_grad[pipeline_slot] = current;
                            ++issue_idx;
                        }
                        if (has_fused_global_task) {
                            for (uint32_t ordinal = 0U;
                                 ordinal < global_group.kv_range.tile_count;
                                 ++ordinal) {
                                const uint32_t pipeline_slot =
                                    issue_idx % PIPELINE_DISTANCE;
                                if (issue_idx - candidate_issue_begin >=
                                    PIPELINE_DISTANCE) {
                                    aic.run_c3_c4_c5(
                                        pending_grad[pipeline_slot]);
                                }
                                const TaskInfo task =
                                    task_at<TILE_K>(global_group, ordinal);
                                AicStageTask current = make_aic_stage(
                                    task, issue_idx,
                                    global_group.kv_range.tile_count,
                                    q_l1_slot, true);
                                current.dq_first = !dq_started;
                                current.dq_last =
                                    ordinal + 1U ==
                                    global_group.kv_range.tile_count;
                                current.dq_atomic = false;
                                configure_global_dkv_stage(
                                    current, cu_q_lens_gm, cu_kv_lens_gm,
                                    core_idx, dkv_color_elems,
                                    global_dkv_plane_elems, kv_stride,
                                    window_left, global_window_size);
                                dq_started = true;
                                aic.run_c1_c2(current);
                                if (carried_done < group_carried_count) {
                                    aic.run_c3_c4_c5(
                                        group_carried[carried_done++]);
                                }
                                pending_grad[pipeline_slot] = current;
                                ++issue_idx;
                            }
                        }
                        while (carried_done < group_carried_count) {
                            aic.run_c3_c4_c5(
                                group_carried[carried_done++]);
                        }
                        if (group_carried_count > 0U) {
                            aic.finish_q(group_carried[0U].q_l1_slot);
                        }
                        const uint32_t candidate_issue_count =
                            issue_idx - candidate_issue_begin;
                        group_carried_count = min_u32(
                            PIPELINE_DISTANCE, candidate_issue_count);
                        for (uint32_t slot = 0U;
                             slot < group_carried_count; ++slot) {
                            group_carried[slot] = pending_grad[
                                (issue_idx - group_carried_count + slot) %
                                PIPELINE_DISTANCE];
                        }
                        if (group_carried_count > 0U) {
                            group_next_q_l1_slot =
                                (q_l1_slot + 1U) % BWD_Q_L1_SLOTS;
                        }
                        ++candidate_ordinal;
                    }
                } else {
                    for (int32_t delta = min_delta; delta <= max_delta; ++delta) {
                        const uint32_t round_issue_begin = issue_idx;
                        for (uint32_t candidate_ordinal = 0U,
                                      linear_candidate = 0U;
                             scheduled_local_round_candidate(
                                 core_idx, candidate_ordinal,
                                 candidate_count, active_aic_cores,
                                 fold_candidate_boundary, linear_candidate);
                             ++candidate_ordinal) {
                            TaskInfo task;
                            if (!fill_local_round_task_span<
                                    TILE_Q, TILE_K,
                                    (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                    candidate_local_begin + linear_candidate,
                                    candidate_begin,
                                    delta, head_in_group,
                                    cu_q_lens_gm, cu_kv_lens_gm,
                                    batch_wave, batch_wave_end,
                                    num_q_heads, num_kv_heads, window_left,
                                    window_right, gqa_layout != 0U,
                                    uniform_q_len, uniform_kv_len, task)) {
                                continue;
                            }
                            if (use_chunked_dkv_color_partials &&
                                !task_in_dkv_color_chunk(
                                    task, batch_kv_begin, chunk_kv_begin,
                                    chunk_kv_rows)) {
                                continue;
                            }
                            const uint32_t pipeline_slot =
                                issue_idx % PIPELINE_DISTANCE;
                            if (issue_idx - round_issue_begin >=
                                PIPELINE_DISTANCE) {
                                aic.run_c3_c4_c5(pending_grad[pipeline_slot]);
                                aic.finish_q(
                                    pending_grad[pipeline_slot].q_l1_slot);
                            }
                            aic.prefetch_q(task, pipeline_slot);
                            aic.activate_q(pipeline_slot);
                            AicStageTask current = make_aic_stage(
                                task, issue_idx, 1U, pipeline_slot, true);
                            if (use_dkv_color_path) {
                                configure_dkv_color_stage(
                                    current, head_in_group, delta,
                                    color_min_delta, color_count,
                                    wave_color_dk_storage_elems,
                                    use_batch_dkv_color_partials ||
                                        use_chunked_dkv_color_partials,
                                    batch_kv_begin, batch_padded_begin,
                                    kv_stride,
                                    use_chunked_dkv_color_partials,
                                    chunk_kv_begin,
                                    use_bf16_dkv_color_partials,
                                    use_fp16_dkv_color_partials);
                            }
                            aic.run_c1_c2(current);
                            pending_grad[pipeline_slot] = current;
                            ++issue_idx;
                        }
                        const uint32_t round_issue_count =
                            issue_idx - round_issue_begin;
                        if (round_issue_count >= PIPELINE_DISTANCE) {
                            AicStageTask& stage = pending_grad[
                                (issue_idx - PIPELINE_DISTANCE) %
                                PIPELINE_DISTANCE];
                            aic.run_c3_c4_c5(stage);
                            aic.finish_q(stage.q_l1_slot);
                        }
                        if (round_issue_count >= 1U) {
                            AicStageTask& stage =
                                pending_grad[(issue_idx - 1U) % PIPELINE_DISTANCE];
                            aic.run_c3_c4_c5(stage);
                            aic.finish_q(stage.q_l1_slot);
                        }
                    }
                }
                for (uint32_t slot = 0U; slot < resident_count; ++slot) {
                    aic.finish_q(slot);
                }
            }
            if (!retain_q_inputs) {
                for (uint32_t slot = 0U;
                     slot < group_carried_count; ++slot) {
                    aic.run_c3_c4_c5(group_carried[slot]);
                }
                if (group_carried_count > 0U) {
                    aic.finish_q(group_carried[0U].q_l1_slot);
                }
            }
            if (use_batch_dkv_color_partials ||
                use_chunked_dkv_color_partials) {
                aic.fence_grad_writes();
                AscendC::SyncAll<false>();
                if (wave + 1U < wave_count) {
                    TaskInfo next_wave_task;
                    if (pack_batch_dkv_color_partials &&
                        first_packed_wave_task_for_core(
                            core_idx, wave + 1U,
                            cu_q_lens_gm, cu_kv_lens_gm, batch,
                            num_q_heads, num_kv_heads,
                            active_aic_cores,
                            static_cast<uint32_t>(
                                packed_color_storage_rows),
                            window_left, window_right, gqa_layout,
                            uniform_q_len, uniform_kv_len,
                            min_delta, max_delta,
                            use_bf16_dkv_color_partials,
                            next_wave_task)) {
                        aic.prefetch_q(next_wave_task, 0U);
                        aic.activate_q(0U);
                        const AicStageTask next_wave_stage =
                            make_aic_stage(
                                next_wave_task, issue_idx, 1U, 0U, true);
                        aic.preissue_c1_c2(next_wave_stage);
                        preissued_wave_task = next_wave_task;
                        has_preissued_wave_c1_c2 = true;
                    }
                    AscendC::SyncAll<false>();
                }
            }
            }
        } else if (window_left > 0U) {
            const uint32_t gqa_ratio = num_q_heads / num_kv_heads;
            const uint32_t wave_count =
                count_q_tile_waves<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch);
            for (uint32_t wave = 0U; wave < wave_count; ++wave) {
                uint32_t batch_wave = 0U;
                uint32_t q_tile_wave = 0U;
                if (!decode_q_tile_wave<TILE_Q>(
                        wave, cu_q_lens_gm, cu_kv_lens_gm, batch,
                        batch_wave, q_tile_wave)) {
                    continue;
                }
                for (uint32_t kv_ordinal = 0U, kv_head = 0U;
                     scheduled_local_round_candidate(
                         core_idx, kv_ordinal, num_kv_heads,
                         active_aic_cores, false, kv_head);
                     ++kv_ordinal) {
                    for (uint32_t head_in_group = 0U;
                         head_in_group < gqa_ratio; ++head_in_group) {
                        QTaskGroup group;
                        if (!fill_q_tile_wave_group<
                                TILE_Q, TILE_K,
                                (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                batch_wave, q_tile_wave, kv_head,
                                head_in_group, cu_q_lens_gm, cu_kv_lens_gm,
                                num_q_heads, num_kv_heads, window_left,
                                window_right, global_window_size, false,
                                gqa_layout != 0U,
                                group)) {
                            continue;
                        }
                        aic.prefetch_q(group.q_task, 0U);
                        aic.activate_q(0U);
                        for (uint32_t ordinal = 0U;
                             ordinal < group.kv_range.tile_count; ++ordinal) {
                            const TaskInfo task =
                                task_at<TILE_K>(group, ordinal);
                            const AicStageTask stage = make_aic_stage(
                                task, issue_idx, group.kv_range.tile_count,
                                0U, true);
                            aic.run_c1_c2(stage);
                            aic.run_c3_c4_c5(stage);
                            ++issue_idx;
                        }
                        aic.finish_q(0U);
                    }
                }
                aic.fence_grad_writes();
                AscendC::SyncAll<false>();
            }
        }
        if (global_window_size > 0U && use_dkv_color_path &&
            !use_direct_global_dq) {
            AscendC::SyncAll<false>();
        }
        if (global_window_size > 0U && !use_direct_global_dq) {
            for (uint32_t group_ordinal = 0U, linear_group = 0U;
                 scheduled_global_q_group(
                     core_idx, group_ordinal, q_group_count,
                     global_dkv_partial_core_count,
                     linear_group);
                 ++group_ordinal) {
                QTaskGroup group;
                if (!decode_global_q_group(
                        linear_group, cu_q_lens_gm, cu_kv_lens_gm,
                        batch, num_q_heads, num_kv_heads, window_left,
                        window_right, global_window_size,
                        gqa_layout != 0U, group)) {
                    continue;
                }
                aic.prefetch_q(group.q_task, 0U);
                aic.activate_q(0U);
                for (uint32_t ordinal = 0U;
                     ordinal < group.kv_range.tile_count; ++ordinal) {
                    const TaskInfo task = task_at<TILE_K>(group, ordinal);
                    AicStageTask stage = make_aic_stage(
                        task, issue_idx, group.kv_range.tile_count,
                        0U, true);
                    if (use_global_dkv_partials) {
                        configure_global_dkv_stage(
                            stage, cu_q_lens_gm, cu_kv_lens_gm,
                            core_idx, dkv_color_elems,
                            global_dkv_plane_elems, kv_stride,
                            window_left, global_window_size);
                    }
                    aic.run_c1_c2(stage);
                    aic.run_c3_c4_c5(stage);
                    ++issue_idx;
                }
                aic.finish_q(0U);
                if (use_global_dkv_partials) {
                    aic.fence_grad_writes();
                }
            }
        }
        aic.finish();
        AscendC::SyncAll<false>();
        aic.destroy();
    } else if constexpr (g_coreType == AscendC::AIV) {
        const uint32_t aiv_idx = AscendC::GetBlockIdx();
        const uint32_t core_idx = (aiv_idx / AIVS_PER_AIC) % active_aic_cores;

        BackwardAIV aiv;
        aiv.init(
            out, dout, softmax_lse, dq, dk, dv, user_workspace,
            dq_storage_elems, dk_storage_elems,
            physical_workspace_elems, dkv_scratch_elems,
            active_aiv_cores,
            total_q_tokens, total_kv_tokens, q_stride, kv_stride,
            num_q_heads, num_kv_heads, output_is_f32 != 0U,
            direct_scaled_dq);
        uint32_t issue_idx = 0U;
        if (window_left > 0U && use_kv_owned_mha) {
            for (uint32_t group_ordinal = 0U,
                          linear_group = 0U;
                 scheduled_global_q_group(
                     core_idx, group_ordinal, q_group_count,
                     global_dkv_partial_core_count,
                     linear_group);
                 ++group_ordinal) {
                QTaskGroup group;
                if (decode_q_group(
                        linear_group, cu_q_lens_gm, cu_kv_lens_gm,
                        batch, num_q_heads, num_kv_heads,
                        window_left, window_right, 0U,
                        gqa_layout != 0U, group)) {
                    aiv.prepare_q(group.q_task);
                    aiv.cache_prepared_d(group.q_task, true);
                }
            }
            AscendC::SyncAll<false>();
            for (uint32_t group_ordinal = 0U,
                          linear_group = 0U;
                 use_kv_owned_dq_core_planes ?
                     scheduled_contiguous_group(
                         core_idx, group_ordinal, kv_group_count,
                         active_aic_cores,
                         linear_group) :
                     scheduled_local_round_candidate(
                         core_idx, group_ordinal, kv_group_count,
                         active_aic_cores, false,
                         linear_group);
                 ++group_ordinal) {
                TaskInfo kv_group;
                if (!fill_kv_group<TILE_K>(
                        linear_group, cu_q_lens_gm, cu_kv_lens_gm,
                        batch, num_kv_heads, kv_group)) {
                    continue;
                }
                const uint32_t q_tiles =
                    ceil_div(kv_group.q_len, TILE_Q);
                for (uint32_t q_tile = 0U; q_tile < q_tiles; ++q_tile) {
                    TaskInfo task;
                    if (!fill_kv_owned_task<
                            TILE_Q, TILE_K,
                            (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                            kv_group, q_tile, num_kv_heads,
                            window_left, window_right, task)) {
                        continue;
                    }
                    aiv.prepare_q_from_cached_d(task, true);
                    const AivStageTask stage =
                        make_aiv_stage(task, issue_idx);
                    aiv.run_v2_v3_v4(
                        stage, window_left, window_right, 0U, scale);
                    aiv.release_mm_slot(stage);
                    ++issue_idx;
                }
            }
        } else if (window_left > 0U && use_dkv_color_path) {
            const uint32_t gqa_ratio = num_q_heads / num_kv_heads;
            const int32_t min_delta =
                local_round_min_delta<TILE_Q, TILE_K>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch);
            const int32_t max_delta =
                local_round_max_delta<TILE_K>(window_left);
            const uint32_t wave_count =
                use_chunked_dkv_color_partials ?
                    count_local_color_chunk_waves(
                        cu_q_lens_gm, cu_kv_lens_gm, batch,
                        color_chunk_rows) :
                use_batch_dkv_color_partials ?
                    (pack_batch_dkv_color_partials ?
                        count_packed_batch_color_waves<TILE_K>(
                            cu_q_lens_gm, cu_kv_lens_gm, batch,
                            static_cast<uint32_t>(
                                packed_color_storage_rows)) :
                        batch) :
                    1U;
            for (uint32_t wave = 0U; wave < wave_count; ++wave) {
            uint32_t batch_wave =
                use_batch_dkv_color_partials &&
                    !pack_batch_dkv_color_partials ?
                wave : 0U;
            uint32_t batch_wave_end =
                use_batch_dkv_color_partials &&
                    !pack_batch_dkv_color_partials ?
                wave + 1U : batch;
            uint32_t chunk_wave = 0U;
            if (use_chunked_dkv_color_partials &&
                !decode_local_color_chunk_wave(
                    wave, cu_q_lens_gm, cu_kv_lens_gm, batch,
                    color_chunk_rows, batch_wave, chunk_wave)) {
                continue;
            }
            if (use_chunked_dkv_color_partials) {
                batch_wave_end = batch_wave + 1U;
            } else if (pack_batch_dkv_color_partials &&
                !decode_packed_batch_color_wave<TILE_K>(
                    wave, cu_q_lens_gm, cu_kv_lens_gm, batch,
                    static_cast<uint32_t>(packed_color_storage_rows),
                    batch_wave, batch_wave_end)) {
                continue;
            }
            const uint64_t wave_color_dk_storage_elems =
                use_batch_dkv_color_partials && kv_stride == DIM ?
                    (static_cast<uint64_t>(
                        cu_kv_lens_gm.GetValue(batch_wave_end) -
                        cu_kv_lens_gm.GetValue(batch_wave)) +
                        static_cast<uint64_t>(
                            batch_wave_end - batch_wave) *
                            (TILE_K - 1U)) *
                        kv_stride :
                    color_dk_storage_elems;
            uint32_t candidate_begin = 0U;
            uint32_t candidate_count =
                count_local_round_candidates<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch, num_kv_heads);
            if (pack_batch_dkv_color_partials) {
                local_round_candidate_range_span<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm,
                    batch_wave, batch_wave_end, num_kv_heads,
                    candidate_begin, candidate_count);
            } else if (use_batch_dkv_color_partials ||
                use_chunked_dkv_color_partials) {
                local_round_candidate_range<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch_wave, num_kv_heads,
                    candidate_begin, candidate_count);
            }
            const uint32_t batch_kv_begin =
                use_batch_dkv_color_partials ||
                    use_chunked_dkv_color_partials ?
                cu_kv_lens_gm.GetValue(batch_wave) : 0U;
            const uint64_t batch_padded_begin =
                use_batch_dkv_color_partials ?
                    static_cast<uint64_t>(batch_kv_begin) +
                    static_cast<uint64_t>(batch_wave) *
                        (TILE_K - 1U) :
                    0U;
            const uint32_t chunk_kv_begin =
                use_chunked_dkv_color_partials ?
                    chunk_wave * color_chunk_rows : 0U;
            const uint32_t batch_kv_len =
                use_chunked_dkv_color_partials ?
                    cu_kv_lens_gm.GetValue(batch_wave + 1U) -
                        batch_kv_begin :
                    max_kv_len;
            const uint32_t chunk_kv_rows =
                use_chunked_dkv_color_partials ?
                    min_u32(
                        color_chunk_rows,
                        batch_kv_len - chunk_kv_begin) : 0U;
            uint32_t candidate_local_begin = 0U;
            if (use_chunked_dkv_color_partials) {
                const uint32_t batch_q_len =
                    cu_q_lens_gm.GetValue(batch_wave + 1U) -
                    cu_q_lens_gm.GetValue(batch_wave);
                local_round_chunk_candidate_range<TILE_Q, TILE_K>(
                    batch_q_len, batch_kv_len, num_kv_heads,
                    chunk_kv_begin, chunk_kv_rows, min_delta, max_delta,
                    candidate_local_begin, candidate_count);
            }
            const bool fold_candidate_boundary =
                candidate_count > active_aic_cores;
            const bool collaborate_tail_candidates =
                (use_dkv_color_partials ||
                    use_batch_dkv_color_partials) &&
                !use_bf16_dkv_color_partials && gqa_ratio > 1U &&
                fold_candidate_boundary &&
                use_tail_candidate_collaboration(
                    candidate_count, active_aic_cores);
            const int32_t collaborative_split_delta =
                min_delta + 2;
            const bool cache_q_d = false;
            const bool pack_cached_lse =
                static_cast<uint64_t>(candidate_count) * gqa_ratio *
                    AIVS_PER_AIC * 2U * BWD_LOCAL_ROWS <=
                dkv_scratch_elems;
            if (cache_q_d) {
                for (uint32_t head_in_group = 0U;
                     head_in_group < gqa_ratio; ++head_in_group) {
                    for (uint32_t candidate_ordinal = 0U,
                                  linear_candidate = 0U;
                         scheduled_local_round_candidate(
                             core_idx, candidate_ordinal,
                             candidate_count, active_aic_cores,
                             fold_candidate_boundary, linear_candidate);
                         ++candidate_ordinal) {
                        TaskInfo task;
                        if (fill_local_round_task_span<
                                TILE_Q, TILE_K,
                                (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                candidate_local_begin + linear_candidate,
                                candidate_begin, 0,
                                head_in_group,
                                cu_q_lens_gm, cu_kv_lens_gm,
                                batch_wave, batch_wave_end,
                                num_q_heads, num_kv_heads, window_left,
                                window_right, gqa_layout != 0U,
                                uniform_q_len, uniform_kv_len, task)) {
                            aiv.prepare_q(task);
                            aiv.cache_prepared_d(
                                task, pack_cached_lse);
                        }
                    }
                }
                AscendC::SyncAll<false>();
            }
            const uint32_t assigned_candidate_count =
                count_local_round_candidates_for_core(
                    core_idx, candidate_count, active_aic_cores,
                    fold_candidate_boundary);
            const bool retain_q_inputs =
                !use_chunked_dkv_color_partials && assigned_candidate_count == 1U;
            const bool candidate_major =
                !retain_q_inputs;
            for (uint32_t head_in_group = 0U;
                 head_in_group < gqa_ratio; ++head_in_group) {
                if (candidate_major) {
                    uint32_t candidate_ordinal = 0U;
                    uint32_t linear_candidate = 0U;
                    bool helper_task = false;
                    bool tail_owner_task = false;
                    while (scheduled_local_round_collaborative_candidate(
                            core_idx, candidate_ordinal,
                            candidate_count, active_aic_cores,
                            fold_candidate_boundary,
                            collaborate_tail_candidates, linear_candidate,
                            helper_task, tail_owner_task)) {
                        TaskInfo first_task;
                        bool has_task = false;
                        for (int32_t delta = min_delta;
                             delta <= max_delta; ++delta) {
                            if (helper_task &&
                                delta < collaborative_split_delta) {
                                continue;
                            }
                            TaskInfo task;
                            if (fill_local_round_task_span<
                                    TILE_Q, TILE_K,
                                    (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                    candidate_local_begin + linear_candidate,
                                    candidate_begin,
                                    delta, head_in_group,
                                    cu_q_lens_gm, cu_kv_lens_gm,
                                    batch_wave, batch_wave_end,
                                    num_q_heads, num_kv_heads, window_left,
                                    window_right, gqa_layout != 0U,
                                    uniform_q_len, uniform_kv_len,
                                    task) &&
                                (!use_chunked_dkv_color_partials ||
                                    task_in_dkv_color_chunk(
                                        task, batch_kv_begin,
                                        chunk_kv_begin, chunk_kv_rows))) {
                                if (!has_task) {
                                    first_task = task;
                                    has_task = true;
                                }
                            }
                        }
                        if (!has_task) {
                            ++candidate_ordinal;
                            continue;
                        }
                        QTaskGroup global_group;
                        const bool has_fused_global_task =
                            use_direct_global_dq && !helper_task &&
                            fill_q_tile_wave_group<
                                TILE_Q, TILE_K,
                                (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                first_task.batch_idx, first_task.q_tile,
                                first_task.kv_head, head_in_group,
                                cu_q_lens_gm, cu_kv_lens_gm,
                                num_q_heads, num_kv_heads, window_left,
                                window_right, global_window_size, true,
                                gqa_layout != 0U, global_group);
                        aiv.prepare_q(first_task);
                        for (int32_t delta = min_delta;
                             delta <= max_delta; ++delta) {
                            if (helper_task &&
                                delta < collaborative_split_delta) {
                                continue;
                            }
                            TaskInfo task;
                            if (!fill_local_round_task_span<
                                    TILE_Q, TILE_K,
                                    (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                    candidate_local_begin + linear_candidate,
                                    candidate_begin,
                                    delta, head_in_group,
                                    cu_q_lens_gm, cu_kv_lens_gm,
                                    batch_wave, batch_wave_end,
                                    num_q_heads, num_kv_heads, window_left,
                                    window_right, gqa_layout != 0U,
                                    uniform_q_len, uniform_kv_len, task)) {
                                continue;
                            }
                            if (use_chunked_dkv_color_partials &&
                                !task_in_dkv_color_chunk(
                                    task, batch_kv_begin, chunk_kv_begin,
                                    chunk_kv_rows)) {
                                continue;
                            }
                            const AivStageTask current =
                                make_aiv_stage(task, issue_idx);
                            aiv.run_v2_v3_v4(
                                current, window_left, window_right, 0U, scale);
                            aiv.release_mm_slot(current);
                            ++issue_idx;
                        }
                        if (has_fused_global_task) {
                            for (uint32_t ordinal = 0U;
                                 ordinal < global_group.kv_range.tile_count;
                                 ++ordinal) {
                                const AivStageTask current =
                                    make_aiv_stage(
                                        task_at<TILE_K>(
                                            global_group, ordinal),
                                        issue_idx);
                                aiv.run_v2_v3_v4(
                                    current, window_left, window_right,
                                    global_window_size, scale);
                                aiv.release_mm_slot(current);
                                ++issue_idx;
                            }
                        }
                        ++candidate_ordinal;
                    }
                    continue;
                }
                uint32_t resident_candidate = 0U;
                if (retain_q_inputs &&
                    scheduled_local_round_candidate(
                        core_idx, 0U, candidate_count,
                        active_aic_cores,
                        fold_candidate_boundary, resident_candidate)) {
                    TaskInfo task;
                    if (fill_local_round_task_span<
                            TILE_Q, TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                            candidate_local_begin + resident_candidate,
                            candidate_begin, 0,
                            head_in_group,
                            cu_q_lens_gm, cu_kv_lens_gm,
                            batch_wave, batch_wave_end,
                            num_q_heads, num_kv_heads, window_left,
                            window_right, gqa_layout != 0U,
                            uniform_q_len, uniform_kv_len, task)) {
                        aiv.prepare_q(task);
                    }
                }
                for (int32_t delta = min_delta; delta <= max_delta; ++delta) {
                    for (uint32_t candidate_ordinal = 0U,
                                  linear_candidate = 0U;
                         scheduled_local_round_candidate(
                             core_idx, candidate_ordinal,
                             candidate_count, active_aic_cores,
                             fold_candidate_boundary, linear_candidate);
                         ++candidate_ordinal) {
                        TaskInfo task;
                        if (!fill_local_round_task_span<
                                TILE_Q, TILE_K, (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                candidate_local_begin + linear_candidate,
                                candidate_begin,
                                delta, head_in_group,
                                cu_q_lens_gm, cu_kv_lens_gm,
                                batch_wave, batch_wave_end,
                                num_q_heads, num_kv_heads, window_left,
                                window_right, gqa_layout != 0U,
                                uniform_q_len, uniform_kv_len, task)) {
                            continue;
                        }
                        if (use_chunked_dkv_color_partials &&
                            !task_in_dkv_color_chunk(
                                task, batch_kv_begin, chunk_kv_begin,
                                chunk_kv_rows)) {
                            continue;
                        }
                        if (!retain_q_inputs) {
                            if (cache_q_d) {
                                aiv.prepare_q_from_cached_d(
                                    task, pack_cached_lse);
                            } else {
                                aiv.prepare_q(task);
                            }
                        }
                        const AivStageTask current =
                            make_aiv_stage(task, issue_idx);
                        aiv.run_v2_v3_v4(
                            current, window_left, window_right, 0U, scale);
                        aiv.release_mm_slot(current);
                        ++issue_idx;
                    }
                }
            }
            if (use_batch_dkv_color_partials ||
                use_chunked_dkv_color_partials) {
                AscendC::SyncAll<false>();
                uint64_t reduction_job_base = 0U;
                for (uint32_t reduce_batch = batch_wave;
                     reduce_batch < batch_wave_end; ++reduce_batch) {
                    const uint32_t q_begin =
                        cu_q_lens_gm.GetValue(reduce_batch);
                    const uint32_t q_len =
                        cu_q_lens_gm.GetValue(reduce_batch + 1U) -
                        q_begin;
                    const uint32_t kv_begin =
                        cu_kv_lens_gm.GetValue(reduce_batch);
                    const uint32_t kv_len =
                        cu_kv_lens_gm.GetValue(reduce_batch + 1U) -
                        kv_begin;
                    if (kv_len < q_len) {
                        continue;
                    }
                    const uint32_t reduce_kv_begin =
                        use_chunked_dkv_color_partials ?
                            chunk_kv_begin : 0U;
                    const uint32_t reduce_kv_rows =
                        use_chunked_dkv_color_partials ?
                            chunk_kv_rows : kv_len;
                    const uint32_t preserve_prefix_rows = min_u32(
                        global_window_size,
                        q_len > 0U && kv_len >= q_len &&
                                kv_len > window_left ?
                            kv_len - window_left : 0U);
                    const uint64_t workspace_row_offset =
                        use_batch_dkv_color_partials ?
                            static_cast<uint64_t>(kv_begin) +
                            static_cast<uint64_t>(reduce_batch) *
                                (TILE_K - 1U) -
                            batch_padded_begin :
                            0U;
                    aiv.reduce_dkv_color_batch_range(
                        reduce_batch, q_len, kv_len, kv_begin,
                        reduce_kv_begin, reduce_kv_rows, window_left,
                        preserve_prefix_rows,
                        color_min_delta, color_max_delta,
                        wave_color_dk_storage_elems,
                        workspace_row_offset, reduction_job_base,
                        active_aiv_cores,
                        use_fp16_dkv_color_partials,
                        direct_dkv_output,
                        direct_scaled_dq ? 1.0F : scale);
                    reduction_job_base +=
                        static_cast<uint64_t>(
                            ceil_div(
                                reduce_kv_rows,
                                TILE_K / BWD_AIVS_PER_AIC)) *
                        num_kv_heads * BWD_DKV_GRAD_KINDS;
                }
                if (use_compact_chunk_dq) {
                    const uint32_t q_global_base =
                        cu_q_lens_gm.GetValue(batch_wave);
                    const uint32_t q_len =
                        cu_q_lens_gm.GetValue(batch_wave + 1U) -
                        q_global_base;
                    const uint32_t kv_len =
                        cu_kv_lens_gm.GetValue(batch_wave + 1U) -
                        cu_kv_lens_gm.GetValue(batch_wave);
                    const uint32_t kv_prefix = kv_len - q_len;
                    const uint32_t chunk_kv_end =
                        chunk_kv_begin + chunk_kv_rows;
                    const uint32_t q_local_begin =
                        chunk_kv_begin > kv_prefix ?
                            min_u32(q_len, chunk_kv_begin - kv_prefix) :
                            0U;
                    const uint32_t q_local_end =
                        chunk_kv_end > kv_prefix ?
                            min_u32(q_len, chunk_kv_end - kv_prefix) :
                            0U;
                    const uint32_t q_rows =
                        q_local_end - q_local_begin;
                    const uint32_t ring_begin =
                        q_local_begin % dq_ring_rows;
                    const uint32_t first_rows =
                        min_u32(q_rows, dq_ring_rows - ring_begin);
                    if (first_rows > 0U) {
                        aiv.finalize_dq_ring_range(
                            q_global_base + q_local_begin,
                            ring_begin, first_rows, scale);
                    }
                    if (q_rows > first_rows) {
                        aiv.finalize_dq_ring_range(
                            q_global_base + q_local_begin + first_rows,
                            0U, q_rows - first_rows, scale);
                    }
                }
                if (wave + 1U < wave_count) {
                    AscendC::SyncAll<false>();
                }
            }
            }
        } else if (window_left > 0U) {
            const uint32_t gqa_ratio = num_q_heads / num_kv_heads;
            const uint32_t wave_count =
                count_q_tile_waves<TILE_Q>(
                    cu_q_lens_gm, cu_kv_lens_gm, batch);
            for (uint32_t wave = 0U; wave < wave_count; ++wave) {
                uint32_t batch_wave = 0U;
                uint32_t q_tile_wave = 0U;
                if (!decode_q_tile_wave<TILE_Q>(
                        wave, cu_q_lens_gm, cu_kv_lens_gm, batch,
                        batch_wave, q_tile_wave)) {
                    continue;
                }
                for (uint32_t kv_ordinal = 0U, kv_head = 0U;
                     scheduled_local_round_candidate(
                         core_idx, kv_ordinal, num_kv_heads,
                         active_aic_cores, false, kv_head);
                     ++kv_ordinal) {
                    for (uint32_t head_in_group = 0U;
                         head_in_group < gqa_ratio; ++head_in_group) {
                        QTaskGroup group;
                        if (!fill_q_tile_wave_group<
                                TILE_Q, TILE_K,
                                (SWA_BWD_ENTRY_IS_CAUSAL != 0)>(
                                batch_wave, q_tile_wave, kv_head,
                                head_in_group, cu_q_lens_gm, cu_kv_lens_gm,
                                num_q_heads, num_kv_heads, window_left,
                                window_right, global_window_size, false,
                                gqa_layout != 0U,
                                group)) {
                            continue;
                        }
                        aiv.prepare_q(group.q_task);
                        for (uint32_t ordinal = 0U;
                             ordinal < group.kv_range.tile_count; ++ordinal) {
                            const AivStageTask stage = make_aiv_stage(
                                task_at<TILE_K>(group, ordinal), issue_idx);
                            aiv.run_v2_v3_v4(
                                stage, window_left, window_right, 0U, scale);
                            aiv.release_mm_slot(stage);
                            ++issue_idx;
                        }
                    }
                }
                AscendC::SyncAll<false>();
            }
        }
        if (global_window_size > 0U && use_dkv_color_path &&
            !use_direct_global_dq) {
            AscendC::SyncAll<false>();
        }
        if (global_window_size > 0U && !use_direct_global_dq) {
            for (uint32_t group_ordinal = 0U, linear_group = 0U;
                 scheduled_q_group(
                     core_idx, group_ordinal, q_group_count,
                     active_aic_cores, linear_group);
                 ++group_ordinal) {
                QTaskGroup group;
                if (!decode_global_q_group(
                        linear_group, cu_q_lens_gm, cu_kv_lens_gm,
                        batch, num_q_heads, num_kv_heads, window_left,
                        window_right, global_window_size,
                        gqa_layout != 0U, group)) {
                    continue;
                }
                aiv.prepare_q(group.q_task);
                for (uint32_t ordinal = 0U;
                     ordinal < group.kv_range.tile_count; ++ordinal) {
                    const AivStageTask stage = make_aiv_stage(
                        task_at<TILE_K>(group, ordinal), issue_idx);
                    aiv.run_v2_v3_v4(
                        stage, window_left, window_right,
                        global_window_size, scale);
                    aiv.release_mm_slot(stage);
                    ++issue_idx;
                }
            }
        }
        aiv.finish();
        AscendC::SyncAll<false>();
        aiv.destroy();
    }

    if constexpr (g_coreType == AscendC::AIV) {
        AscendC::GlobalTensor<float> src;
        AscendC::GlobalTensor<bfloat16_t> color_bf16_src;
        AscendC::GlobalTensor<bfloat16_t> dq_dst;
        AscendC::GlobalTensor<bfloat16_t> dk_dst;
        AscendC::GlobalTensor<bfloat16_t> dv_dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(user_workspace));
        color_bf16_src.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(user_workspace) +
            2ULL * (physical_workspace_elems + dkv_scratch_elems));
        dq_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dq));
        dk_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dk));
        dv_dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dv));

        BackwardFinalize finalize;
        finalize.init();
        if (use_kv_owned_mha) {
            if (use_kv_owned_dq_core_planes) {
                finalize.reduce_cast_dq_planes(
                    src, physical_workspace_elems + dkv_scratch_elems,
                    dq_storage_elems, dq_core_plane_count, dq_dst,
                    cu_q_lens_gm, batch, num_q_heads, q_stride,
                    window_left, kv_core_chunk_size,
                    active_aiv_cores);
            } else {
                finalize.reduce_cast_dq_colors(
                    color_bf16_src, dq_storage_elems, dq_dst,
                    cu_q_lens_gm, batch, num_q_heads, q_stride,
                    window_left, color_min_delta, color_max_delta,
                    active_aiv_cores);
            }
        } else if (use_dkv_color_partials) {
            if (use_bf16_dkv_color_partials) {
                finalize.reduce_cast_dkv_colors(
                    color_bf16_src, 0U, dk_storage_elems,
                    dk_dst, dv_dst, cu_q_lens_gm, cu_kv_lens_gm,
                    batch, num_kv_heads, kv_stride, window_left,
                    color_min_delta, color_max_delta,
                    direct_scaled_dq ? 1.0F : scale, active_aiv_cores);
            } else {
                finalize.reduce_cast_dkv_colors(
                    src, physical_workspace_elems + dkv_scratch_elems,
                    dk_storage_elems, dk_dst, dv_dst,
                    cu_q_lens_gm, cu_kv_lens_gm, batch,
                    num_kv_heads, kv_stride, window_left,
                    color_min_delta, color_max_delta,
                    direct_scaled_dq ? 1.0F : scale, active_aiv_cores);
            }
        }
        if (direct_scaled_dq && !use_kv_owned_mha) {
            finalize.cast_packed_q_tails(
                src, 0U, dq_dst, cu_q_lens_gm, batch, num_q_heads,
                q_stride, active_aiv_cores, use_compact_batch_dq);
        }
        uint64_t global_chunk_base = 0U;
        if (!direct_scaled_dq && !use_compact_chunk_dq) {
            finalize.cast_packed_range(
                src, 0U, dq_dst, cu_q_lens_gm, batch, q_stride,
                scale, active_aiv_cores, global_chunk_base);
        }
        if (use_kv_owned_mha ||
            (!use_dkv_color_partials && !direct_dkv_output)) {
            if (use_kv_owned_mha && kv_stride == DIM) {
                finalize.cast_packed_q_tails(
                    src, dq_storage_elems,
                    dk_dst, cu_kv_lens_gm, batch, num_kv_heads,
                    kv_stride, active_aiv_cores);
            } else {
                finalize.cast_packed_range(
                    src, dq_storage_elems, dk_dst, cu_kv_lens_gm,
                    batch, kv_stride,
                    direct_scaled_dq ? 1.0F : scale,
                    active_aiv_cores, global_chunk_base);
            }
            if (use_kv_owned_mha && kv_stride == DIM) {
                finalize.cast_packed_q_tails(
                    src, dq_storage_elems + dk_storage_elems,
                    dv_dst, cu_kv_lens_gm, batch, num_kv_heads,
                    kv_stride, active_aiv_cores);
            } else {
                finalize.cast_packed_range(
                    src, dq_storage_elems + dk_storage_elems,
                    dv_dst, cu_kv_lens_gm, batch, kv_stride,
                    1.0F, active_aiv_cores, global_chunk_base);
            }
        }
        if (use_global_dkv_partials) {
            finalize.reduce_cast_add_dkv_planes(
                src, global_dkv_partial_offset,
                global_dkv_plane_elems, global_dkv_partial_core_count,
                dq_storage_elems, dk_storage_elems, dk_dst, dv_dst,
                cu_q_lens_gm, cu_kv_lens_gm, batch, num_kv_heads,
                kv_stride, window_left, global_window_size,
                direct_scaled_dq ? 1.0F : scale,
                active_aiv_cores);
        }
        finalize.destroy();
    }
}
