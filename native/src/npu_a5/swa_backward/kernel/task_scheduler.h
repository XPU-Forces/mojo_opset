#pragma once

#include <cstdint>

#if defined(SWA_BWD_HOST_TEST)
#define SWA_BWD_INLINE inline
#else
#define SWA_BWD_INLINE __aicore__ inline
#endif

namespace SWA_v5::Backward {

struct KvTileRange {
    uint32_t first_tile;
    uint32_t second_tile;
    uint32_t local_tile_count;
    uint32_t tile_count;
};

static constexpr uint32_t BWD_MASK_LOCAL_ONLY = 0U;
static constexpr uint32_t BWD_MASK_GLOBAL_ONLY = 1U;

struct TaskInfo {
    uint32_t linear_task;
    uint32_t batch_idx;
    uint32_t kv_head;
    uint32_t gqa_group;
    uint32_t q_head;
    uint32_t q_tile;
    uint32_t kv_tile;
    uint32_t kv_tile_ordinal;
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t q_global_begin;
    uint32_t kv_global_begin;
    uint32_t q_rows;
    uint32_t kv_rows;
    uint32_t mask_mode;
};

struct QTaskGroup {
    TaskInfo q_task;
    KvTileRange kv_range;
};

SWA_BWD_INLINE uint32_t ceil_div(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1U) / divisor;
}

SWA_BWD_INLINE uint32_t min_u32(uint32_t lhs, uint32_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

// Kernel window_left is a span including the query; zero means disabled.
SWA_BWD_INLINE uint32_t local_window_radius(uint32_t window_left) {
    return window_left > 0U ? window_left - 1U : 0U;
}

template<uint32_t TILE_K>
SWA_BWD_INLINE uint32_t global_partial_rows(
    uint32_t q_len, uint32_t kv_len, uint32_t window_left,
    uint32_t global_window_size) {
    if (q_len == 0U || kv_len < q_len || global_window_size == 0U ||
        kv_len <= window_left) {
        return 0U;
    }
    const uint32_t gap_rows = min_u32(
        global_window_size, kv_len - window_left);
    return ceil_div(gap_rows, TILE_K) * TILE_K;
}

template<uint32_t TILE_K, class CuLens>
SWA_BWD_INLINE uint64_t global_partial_row_base(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch_end,
    uint32_t window_left, uint32_t global_window_size) {
    uint64_t rows = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch_end; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) -
            cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) -
            cu_kv_lens.GetValue(batch_idx);
        rows += global_partial_rows<TILE_K>(
            q_len, kv_len, window_left, global_window_size);
    }
    return rows;
}

SWA_BWD_INLINE uint32_t map_aabb_kv_head(uint32_t q_head, uint32_t num_q_heads, uint32_t num_kv_heads) {
    return min_u32((q_head * num_kv_heads) / num_q_heads, num_kv_heads - 1U);
}

template<uint32_t TILE_K>
SWA_BWD_INLINE uint32_t clamp_tile_end(int32_t valid_end, uint32_t kv_len) {
    if (valid_end <= 0) {
        return 0U;
    }
    const uint32_t end = ceil_div(static_cast<uint32_t>(valid_end), TILE_K);
    const uint32_t kv_tiles = ceil_div(kv_len, TILE_K);
    return min_u32(end, kv_tiles);
}

template<uint32_t TILE_K, bool IS_CAUSAL>
SWA_BWD_INLINE KvTileRange make_kv_tile_range(
    uint32_t q_abs_start, uint32_t q_rows, uint32_t kv_len,
    uint32_t window_left, uint32_t window_right, uint32_t global_window_size) {
    const int32_t q_start = static_cast<int32_t>(q_abs_start);
    const int32_t q_end = q_start + static_cast<int32_t>(q_rows);
    const int32_t valid_start = q_start - static_cast<int32_t>(window_left) + 1;
    int32_t valid_end = q_end;
    if constexpr (!IS_CAUSAL) {
        valid_end += static_cast<int32_t>(window_right);
    }

    const uint32_t local_start = window_left == 0U ? 0U :
        (valid_start <= 0 ? 0U : static_cast<uint32_t>(valid_start) / TILE_K);
    const uint32_t local_end =
        window_left == 0U ? 0U : clamp_tile_end<TILE_K>(valid_end, kv_len);
    const uint32_t local_count = local_end <= local_start ? 0U : local_end - local_start;

    uint32_t global_count = 0U;
    if (global_window_size > 0U) {
        if (kv_len > 0U) {
            const int32_t last_q = q_end - 1;
            const int32_t max_local_begin = last_q - static_cast<int32_t>(window_left) + 1;
            if (max_local_begin > 0) {
                const uint32_t global_valid_end = min_u32(
                    min_u32(global_window_size, kv_len),
                    static_cast<uint32_t>(max_local_begin));
                global_count = ceil_div(global_valid_end, TILE_K);
            }
        }
    }

    return KvTileRange{local_start, 0U, local_count, local_count + global_count};
}

SWA_BWD_INLINE uint32_t kv_tile_at(const KvTileRange& range, uint32_t ordinal) {
    return ordinal < range.local_tile_count ?
        range.first_tile + ordinal : range.second_tile + ordinal - range.local_tile_count;
}

SWA_BWD_INLINE uint32_t kv_mask_mode_at(const KvTileRange& range, uint32_t ordinal) {
    return ordinal < range.local_tile_count ? BWD_MASK_LOCAL_ONLY : BWD_MASK_GLOBAL_ONLY;
}

template<uint32_t TILE_K>
SWA_BWD_INLINE TaskInfo task_at(const QTaskGroup& group, uint32_t kv_ordinal) {
    TaskInfo task = group.q_task;
    task.kv_tile_ordinal = kv_ordinal;
    task.kv_tile = kv_tile_at(group.kv_range, kv_ordinal);
    task.kv_global_begin += task.kv_tile * TILE_K;
    task.kv_rows = min_u32(TILE_K, task.kv_len - task.kv_tile * TILE_K);
    task.mask_mode = kv_mask_mode_at(group.kv_range, kv_ordinal);
    return task;
}

template<uint32_t TILE_Q, class CuLens>
SWA_BWD_INLINE uint32_t count_q_groups(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads) {
    if (num_kv_heads == 0U || num_q_heads == 0U) {
        return 0U;
    }
    uint32_t total = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len = cu_q_lens.GetValue(batch_idx + 1U) - cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len = cu_kv_lens.GetValue(batch_idx + 1U) - cu_kv_lens.GetValue(batch_idx);
        if (q_len > 0U && kv_len >= q_len) {
            total += num_q_heads * ceil_div(q_len, TILE_Q);
        }
    }
    return total;
}

template<uint32_t TILE_K, class CuLens>
SWA_BWD_INLINE uint32_t count_kv_groups(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_kv_heads) {
    uint32_t total = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) -
            cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) -
            cu_kv_lens.GetValue(batch_idx);
        if (q_len > 0U && kv_len == q_len) {
            total += num_kv_heads * ceil_div(kv_len, TILE_K);
        }
    }
    return total;
}

template<uint32_t TILE_K, class CuLens>
SWA_BWD_INLINE bool fill_kv_group(
    uint32_t linear_group, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t num_kv_heads, TaskInfo& group) {
    uint32_t remaining = linear_group;
    uint32_t q_group_base = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_begin = cu_q_lens.GetValue(batch_idx);
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) - q_begin;
        const uint32_t kv_begin = cu_kv_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) - kv_begin;
        const uint32_t kv_tiles =
            q_len > 0U && kv_len == q_len ? ceil_div(kv_len, TILE_K) : 0U;
        const uint32_t batch_groups = num_kv_heads * kv_tiles;
        if (remaining >= batch_groups) {
            remaining -= batch_groups;
            q_group_base += num_kv_heads * ceil_div(q_len, TILE_K);
            continue;
        }
        if (kv_tiles == 0U) {
            return false;
        }
        const uint32_t kv_tile = remaining / num_kv_heads;
        const uint32_t kv_head = remaining - kv_tile * num_kv_heads;
        const uint32_t kv_row = kv_tile * TILE_K;
        group = {
            q_group_base,
            batch_idx,
            kv_head,
            0U,
            kv_head,
            0U,
            kv_tile,
            0U,
            q_len,
            kv_len,
            q_begin,
            kv_begin + kv_row,
            0U,
            min_u32(TILE_K, kv_len - kv_row),
            BWD_MASK_LOCAL_ONLY,
        };
        return true;
    }
    return false;
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL>
SWA_BWD_INLINE bool fill_kv_owned_task(
    const TaskInfo& group, uint32_t q_tile, uint32_t num_kv_heads,
    uint32_t window_left, uint32_t window_right, TaskInfo& task) {
    const uint32_t q_row = q_tile * TILE_Q;
    if (q_row >= group.q_len) {
        return false;
    }
    const uint32_t q_rows = min_u32(TILE_Q, group.q_len - q_row);
    const KvTileRange range =
        make_kv_tile_range<TILE_K, IS_CAUSAL>(
            q_row, q_rows, group.kv_len,
            window_left, window_right, 0U);
    if (group.kv_tile < range.first_tile ||
        group.kv_tile >= range.first_tile + range.local_tile_count) {
        return false;
    }
    task = group;
    task.linear_task =
        group.linear_task + q_tile * num_kv_heads + group.kv_head;
    task.q_tile = q_tile;
    task.kv_tile_ordinal = group.kv_tile - range.first_tile;
    task.q_global_begin += q_row;
    task.q_rows = q_rows;
    return true;
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL>
SWA_BWD_INLINE uint32_t count_head_tasks(
    uint32_t q_len, uint32_t kv_len, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size) {
    if (q_len == 0U || kv_len < q_len) {
        return 0U;
    }
    const uint32_t kv_prefix = kv_len - q_len;
    const uint32_t q_tiles = ceil_div(q_len, TILE_Q);
    uint32_t count = 0U;
    for (uint32_t q_tile = 0U; q_tile < q_tiles; ++q_tile) {
        const uint32_t q_begin = q_tile * TILE_Q;
        const uint32_t q_rows = min_u32(TILE_Q, q_len - q_begin);
        count += make_kv_tile_range<TILE_K, IS_CAUSAL>(
            kv_prefix + q_begin, q_rows, kv_len,
            window_left, window_right, global_window_size).tile_count;
    }
    return count;
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL, class CuLens>
SWA_BWD_INLINE uint32_t count_tasks(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size) {
    if (num_kv_heads == 0U || num_q_heads == 0U) {
        return 0U;
    }
    uint32_t total = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len = cu_q_lens.GetValue(batch_idx + 1U) - cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len = cu_kv_lens.GetValue(batch_idx + 1U) - cu_kv_lens.GetValue(batch_idx);
        total += num_q_heads *
            count_head_tasks<TILE_Q, TILE_K, IS_CAUSAL>(
                q_len, kv_len, window_left, window_right, global_window_size);
    }
    return total;
}

SWA_BWD_INLINE void map_head_linear(
    uint32_t head_linear, uint32_t num_q_heads, uint32_t num_kv_heads,
    bool gqa_layout_abab, uint32_t& kv_head, uint32_t& gqa_group, uint32_t& q_head) {
    if (gqa_layout_abab) {
        const uint32_t groups = ceil_div(num_q_heads, num_kv_heads);
        uint32_t ordinal = 0U;
        for (uint32_t kv = 0U; kv < num_kv_heads; ++kv) {
            for (uint32_t group = 0U; group < groups; ++group) {
                const uint32_t candidate_q_head = group * num_kv_heads + kv;
                if (candidate_q_head >= num_q_heads) {
                    continue;
                }
                if (ordinal == head_linear) {
                    kv_head = kv;
                    gqa_group = group;
                    q_head = candidate_q_head;
                    return;
                }
                ++ordinal;
            }
        }
        kv_head = 0U;
        gqa_group = 0U;
        q_head = 0U;
    } else {
        q_head = head_linear;
        kv_head = map_aabb_kv_head(head_linear, num_q_heads, num_kv_heads);
        gqa_group = head_linear;
    }
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL, class CuLens>
SWA_BWD_INLINE bool fill_task(uint32_t linear_task, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t window_left, uint32_t window_right, uint32_t global_window_size,
    bool gqa_layout_abab, TaskInfo& task) {
    if (num_kv_heads == 0U || num_q_heads == 0U) {
        return false;
    }

    uint32_t remaining = linear_task;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_global_begin = cu_q_lens.GetValue(batch_idx);
        const uint32_t q_end = cu_q_lens.GetValue(batch_idx + 1U);
        const uint32_t kv_global_begin = cu_kv_lens.GetValue(batch_idx);
        const uint32_t kv_end = cu_kv_lens.GetValue(batch_idx + 1U);
        const uint32_t q_len = q_end - q_global_begin;
        const uint32_t kv_len = kv_end - kv_global_begin;
        const uint32_t tasks_per_head =
            count_head_tasks<TILE_Q, TILE_K, IS_CAUSAL>(
                q_len, kv_len, window_left, window_right, global_window_size);
        const uint32_t batch_tasks = num_q_heads * tasks_per_head;
        if (remaining >= batch_tasks) {
            remaining -= batch_tasks;
            continue;
        }
        if (tasks_per_head == 0U) {
            return false;
        }

        const uint32_t head_linear = remaining / tasks_per_head;
        remaining -= head_linear * tasks_per_head;
        uint32_t kv_head = 0U;
        uint32_t gqa_group = 0U;
        uint32_t q_head = 0U;
        map_head_linear(
            head_linear, num_q_heads, num_kv_heads, gqa_layout_abab,
            kv_head, gqa_group, q_head);
        const uint32_t kv_prefix = kv_len - q_len;
        const uint32_t q_tiles = ceil_div(q_len, TILE_Q);
        for (uint32_t q_tile = 0U; q_tile < q_tiles; ++q_tile) {
            const uint32_t q_begin = q_tile * TILE_Q;
            const uint32_t q_rows = min_u32(TILE_Q, q_len - q_begin);
            const KvTileRange range =
                make_kv_tile_range<TILE_K, IS_CAUSAL>(
                    kv_prefix + q_begin, q_rows, kv_len,
                    window_left, window_right, global_window_size);
            if (remaining >= range.tile_count) {
                remaining -= range.tile_count;
                continue;
            }

            const uint32_t kv_tile = kv_tile_at(range, remaining);
            const uint32_t kv_begin = kv_tile * TILE_K;
            task = {
                linear_task,
                batch_idx,
                kv_head,
                gqa_group,
                q_head,
                q_tile,
                kv_tile,
                remaining,
                q_len,
                kv_len,
                q_global_begin + q_begin,
                kv_global_begin + kv_begin,
                q_rows,
                min_u32(TILE_K, kv_len - kv_begin),
                kv_mask_mode_at(range, remaining),
            };
            return true;
        }
        return false;
    }
    return false;
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL, class CuLens>
SWA_BWD_INLINE bool fill_q_group(uint32_t linear_group, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t window_left, uint32_t window_right, uint32_t global_window_size,
    bool gqa_layout_abab, QTaskGroup& group) {
    if (num_kv_heads == 0U || num_q_heads == 0U) {
        return false;
    }

    uint32_t remaining = linear_group;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_global_begin = cu_q_lens.GetValue(batch_idx);
        const uint32_t q_end = cu_q_lens.GetValue(batch_idx + 1U);
        const uint32_t kv_global_begin = cu_kv_lens.GetValue(batch_idx);
        const uint32_t kv_end = cu_kv_lens.GetValue(batch_idx + 1U);
        const uint32_t q_len = q_end - q_global_begin;
        const uint32_t kv_len = kv_end - kv_global_begin;
        if (q_len == 0U || kv_len < q_len) {
            continue;
        }
        const uint32_t q_tiles = ceil_div(q_len, TILE_Q);
        const uint32_t batch_groups = num_q_heads * q_tiles;
        if (remaining >= batch_groups) {
            remaining -= batch_groups;
            continue;
        }

        // Keep each scheduling wave within nearby KV windows so GQA heads can
        // reuse the same K/V region through L2.
        const uint32_t q_tile = remaining / num_q_heads;
        const uint32_t head_linear = remaining - q_tile * num_q_heads;
        uint32_t kv_head = 0U;
        uint32_t gqa_group = 0U;
        uint32_t q_head = 0U;
        map_head_linear(
            head_linear, num_q_heads, num_kv_heads, gqa_layout_abab,
            kv_head, gqa_group, q_head);
        const uint32_t q_begin = q_tile * TILE_Q;
        const uint32_t q_rows = min_u32(TILE_Q, q_len - q_begin);
        const uint32_t kv_prefix = kv_len - q_len;
        const KvTileRange range =
            make_kv_tile_range<TILE_K, IS_CAUSAL>(
                kv_prefix + q_begin, q_rows, kv_len,
                window_left, window_right, global_window_size);
        group = {
            {
                linear_group,
                batch_idx,
                kv_head,
                gqa_group,
                q_head,
                q_tile,
                0U,
                0U,
                q_len,
                kv_len,
                q_global_begin + q_begin,
                kv_global_begin,
                q_rows,
                0U,
            },
            range,
        };
        return range.tile_count > 0U;
    }
    return false;
}

template<uint32_t TILE_Q, class CuLens>
SWA_BWD_INLINE uint32_t count_local_round_candidates(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_kv_heads) {
    uint32_t total = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) - cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) - cu_kv_lens.GetValue(batch_idx);
        if (q_len > 0U && kv_len >= q_len) {
            total += num_kv_heads * ceil_div(q_len, TILE_Q);
        }
    }
    return total;
}

template<uint32_t TILE_Q, class CuLens>
SWA_BWD_INLINE void local_round_candidate_range(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch_idx,
    uint32_t num_kv_heads, uint32_t& candidate_begin,
    uint32_t& candidate_count) {
    candidate_begin = 0U;
    candidate_count = 0U;
    for (uint32_t idx = 0U; idx <= batch_idx; ++idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(idx + 1U) - cu_q_lens.GetValue(idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(idx + 1U) - cu_kv_lens.GetValue(idx);
        const uint32_t count = q_len > 0U && kv_len >= q_len ?
            num_kv_heads * ceil_div(q_len, TILE_Q) : 0U;
        if (idx == batch_idx) {
            candidate_count = count;
        } else {
            candidate_begin += count;
        }
    }
}

template<uint32_t TILE_Q, class CuLens>
SWA_BWD_INLINE void local_round_candidate_range_span(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch_begin,
    uint32_t batch_end, uint32_t num_kv_heads,
    uint32_t& candidate_begin, uint32_t& candidate_count) {
    candidate_begin = 0U;
    candidate_count = 0U;
    for (uint32_t idx = 0U; idx < batch_end; ++idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(idx + 1U) - cu_q_lens.GetValue(idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(idx + 1U) - cu_kv_lens.GetValue(idx);
        const uint32_t count = q_len > 0U && kv_len >= q_len ?
            num_kv_heads * ceil_div(q_len, TILE_Q) : 0U;
        if (idx < batch_begin) {
            candidate_begin += count;
        } else {
            candidate_count += count;
        }
    }
}

template<uint32_t TILE_Q, uint32_t TILE_K>
SWA_BWD_INLINE void local_round_chunk_candidate_range(
    uint32_t q_len, uint32_t kv_len, uint32_t num_kv_heads,
    uint32_t chunk_kv_begin, uint32_t chunk_kv_rows,
    int32_t min_delta, int32_t max_delta,
    uint32_t& candidate_offset, uint32_t& candidate_count) {
    candidate_offset = 0U;
    candidate_count = 0U;
    if (q_len == 0U || kv_len < q_len || num_kv_heads == 0U ||
        chunk_kv_rows == 0U) {
        return;
    }

    const int64_t chunk_tile_begin =
        static_cast<int64_t>(chunk_kv_begin / TILE_K);
    const int64_t chunk_tile_end = static_cast<int64_t>(
        ceil_div(chunk_kv_begin + chunk_kv_rows, TILE_K));
    const int64_t anchor_begin =
        chunk_tile_begin + static_cast<int64_t>(min_delta);
    const int64_t anchor_end =
        chunk_tile_end + static_cast<int64_t>(max_delta);
    const int64_t kv_prefix = static_cast<int64_t>(kv_len - q_len);
    const int64_t q_begin_min =
        anchor_begin * static_cast<int64_t>(TILE_K) - kv_prefix;
    const int64_t q_begin_end =
        anchor_end * static_cast<int64_t>(TILE_K) - kv_prefix;
    const uint32_t q_tiles = ceil_div(q_len, TILE_Q);
    const uint32_t first_q_tile = q_begin_min <= 0 ?
        0U : min_u32(
            static_cast<uint32_t>(
                (q_begin_min + static_cast<int64_t>(TILE_Q) - 1) /
                static_cast<int64_t>(TILE_Q)),
            q_tiles);
    const uint32_t last_q_tile = q_begin_end <= 0 ?
        0U : min_u32(
            static_cast<uint32_t>(
                (q_begin_end + static_cast<int64_t>(TILE_Q) - 1) /
                static_cast<int64_t>(TILE_Q)),
            q_tiles);
    if (last_q_tile <= first_q_tile) {
        return;
    }
    candidate_offset = first_q_tile * num_kv_heads;
    candidate_count =
        (last_q_tile - first_q_tile) * num_kv_heads;
}

template<uint32_t TILE_K, class CuLens>
SWA_BWD_INLINE uint32_t count_packed_batch_color_waves(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t capacity_rows) {
    uint32_t waves = 0U;
    uint32_t used_rows = 0U;
    for (uint32_t idx = 0U; idx < batch; ++idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(idx + 1U) - cu_q_lens.GetValue(idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(idx + 1U) - cu_kv_lens.GetValue(idx);
        const uint32_t rows = q_len > 0U && kv_len >= q_len ?
            kv_len + TILE_K - 1U : 0U;
        if (rows == 0U) {
            continue;
        }
        if (used_rows == 0U || used_rows + rows > capacity_rows) {
            ++waves;
            used_rows = rows;
        } else {
            used_rows += rows;
        }
    }
    return waves;
}

template<uint32_t TILE_K, class CuLens>
SWA_BWD_INLINE bool decode_packed_batch_color_wave(
    uint32_t wave, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t capacity_rows,
    uint32_t& batch_begin, uint32_t& batch_end) {
    uint32_t current_wave = 0U;
    uint32_t used_rows = 0U;
    batch_begin = 0U;
    batch_end = 0U;
    for (uint32_t idx = 0U; idx < batch; ++idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(idx + 1U) - cu_q_lens.GetValue(idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(idx + 1U) - cu_kv_lens.GetValue(idx);
        const uint32_t rows = q_len > 0U && kv_len >= q_len ?
            kv_len + TILE_K - 1U : 0U;
        if (rows == 0U) {
            continue;
        }
        if (used_rows == 0U || used_rows + rows > capacity_rows) {
            if (used_rows != 0U) {
                if (current_wave == wave) {
                    batch_end = idx;
                    return true;
                }
                ++current_wave;
            }
            batch_begin = idx;
            used_rows = rows;
        } else {
            used_rows += rows;
        }
    }
    if (used_rows != 0U && current_wave == wave) {
        batch_end = batch;
        return true;
    }
    return false;
}

SWA_BWD_INLINE bool map_local_candidate_segment(
    uint32_t logical_core, uint32_t ordinal, uint32_t candidate_count,
    uint32_t core_count, uint32_t& local_candidate) {
    if (core_count == 0U || logical_core >= core_count) {
        return false;
    }
    if (candidate_count <= core_count) {
        if (ordinal != 0U || logical_core >= candidate_count) {
            return false;
        }
        local_candidate = logical_core;
        return true;
    }
    if (candidate_count <= 2U * core_count) {
        const uint32_t paired_cores = candidate_count - core_count;
        if (logical_core < paired_cores) {
            if (ordinal >= 2U) {
                return false;
            }
            local_candidate = ordinal == 0U ?
                logical_core : 2U * paired_cores - 1U - logical_core;
            return true;
        }
        if (ordinal != 0U) {
            return false;
        }
        local_candidate = logical_core + paired_cores;
        return true;
    }

    const uint32_t segment_begin = ordinal * core_count;
    if (segment_begin >= candidate_count) {
        return false;
    }
    const uint32_t segment_count =
        min_u32(core_count, candidate_count - segment_begin);
    const uint32_t tail_count = candidate_count % core_count;
    const uint32_t segment_rotation =
        candidate_count > 3U * core_count && tail_count > 0U ?
            core_count - tail_count : 0U;
    const uint32_t segment_core =
        (logical_core + ordinal * segment_rotation) % core_count;
    if (segment_core >= segment_count) {
        return false;
    }
    local_candidate = (ordinal & 1U) == 0U ?
        segment_begin + segment_core :
        segment_begin + segment_count - 1U - segment_core;
    return true;
}

SWA_BWD_INLINE bool scheduled_local_round_candidate(
    uint32_t core_idx, uint32_t ordinal, uint32_t candidate_count,
    uint32_t core_count, bool fold_boundary, uint32_t& linear_candidate) {
    if (core_count == 0U || core_idx >= core_count) {
        return false;
    }
    if (fold_boundary) {
        return map_local_candidate_segment(
            core_idx, ordinal, candidate_count, core_count, linear_candidate);
    }
    linear_candidate = core_idx + ordinal * core_count;
    return linear_candidate < candidate_count;
}

SWA_BWD_INLINE bool use_tail_candidate_collaboration(
    uint32_t candidate_count, uint32_t core_count) {
    if (core_count == 0U || candidate_count <= 2U * core_count) {
        return false;
    }
    const uint32_t tail_count = candidate_count - 2U * core_count;
    return tail_count <= core_count / 2U;
}

SWA_BWD_INLINE bool scheduled_local_round_collaborative_candidate(
    uint32_t core_idx, uint32_t ordinal, uint32_t candidate_count,
    uint32_t core_count, bool fold_boundary, bool collaborate_tail,
    uint32_t& linear_candidate, bool& helper_task,
    bool& tail_owner_task) {
    helper_task = false;
    tail_owner_task = false;
    uint32_t base_count = 0U;
    uint32_t base_candidate = 0U;
    while (scheduled_local_round_candidate(
            core_idx, base_count, candidate_count, core_count,
            fold_boundary, base_candidate)) {
        if (ordinal == base_count) {
            linear_candidate = base_candidate;
            tail_owner_task =
                collaborate_tail && linear_candidate >= 2U * core_count;
            return true;
        }
        ++base_count;
    }
    if (!collaborate_tail || ordinal != base_count) {
        return false;
    }
    const uint32_t tail_count = candidate_count - 2U * core_count;
    if (core_idx < tail_count || core_idx >= 2U * tail_count) {
        return false;
    }
    linear_candidate = 2U * core_count + core_idx - tail_count;
    helper_task = true;
    return true;
}

SWA_BWD_INLINE uint32_t count_local_round_candidates_for_core(
    uint32_t core_idx, uint32_t candidate_count, uint32_t core_count,
    bool fold_boundary) {
    uint32_t count = 0U;
    uint32_t linear_candidate = 0U;
    while (scheduled_local_round_candidate(
            core_idx, count, candidate_count, core_count,
            fold_boundary, linear_candidate)) {
        ++count;
    }
    return count;
}

template<class CuLens>
SWA_BWD_INLINE uint32_t count_local_color_chunk_waves(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t chunk_rows) {
    if (chunk_rows == 0U) {
        return 0U;
    }
    uint32_t waves = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) -
            cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) -
            cu_kv_lens.GetValue(batch_idx);
        if (kv_len > 0U && kv_len >= q_len) {
            waves += ceil_div(kv_len, chunk_rows);
        }
    }
    return waves;
}

template<class CuLens>
SWA_BWD_INLINE bool decode_local_color_chunk_wave(
    uint32_t wave, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t chunk_rows,
    uint32_t& batch_idx, uint32_t& chunk_idx) {
    if (chunk_rows == 0U) {
        return false;
    }
    uint32_t remaining = wave;
    for (batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) -
            cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) -
            cu_kv_lens.GetValue(batch_idx);
        const uint32_t batch_waves =
            kv_len > 0U && kv_len >= q_len ?
                ceil_div(kv_len, chunk_rows) : 0U;
        if (remaining < batch_waves) {
            chunk_idx = remaining;
            return true;
        }
        remaining -= batch_waves;
    }
    return false;
}

template<uint32_t TILE_Q, class CuLens>
SWA_BWD_INLINE uint32_t count_q_tile_waves(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch) {
    uint32_t waves = 0U;
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) -
            cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) -
            cu_kv_lens.GetValue(batch_idx);
        if (q_len > 0U && kv_len >= q_len) {
            waves += ceil_div(q_len, TILE_Q);
        }
    }
    return waves;
}

template<uint32_t TILE_Q, class CuLens>
SWA_BWD_INLINE bool decode_q_tile_wave(
    uint32_t wave, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch, uint32_t& batch_idx, uint32_t& q_tile) {
    uint32_t remaining = wave;
    for (batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) -
            cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) -
            cu_kv_lens.GetValue(batch_idx);
        const uint32_t batch_waves =
            q_len > 0U && kv_len >= q_len ?
                ceil_div(q_len, TILE_Q) : 0U;
        if (remaining < batch_waves) {
            q_tile = remaining;
            return true;
        }
        remaining -= batch_waves;
    }
    return false;
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL, class CuLens>
SWA_BWD_INLINE bool fill_q_tile_wave_group(
    uint32_t batch_idx, uint32_t q_tile, uint32_t kv_head,
    uint32_t head_in_group, CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
    uint32_t window_right, uint32_t global_window_size, bool global_only,
    bool gqa_layout_abab, QTaskGroup& group) {
    if (num_kv_heads == 0U || num_q_heads == 0U ||
        num_q_heads % num_kv_heads != 0U || kv_head >= num_kv_heads) {
        return false;
    }
    const uint32_t gqa_ratio = num_q_heads / num_kv_heads;
    if (head_in_group >= gqa_ratio) {
        return false;
    }
    const uint32_t q_global_begin = cu_q_lens.GetValue(batch_idx);
    const uint32_t q_len =
        cu_q_lens.GetValue(batch_idx + 1U) - q_global_begin;
    const uint32_t kv_global_begin = cu_kv_lens.GetValue(batch_idx);
    const uint32_t kv_len =
        cu_kv_lens.GetValue(batch_idx + 1U) - kv_global_begin;
    const uint32_t q_begin = q_tile * TILE_Q;
    if (q_begin >= q_len || kv_len < q_len || kv_len == 0U) {
        return false;
    }
    const uint32_t q_rows = min_u32(TILE_Q, q_len - q_begin);
    const uint32_t q_head = gqa_layout_abab ?
        head_in_group * num_kv_heads + kv_head :
        kv_head * gqa_ratio + head_in_group;
    const uint32_t kv_prefix = kv_len - q_len;
    KvTileRange range =
        make_kv_tile_range<TILE_K, IS_CAUSAL>(
            kv_prefix + q_begin, q_rows, kv_len,
            window_left, window_right, 0U);
    if (global_only) {
        const KvTileRange combined_range =
            make_kv_tile_range<TILE_K, IS_CAUSAL>(
                kv_prefix + q_begin, q_rows, kv_len,
                window_left, window_right, global_window_size);
        range = {
            0U,
            0U,
            0U,
            combined_range.tile_count - combined_range.local_tile_count,
        };
    }
    group = {
        {
            q_tile * num_kv_heads + kv_head,
            batch_idx,
            kv_head,
            head_in_group,
            q_head,
            q_tile,
            0U,
            0U,
            q_len,
            kv_len,
            q_global_begin + q_begin,
            kv_global_begin,
            q_rows,
            0U,
            global_only ? BWD_MASK_GLOBAL_ONLY : BWD_MASK_LOCAL_ONLY,
        },
        range,
    };
    return range.tile_count > 0U;
}

template<uint32_t TILE_Q, uint32_t TILE_K, class CuLens>
SWA_BWD_INLINE int32_t local_round_min_delta(
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch) {
    for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
        const uint32_t q_len =
            cu_q_lens.GetValue(batch_idx + 1U) - cu_q_lens.GetValue(batch_idx);
        const uint32_t kv_len =
            cu_kv_lens.GetValue(batch_idx + 1U) - cu_kv_lens.GetValue(batch_idx);
        if (q_len > 0U && kv_len >= q_len &&
            (kv_len - q_len) % TILE_K != 0U) {
            return -1;
        }
    }
    return 0;
}

template<uint32_t TILE_K>
SWA_BWD_INLINE int32_t local_round_max_delta(uint32_t window_left) {
    return static_cast<int32_t>(ceil_div(local_window_radius(window_left), TILE_K));
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL, class CuLens>
SWA_BWD_INLINE bool fill_local_round_task_span(
    uint32_t local_candidate, uint32_t candidate_base,
    int32_t delta, uint32_t head_in_group,
    CuLens& cu_q_lens, CuLens& cu_kv_lens,
    uint32_t batch_begin, uint32_t batch_end,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
    uint32_t window_right, bool gqa_layout_abab, uint32_t uniform_q_len,
    uint32_t uniform_kv_len, TaskInfo& task) {
    if (num_kv_heads == 0U || num_q_heads == 0U ||
        num_q_heads % num_kv_heads != 0U) {
        return false;
    }
    const uint32_t gqa_ratio = num_q_heads / num_kv_heads;
    if (head_in_group >= gqa_ratio) {
        return false;
    }

    uint32_t batch_idx = batch_begin;
    uint32_t q_global_begin = 0U;
    uint32_t kv_global_begin = 0U;
    uint32_t q_len = 0U;
    uint32_t kv_len = 0U;
    uint32_t remaining = local_candidate;
    if (uniform_q_len > 0U) {
        const uint32_t q_tiles = ceil_div(uniform_q_len, TILE_Q);
        const uint32_t batch_candidates = num_kv_heads * q_tiles;
        if (batch_candidates == 0U ||
            local_candidate >=
                (batch_end - batch_begin) * batch_candidates) {
            return false;
        }
        batch_idx =
            batch_begin + local_candidate / batch_candidates;
        remaining =
            local_candidate -
            (batch_idx - batch_begin) * batch_candidates;
        q_global_begin = batch_idx * uniform_q_len;
        kv_global_begin = batch_idx * uniform_kv_len;
        q_len = uniform_q_len;
        kv_len = uniform_kv_len;
    } else {
        bool found = false;
        for (; batch_idx < batch_end; ++batch_idx) {
            q_global_begin = cu_q_lens.GetValue(batch_idx);
            const uint32_t q_end = cu_q_lens.GetValue(batch_idx + 1U);
            kv_global_begin = cu_kv_lens.GetValue(batch_idx);
            const uint32_t kv_end = cu_kv_lens.GetValue(batch_idx + 1U);
            q_len = q_end - q_global_begin;
            kv_len = kv_end - kv_global_begin;
            if (q_len == 0U || kv_len < q_len) {
                continue;
            }
            const uint32_t q_tiles = ceil_div(q_len, TILE_Q);
            const uint32_t batch_candidates = num_kv_heads * q_tiles;
            if (remaining >= batch_candidates) {
                remaining -= batch_candidates;
                continue;
            }
            found = true;
            break;
        }
        if (!found) {
            return false;
        }
    }

    const uint32_t q_tile = remaining / num_kv_heads;
        const uint32_t kv_head = remaining - q_tile * num_kv_heads;
        const uint32_t q_head = gqa_layout_abab ?
            head_in_group * num_kv_heads + kv_head :
            kv_head * gqa_ratio + head_in_group;
        const uint32_t q_begin = q_tile * TILE_Q;
        const uint32_t q_rows = min_u32(TILE_Q, q_len - q_begin);
        const uint32_t kv_prefix = kv_len - q_len;
        const uint32_t q_abs_start = kv_prefix + q_begin;
        const int32_t q_anchor =
            static_cast<int32_t>(q_abs_start / TILE_K);
        const int32_t kv_tile_signed = q_anchor - delta;
        if (kv_tile_signed < 0) {
            return false;
        }

        const KvTileRange range =
            make_kv_tile_range<TILE_K, IS_CAUSAL>(
                q_abs_start, q_rows, kv_len, window_left, window_right, 0U);
        const uint32_t kv_tile = static_cast<uint32_t>(kv_tile_signed);
        if (kv_tile < range.first_tile ||
            kv_tile >= range.first_tile + range.local_tile_count) {
            return false;
        }
        const uint32_t kv_begin = kv_tile * TILE_K;
        task = {
            candidate_base + local_candidate,
            batch_idx,
            kv_head,
            head_in_group,
            q_head,
            q_tile,
            kv_tile,
            kv_tile - range.first_tile,
            q_len,
            kv_len,
            q_global_begin + q_begin,
            kv_global_begin + kv_begin,
            q_rows,
            min_u32(TILE_K, kv_len - kv_begin),
            BWD_MASK_LOCAL_ONLY,
        };
        return true;
}

template<uint32_t TILE_Q, uint32_t TILE_K, bool IS_CAUSAL, class CuLens>
SWA_BWD_INLINE bool fill_local_round_task(
    uint32_t linear_candidate, int32_t delta, uint32_t head_in_group,
    CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t window_left,
    uint32_t window_right, bool gqa_layout_abab, uint32_t uniform_q_len,
    uint32_t uniform_kv_len, TaskInfo& task) {
    return fill_local_round_task_span<TILE_Q, TILE_K, IS_CAUSAL>(
        linear_candidate, 0U, delta, head_in_group,
        cu_q_lens, cu_kv_lens, 0U, batch,
        num_q_heads, num_kv_heads, window_left, window_right,
        gqa_layout_abab, uniform_q_len, uniform_kv_len, task);
}

}  // namespace SWA_v5::Backward

#undef SWA_BWD_INLINE
