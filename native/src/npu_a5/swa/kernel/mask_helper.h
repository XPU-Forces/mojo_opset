#pragma once

#include "kernel_operator.h"

namespace SWA_v5::Pipeline::FullPath {

struct L0TileRange {
    uint32_t start_l0_tile;
    uint32_t second_l0_tile;
    uint32_t last_l0_tile;
    uint32_t first_l0_count;
    uint32_t valid_l0_count;
};

class MaskHelper {
public:
    __aicore__ static inline uint32_t ceil_div(uint32_t a, uint32_t b) {
        return (a + b - 1U) / b;
    }

    template<uint32_t TILE_L0>
    __aicore__ static inline uint32_t clamp_l0_end(int32_t valid_end, uint32_t kv_len) {
        if (valid_end <= 0) {
            return 0U;
        }
        const uint32_t end = ceil_div(static_cast<uint32_t>(valid_end), TILE_L0);
        const uint32_t key_tiles = ceil_div(kv_len, TILE_L0);
        return end > key_tiles ? key_tiles : end;
    }

    template<uint32_t TILE_L0, bool IS_CAUSAL = true>
    __aicore__ static inline L0TileRange make_l0_tile_range(
        uint32_t q_abs_start, uint32_t q_rows, uint32_t kv_len,
        uint32_t window_left, uint32_t window_right, uint32_t global_window_size) {
        const int32_t q_start = static_cast<int32_t>(q_abs_start);
        const int32_t q_end = q_start + static_cast<int32_t>(q_rows);
        // window_left is the local span, including the query; zero disables it.
        const int32_t valid_start = q_start - static_cast<int32_t>(window_left) + 1;
        int32_t valid_end = q_end;
        if constexpr (!IS_CAUSAL) {
            valid_end += static_cast<int32_t>(window_right);
        }
        const uint32_t local_start_l0 =
            window_left == 0U ? 0U :
            (valid_start <= 0 ? 0U : static_cast<uint32_t>(valid_start) / TILE_L0);
        const uint32_t local_end_l0 =
            window_left == 0U ? 0U : clamp_l0_end<TILE_L0>(valid_end, kv_len);
        const uint32_t local_count = local_end_l0 > local_start_l0 ?
            local_end_l0 - local_start_l0 : 0U;
        const uint32_t global_valid_end = valid_end <= 0 ? 0U :
            static_cast<uint32_t>(valid_end);
        const uint32_t global_end = global_window_size < kv_len ?
            global_window_size : kv_len;
        const uint32_t causal_global_end = global_end < global_valid_end ?
            global_end : global_valid_end;
        const uint32_t global_end_l0 =
            causal_global_end == 0U ? 0U : ceil_div(causal_global_end, TILE_L0);

        if (global_end_l0 == 0U) {
            return {
                local_start_l0,
                local_end_l0,
                local_count == 0U ? local_start_l0 : local_end_l0 - 1U,
                local_count,
                local_count,
            };
        }
        if (local_count == 0U) {
            return {0U, global_end_l0, global_end_l0 - 1U,
                global_end_l0, global_end_l0};
        }
        if (global_end_l0 >= local_start_l0) {
            const uint32_t union_end = global_end_l0 > local_end_l0 ?
                global_end_l0 : local_end_l0;
            return {0U, union_end, union_end - 1U, union_end, union_end};
        }
        return {0U, local_start_l0, local_end_l0 - 1U,
            global_end_l0, global_end_l0 + local_count};
    }

    __aicore__ static inline bool is_l0_range_contiguous(
        uint32_t start_l0_tile, uint32_t second_l0_tile,
        uint32_t first_l0_count, uint32_t valid_l0_count) {
        return valid_l0_count <= first_l0_count ||
            second_l0_tile == start_l0_tile + first_l0_count;
    }

    __aicore__ static inline uint32_t l0_tile_at(
        uint32_t start_l0_tile, uint32_t second_l0_tile,
        uint32_t first_l0_count, uint32_t ordinal) {
        return ordinal < first_l0_count ? start_l0_tile + ordinal :
            second_l0_tile + ordinal - first_l0_count;
    }

    template<uint32_t TILE_L0, uint32_t LOCAL_ROWS, bool IS_CAUSAL = true>
    __aicore__ static inline bool is_l0_tile_all_valid(
        uint32_t q0_abs, uint32_t k0_local, uint32_t active_rows, uint32_t kv_len,
        uint32_t window_left, uint32_t window_right) {
        if (window_left == 0U) {
            return false;
        }
        uint32_t valid_end = q0_abs;
        if constexpr (!IS_CAUSAL) {
            valid_end += window_right;
        }
        return active_rows == LOCAL_ROWS && k0_local + TILE_L0 <= kv_len &&
            k0_local >= (q0_abs + LOCAL_ROWS > window_left ?
                q0_abs + LOCAL_ROWS - window_left : 0U) &&
            k0_local + TILE_L0 - 1U <= valid_end;
    }
};

}  // namespace SWA_v5::Pipeline::FullPath
