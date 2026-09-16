#pragma once

#include "kernel_operator.h"

namespace SWA_v5::Pipeline::FullPath {

struct L0TileRange {
    uint32_t start_l0_tile;
    uint32_t second_l0_tile;
    uint32_t last_l0_tile;
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

    template<uint32_t TILE_L0, uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT, uint32_t GLOBAL_WINDOW_SIZE,
        bool IS_CAUSAL = true>
    __aicore__ static inline L0TileRange make_l0_tile_range(
        uint32_t q_abs_start, uint32_t q_rows, uint32_t kv_len) {
        const int32_t q_start = static_cast<int32_t>(q_abs_start);
        const int32_t q_end = q_start + static_cast<int32_t>(q_rows);
        const int32_t valid_start = q_start - static_cast<int32_t>(WINDOW_LEFT);
        int32_t valid_end = q_end;
        if constexpr (!IS_CAUSAL && WINDOW_RIGHT > 0U) {
            valid_end += static_cast<int32_t>(WINDOW_RIGHT);
        }
        const uint32_t local_start_l0 = valid_start <= 0 ? 0U : static_cast<uint32_t>(valid_start) / TILE_L0;
        const uint32_t local_end_l0 = clamp_l0_end<TILE_L0>(valid_end, kv_len);
        const bool has_global_l0 = GLOBAL_WINDOW_SIZE > 0U && kv_len > 0U;

        if (!has_global_l0) {
            if (local_end_l0 <= local_start_l0) {
                return {local_start_l0, local_start_l0, local_start_l0, 0U};
            }
            return {
                local_start_l0,
                local_start_l0 + 1U,
                local_end_l0 - 1U,
                local_end_l0 - local_start_l0,
            };
        }
        if (local_end_l0 <= local_start_l0) {
            return {0U, 0U, 0U, 1U};
        }
        if (local_start_l0 <= 1U) {
            return {0U, 1U, local_end_l0 - 1U, local_end_l0};
        }
        return {0U, local_start_l0, local_end_l0 - 1U, 1U + local_end_l0 - local_start_l0};
    }

    __aicore__ static inline bool is_l0_range_contiguous(
        uint32_t start_l0_tile, uint32_t second_l0_tile, uint32_t valid_l0_count) {
        return valid_l0_count <= 1U || second_l0_tile == start_l0_tile + 1U;
    }

    __aicore__ static inline uint32_t l0_tile_at(
        uint32_t start_l0_tile, uint32_t second_l0_tile, uint32_t ordinal) {
        return ordinal == 0U ? start_l0_tile : second_l0_tile + ordinal - 1U;
    }

    template<uint32_t TILE_L0, uint32_t LOCAL_ROWS, uint32_t WINDOW_LEFT, uint32_t WINDOW_RIGHT,
        bool IS_CAUSAL = true>
    __aicore__ static inline bool is_l0_tile_all_valid(
        uint32_t q0_abs, uint32_t k0_local, uint32_t active_rows, uint32_t kv_len) {
        uint32_t valid_end = q0_abs;
        if constexpr (!IS_CAUSAL && WINDOW_RIGHT > 0U) {
            valid_end += WINDOW_RIGHT;
        }
        return active_rows == LOCAL_ROWS && k0_local + TILE_L0 <= kv_len &&
            k0_local >= (q0_abs + LOCAL_ROWS - 1U > WINDOW_LEFT ? q0_abs + LOCAL_ROWS - 1U - WINDOW_LEFT : 0U) &&
            k0_local + TILE_L0 - 1U <= valid_end;
    }
};

}  // namespace SWA_v5::Pipeline::FullPath
