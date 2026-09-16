#pragma once

#include "kernel/compute.h"
#include "kernel/copy.h"
#include "kernel/runtime.cpp"

namespace SWA_v5::Pipeline::FullPath {

using SWA_v5::GMTensor_ND;
using SWA_v5::UBTensor_ND;

static constexpr uint32_t CUBE2_QK_READY_BASE = 0;
static constexpr uint32_t CUBE2_QK_CONSUMED_BASE = 2;
static constexpr uint32_t CUBE2_P_READY_BASE = 4;
static constexpr uint32_t CUBE2_BMM2_READY_BASE = 7;
static constexpr uint32_t CUBE2_BMM2_CONSUMED_BASE = 9;
static constexpr uint32_t CUBE2_P_SLOTS = 3;
static constexpr uint32_t CUBE2_MM_SLOTS = 2;

#ifndef SWA_V5_CUBE2_SKIP_P_READY_WAIT
#define SWA_V5_CUBE2_SKIP_P_READY_WAIT 0
#endif

__aicore__ inline uint16_t event_slot(uint16_t base, uint32_t slot) {
    return static_cast<uint16_t>(base + slot);
}

struct AicQTaskScalars {
    uint32_t linear_q_tile;
    uint32_t q_offset;
    uint32_t kv_offset;
    uint32_t q_rows;
    uint32_t kv_len;
    uint32_t start_l0_tile;
    uint32_t second_l0_tile;
    uint32_t valid_l0_count;
};

struct AicL0TileScalars {
    uint32_t local_l0_idx;
    uint32_t global_l0_idx;
    uint32_t global_l1_idx;
    uint32_t local_l1_idx;
    uint32_t part_idx;
    uint32_t num_parts;
};

class FullPathAIC {
    static constexpr uint32_t K_L1_SLOTS = 2U;
    static constexpr uint32_t V_L1_SLOTS = 2U;

    __aicore__ inline uint32_t base_l0_tile(const AicQTaskScalars& q_task,
        const AicL0TileScalars& tile) {
        const uint32_t current_l0_tile = tile.local_l0_idx == 0U ?
            q_task.start_l0_tile : q_task.second_l0_tile + tile.local_l0_idx - 1U;
        return current_l0_tile - tile.part_idx;
    }

    template<uint32_t TILE_Q, uint32_t TILE_L0, uint32_t TILE_L1, uint32_t DIM, class InputT>
    __aicore__ inline void prefetch_k(AICore_RT<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>& rt,
        GMTensor_ND<InputT>& k_gm, uint32_t k_stride, const AicQTaskScalars& q_task,
        const AicL0TileScalars& tile) {
        const uint32_t tile_k_begin = base_l0_tile(q_task, tile) * TILE_L0;
        const uint32_t tile_k_offset = q_task.kv_offset + tile_k_begin * k_stride;
        const uint32_t max_copy_rows = tile.num_parts * TILE_L0;
        const uint32_t remaining_rows = q_task.kv_len > tile_k_begin ? q_task.kv_len - tile_k_begin : 0U;
        const uint32_t copy_rows = remaining_rows < max_copy_rows ? remaining_rows : max_copy_rows;
        SWA_v5::Compute::copy_sync(
            rt.cube.K_B1[tile.global_l1_idx % K_L1_SLOTS],
            k_gm, k_stride, tile_k_offset, copy_rows);
    }

    template<uint32_t TILE_Q, uint32_t TILE_L0, uint32_t TILE_L1, uint32_t DIM, class InputT>
    __aicore__ inline void prefetch_v(AICore_RT<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>& rt,
        GMTensor_ND<InputT>& v_gm, uint32_t v_stride, const AicQTaskScalars& q_task,
        const AicL0TileScalars& tile) {
        const uint32_t tile_v_begin = base_l0_tile(q_task, tile) * TILE_L0;
        const uint32_t tile_v_offset = q_task.kv_offset + tile_v_begin * v_stride;
        const uint32_t max_copy_rows = tile.num_parts * TILE_L0;
        const uint32_t remaining_rows = q_task.kv_len > tile_v_begin ? q_task.kv_len - tile_v_begin : 0U;
        const uint32_t copy_rows = remaining_rows < max_copy_rows ? remaining_rows : max_copy_rows;
        SWA_v5::Compute::copy_sync(
            rt.cube.V_B1[tile.global_l1_idx % V_L1_SLOTS],
            v_gm, v_stride, tile_v_offset, copy_rows);
    }

public:
    template<uint32_t TILE_Q, uint32_t TILE_L0, uint32_t TILE_L1, uint32_t DIM, class InputT>
    __aicore__ inline void init(AICore_RT<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>& rt,
        GMTensor_ND<InputT>& q_gm, GMTensor_ND<InputT>& k_gm, GMTensor_ND<InputT>& v_gm,
        uint32_t q_stride, uint32_t kv_stride, const AicQTaskScalars& q_task,
        const AicL0TileScalars& first_tile) {
        static_assert(TILE_L1 % TILE_L0 == 0 && TILE_L1 >= TILE_L0,
            "FullPathAIC expects TILE_L1 to be a multiple of TILE_L0");

        SWA_v5::Compute::copy_sync(rt.cube.Q_A1, q_gm, q_stride, q_task.q_offset, q_task.q_rows);
        rt.cube.Q_A1.consumer_wait();
        SWA_v5::Copy::copy_sync(rt.cube.Q_A2, rt.cube.Q_A1);
        prefetch_k<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(rt, k_gm, kv_stride, q_task, first_tile);
        prefetch_v<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(rt, v_gm, kv_stride, q_task, first_tile);
    }

    template<uint32_t TILE_Q, uint32_t TILE_L0, uint32_t TILE_L1, uint32_t DIM, class InputT>
    __aicore__ inline void run_cube1(AICore_RT<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>& rt,
        GMTensor_ND<InputT>& k_gm, uint32_t kv_stride, const AicQTaskScalars& q_task,
        const AicL0TileScalars& tile, bool has_next_task, const AicL0TileScalars& next_tile) {
        static_assert(TILE_L1 % TILE_L0 == 0 && TILE_L1 >= TILE_L0,
            "FullPathAIC expects TILE_L1 to be a multiple of TILE_L0");
        static_assert(TILE_Q == 128 && TILE_L0 == 128 && DIM == 128,
            "FullPathAIC is currently specialized for 128x128 tiles");

        const uint32_t qk_slot = tile.global_l0_idx % CUBE2_MM_SLOTS;
        const uint32_t following_local_l0_idx = tile.local_l0_idx + 1U;

        if (tile.local_l0_idx == 0U) {
            rt.cube.Q_A2.consumer_wait();
        }

        constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
        const uint32_t k_l1_slot = tile.global_l1_idx % K_L1_SLOTS;
        if (tile.part_idx == 0U) {
            rt.cube.K_B1[k_l1_slot].consumer_wait();
        }
        SWA_v5::Copy::split_copy_sync_trans<InputT, TILE_L1, DIM, L1_PARTS>(
            rt.cube.K_B2, rt.cube.K_B1[k_l1_slot], tile.part_idx);
        if (has_next_task) {
            prefetch_k<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(rt, k_gm, kv_stride, q_task, next_tile);
        }

        rt.cube.K_B2.consumer_wait();
        const bool k_l1_tile_consumed = (following_local_l0_idx >= q_task.valid_l0_count) ||
            (next_tile.local_l1_idx != tile.local_l1_idx);
        if (k_l1_tile_consumed) {
            rt.cube.K_B1[tile.global_l1_idx % K_L1_SLOTS].consumer_set();
        }
        SWA_v5::prim::mmad<InputT, InputT, TILE_Q, TILE_L0, DIM>(
            rt.cube.QK_CO1.tensor, rt.cube.Q_A2.tensor,
            rt.cube.K_B2.tensor, true, 3);
        rt.cube.K_B2.consumer_set();

        if (following_local_l0_idx >= q_task.valid_l0_count) {
            rt.cube.Q_A2.consumer_set();
            rt.cube.Q_A1.consumer_set();
        }

        UBTensor_ND<float, TILE_Q / 2, TILE_L0> qk_ub = rt.vec.qk_ub_abs(rt.pipe, qk_slot);
        rt.template wait_aiv<PIPE_FIX>(event_slot(CUBE2_QK_CONSUMED_BASE, qk_slot));
        SWA_v5::Copy::copy(qk_ub, rt.cube.QK_CO1, 3);
        rt.template notify_aiv<PIPE_FIX>(event_slot(CUBE2_QK_READY_BASE, qk_slot));
    }

    template<uint32_t TILE_Q, uint32_t TILE_L0, uint32_t TILE_L1, uint32_t DIM, class InputT>
    __aicore__ inline void run_cube2(AICore_RT<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>& rt,
        GMTensor_ND<InputT>& v_gm, uint32_t kv_stride, const AicQTaskScalars& q_task,
        const AicL0TileScalars& tile, bool preload_future_cube2, const AicQTaskScalars& future_q_task,
        const AicL0TileScalars& future_tile) {
        static_assert(TILE_L1 % TILE_L0 == 0 && TILE_L1 >= TILE_L0,
            "FullPathAIC expects TILE_L1 to be a multiple of TILE_L0");
        static_assert(TILE_Q == 128 && TILE_L0 == 128 && DIM == 128,
            "FullPathAIC is currently specialized for 128x128 tiles");

        const uint32_t p_slot = tile.global_l0_idx % CUBE2_P_SLOTS;
        const uint32_t bmm2_slot = tile.global_l0_idx % CUBE2_MM_SLOTS;

        constexpr uint32_t L1_PARTS = TILE_L1 / TILE_L0;
        const uint32_t v_l1_slot = tile.global_l1_idx % V_L1_SLOTS;
        if (tile.part_idx == 0U) {
            rt.cube.V_B1[v_l1_slot].consumer_wait();
        }
        SWA_v5::Copy::split_copy_sync<InputT, TILE_L1, DIM, L1_PARTS>(
            rt.cube.V_B2, rt.cube.V_B1[v_l1_slot], tile.part_idx);
        if (preload_future_cube2) {
            prefetch_v<TILE_Q, TILE_L0, TILE_L1, DIM, InputT>(rt, v_gm, kv_stride, future_q_task, future_tile);
        }
        rt.cube.V_B2.consumer_wait();
        const uint32_t following_local_l0_idx = tile.local_l0_idx + 1U;
        const bool v_l1_tile_consumed = (following_local_l0_idx >= q_task.valid_l0_count) ||
            (future_tile.local_l1_idx != tile.local_l1_idx);
        if (v_l1_tile_consumed) {
            rt.cube.V_B1[tile.global_l1_idx % V_L1_SLOTS].consumer_set();
        }

        if constexpr (SWA_V5_CUBE2_SKIP_P_READY_WAIT == 0) {
            rt.template wait_aiv<PIPE_MTE1>(event_slot(CUBE2_P_READY_BASE, p_slot));
        }
        SWA_v5::Copy::copy_sync(rt.cube.P_A2, rt.cube.P_A1[p_slot]);

        rt.cube.P_A2.consumer_wait();
        SWA_v5::Compute::matmul_sync<InputT, InputT, TILE_Q, DIM, TILE_L0>(
            rt.cube.PV_CO1, rt.cube.P_A2, rt.cube.V_B2, true, 3);
        rt.cube.P_A2.consumer_set();
        rt.cube.V_B2.consumer_set();

        UBTensor_ND<float, TILE_Q / 2, DIM> bmm2_ub = rt.vec.bmm2_ub_abs(rt.pipe, bmm2_slot);
        rt.template wait_aiv<PIPE_FIX>(event_slot(CUBE2_BMM2_CONSUMED_BASE, bmm2_slot));
        rt.cube.PV_CO1.consumer_wait();
        SWA_v5::Copy::copy(bmm2_ub, rt.cube.PV_CO1, 3);
        rt.cube.PV_CO1.consumer_set();
        rt.template notify_aiv<PIPE_FIX>(event_slot(CUBE2_BMM2_READY_BASE, bmm2_slot));
    }
};

}  // namespace SWA_v5::Pipeline::FullPath
