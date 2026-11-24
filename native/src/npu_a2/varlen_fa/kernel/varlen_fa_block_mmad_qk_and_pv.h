#pragma once

#include <cstdint>
#include <type_traits>

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "debug_utils.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"

namespace xpu_ops::kernels {

template <bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_KV_, uint32_t L1_STAGE_NUM_, uint32_t L0_STAGE_NUM_,
          bool IS_CAUSAL_>
struct QKAndPVPolicy {
  static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
  static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_KV_;
  static constexpr uint32_t L1_STAGE_NUM = L1_STAGE_NUM_;
  static constexpr uint32_t L0_STAGE_NUM = L0_STAGE_NUM_;
  static constexpr bool IS_CAUSAL = IS_CAUSAL_;
};

template <class ArchTag_, class ElementInOut_, class ElementCalc_, class Policy_, class EventSet_, class L1TileSizeFa_,
          class L0TileSizeQK_, class L0TileSizePV_>
class BlockMmadQKAndPV {
public:
  using ArchTag = ArchTag_;
  using ElementInOut = ElementInOut_;
  using ElementCalc = ElementCalc_;
  using Policy = Policy_;
  using EventSet = EventSet_;
  using L1TileSizeFa = L1TileSizeFa_;
  using L0TileSizeQK = L0TileSizeQK_;
  using L0TileSizePV = L0TileSizePV_;

  using LayoutQ = Catlass::layout::RowMajor;     // [M, K] -> [len_qo, head_qk]
  using LayoutK = Catlass::layout::ColumnMajor;  // [N, K] -> [len_kv, head_qk]
  using LayoutQK = Catlass::layout::RowMajor;    // [M, N] -> [tile_qo, tile_kv]
  using LayoutP = Catlass::layout::RowMajor;     // [M, K] -> [tile_qo, tile_kv]
  using LayoutV = Catlass::layout::RowMajor;     // [K, N] -> [len_kv, head_vo]
  using LayoutPV = Catlass::layout::RowMajor;    // [M, N] -> [tile_qo, head_vo]

  using QType = Catlass::Gemm::GemmType<ElementInOut, LayoutQ>;
  using KType = Catlass::Gemm::GemmType<ElementInOut, LayoutK>;
  using QKType = Catlass::Gemm::GemmType<ElementCalc, LayoutQK>;
  using PType = Catlass::Gemm::GemmType<ElementInOut, LayoutP>;
  using VType = Catlass::Gemm::GemmType<ElementInOut, LayoutV>;
  using PVType = Catlass::Gemm::GemmType<ElementCalc, LayoutPV>;

  using TileCopyQK = Catlass::Gemm::Tile::TileCopy<ArchTag, QType, KType, QKType>;
  using TileMmadQK = Catlass::Gemm::Tile::TileMmad<ArchTag, QType, KType, void>;
  using ElementAccumulateQK = typename TileMmadQK::ElementAccumulator;

  using CopyGmToL1AQK = typename TileCopyQK::CopyGmToL1A;
  using CopyGmToL1BQK = typename TileCopyQK::CopyGmToL1B;
  using CopyL1ToL0AQK = typename TileCopyQK::CopyL1ToL0A;
  using CopyL1ToL0BQK = typename TileCopyQK::CopyL1ToL0B;
  using CopyL0CToGmQK = typename TileCopyQK::CopyL0CToGm;

  using LayoutQInL1 = typename CopyL1ToL0AQK::LayoutSrc;
  using LayoutKInL1 = typename CopyL1ToL0BQK::LayoutSrc;
  using LayoutQInL0 = typename CopyL1ToL0AQK::LayoutDst;
  using LayoutKInL0 = typename CopyL1ToL0BQK::LayoutDst;
  using LayoutQKInL0 = Catlass::layout::zN;

  using TileCopyPV = Catlass::Gemm::Tile::TileCopy<ArchTag, PType, VType, PVType>;
  using TileMmadPV = Catlass::Gemm::Tile::TileMmad<ArchTag, PType, VType, void>;
  using ElementAccumulatePV = typename TileMmadPV::ElementAccumulator;

  using CopyGmToL1APV = typename TileCopyPV::CopyGmToL1A;
  using CopyGmToL1BPV = typename TileCopyPV::CopyGmToL1B;
  using CopyL1ToL0APV = typename TileCopyPV::CopyL1ToL0A;
  using CopyL1ToL0BPV = typename TileCopyPV::CopyL1ToL0B;
  using CopyL0CToGmPV = typename TileCopyPV::CopyL0CToGm;

  using LayoutPInL1 = typename CopyL1ToL0APV::LayoutSrc;
  using LayoutVInL1 = typename CopyL1ToL0BPV::LayoutSrc;
  using LayoutPInL0 = typename CopyL1ToL0APV::LayoutDst;
  using LayoutVInL0 = typename CopyL1ToL0BPV::LayoutDst;
  using LayoutPVInL0 = Catlass::layout::zN;

  static constexpr bool ENABLE_UNIT_FLAG = Policy::ENABLE_UNIT_FLAG;
  static constexpr bool ENABLE_SHUFFLE_K = Policy::ENABLE_SHUFFLE_K;
  static constexpr uint32_t L1_STAGE_NUM = Policy::L1_STAGE_NUM;
  static constexpr uint32_t L0_STAGE_NUM = Policy::L0_STAGE_NUM;
  static constexpr bool IS_CAUSAL = Policy::IS_CAUSAL;

  static constexpr uint32_t BLOCK_QO = L1TileSizeFa::M;
  static constexpr uint32_t BLOCK_KV = L1TileSizeFa::N;
  static constexpr uint32_t HEAD_DIM = L1TileSizeFa::K;

  static_assert(L0TileSizeQK::M == BLOCK_QO && L0TileSizeQK::N == BLOCK_KV && HEAD_DIM % L0TileSizeQK::K == 0,
                "L0TileSizeQK is not matched with L1TileSizeFa");
  static_assert(L0TileSizePV::M == BLOCK_QO && L0TileSizePV::N == HEAD_DIM && BLOCK_KV % L0TileSizePV::K == 0,
                "L0TileSizePV is not matched with L1TileSizeFa");

  static_assert(1 + 2 * L1_STAGE_NUM <= 8, "MTE2_MTE1 event resource is not enough");
  static_assert(2 * L0_STAGE_NUM <= 8, "M_MTE1 event resource is not enough");

  static constexpr uint32_t QK_READY = EventSet::QK_READY;
  static constexpr uint32_t SCORE_READY = EventSet::SCORE_READY;
  static constexpr uint32_t PV_READY = EventSet::PV_READY;

  static constexpr uint32_t L1_SIZE_Q = BLOCK_QO * HEAD_DIM * sizeof(ElementInOut);
  static constexpr uint32_t L1_SIZE_KV = BLOCK_KV * HEAD_DIM * sizeof(ElementInOut);
  static constexpr uint32_t L1_SIZE_P = BLOCK_KV * BLOCK_QO * sizeof(ElementInOut);

  static_assert(L1_SIZE_Q + (L1_SIZE_P + L1_SIZE_KV) * L1_STAGE_NUM <= ArchTag::L1_SIZE,
                "L1 size for PV is not enough");

  static constexpr uint32_t L0_SIZE_Q = L0TileSizeQK::M * L0TileSizeQK::K * sizeof(ElementInOut);
  static constexpr uint32_t L0_SIZE_K = L0TileSizeQK::N * L0TileSizeQK::K * sizeof(ElementInOut);
  static constexpr uint32_t L0_SIZE_P = L0TileSizePV::M * L0TileSizePV::K * sizeof(ElementInOut);
  static constexpr uint32_t L0_SIZE_V = L0TileSizePV::N * L0TileSizePV::K * sizeof(ElementInOut);
  static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGE_NUM;
  static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGE_NUM;

  static_assert(L0_SIZE_Q * L0_STAGE_NUM <= ArchTag::L0A_SIZE, "L0A size for QK is not enough");
  static_assert(L0_SIZE_K * L0_STAGE_NUM <= ArchTag::L0B_SIZE, "L0B size for QK is not enough");
  static_assert(L0_SIZE_P * L0_STAGE_NUM <= ArchTag::L0A_SIZE, "L0A size for PV is not enough");

  static constexpr uint32_t L0C_SIZE_QK = BLOCK_QO * BLOCK_KV * sizeof(ElementAccumulateQK);
  static constexpr uint32_t L0C_SIZE_PV = BLOCK_QO * HEAD_DIM * sizeof(ElementAccumulatePV);
  static_assert(L0C_SIZE_QK <= ArchTag::L0C_SIZE && L0C_SIZE_PV <= ArchTag::L0C_SIZE, "L0C size is not enough");

  struct Params {
    uint32_t tile_kv;
    uint32_t pipeline_stages;
  };

  CATLASS_DEVICE
  BlockMmadQKAndPV(const Params &params, Catlass::Arch::Resource<ArchTag> &resource, uint32_t l1_offset = 0) {
    tile_kv = params.tile_kv;
    pipeline_stages = params.pipeline_stages;
    qk_single_size = BLOCK_QO * tile_kv;
    pv_single_size = BLOCK_QO * HEAD_DIM;
    score_single_size = BLOCK_QO * tile_kv;

    l1_q_buf = resource.l1Buf.template GetBufferByByte<ElementInOut>(l1_offset);
    l1_q_event = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_q_event);

    l1_offset += L1_SIZE_Q;
    for (int i = 0; i < L1_STAGE_NUM; ++i) {
      l1_k_bufs[i] = resource.l1Buf.template GetBufferByByte<ElementInOut>(l1_offset + i * L1_SIZE_KV);
      l1_v_bufs[i] = resource.l1Buf.template GetBufferByByte<ElementInOut>(l1_offset + i * L1_SIZE_KV);
      l1_p_bufs[i] =
          resource.l1Buf.template GetBufferByByte<ElementInOut>(l1_offset + L1_STAGE_NUM * L1_SIZE_KV + i * L1_SIZE_P);

      l1_kv_events[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
      l1_p_events[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[i]);
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[i]);
    }

    for (int i = 0; i < L0_STAGE_NUM; ++i) {
      l0_q_bufs[i] = resource.l0ABuf.template GetBufferByByte<ElementInOut>(i * L0A_PINGPONG_BUF_SIZE);
      l0_k_bufs[i] = resource.l0BBuf.template GetBufferByByte<ElementInOut>(i * L0B_PINGPONG_BUF_SIZE);
      l0_p_bufs[i] = resource.l0ABuf.template GetBufferByByte<ElementInOut>(i * L0A_PINGPONG_BUF_SIZE);
      l0_v_bufs[i] = resource.l0BBuf.template GetBufferByByte<ElementInOut>(i * L0B_PINGPONG_BUF_SIZE);

      l0_a_events[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
      l0_b_events[i] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_a_events[i]);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_b_events[i]);
    }

    l0_qk_buf = resource.l0CBuf.template GetBufferByByte<ElementAccumulateQK>(0);
    l0_pv_buf = resource.l0CBuf.template GetBufferByByte<ElementAccumulatePV>(0);

    l0c_event = GetTPipePtr()->AllocEventID<AscendC::HardEvent::FIX_M>();
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0c_event);

    XPU_OPS_DEBUG_PRINT("BlockMmadQKAndPV::Initialized.\n");
  }

  CATLASS_DEVICE
  ~BlockMmadQKAndPV() {
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_q_event);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(l1_q_event);

    for (int i = 0; i < L1_STAGE_NUM; ++i) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[i]);
      GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[i]);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[i]);
      GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[i]);
    }

    for (int i = 0; i < L0_STAGE_NUM; ++i) {
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_a_events[i]);
      GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_MTE1>(l0_a_events[i]);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_b_events[i]);
      GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_MTE1>(l0_b_events[i]);
    }

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0c_event);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::FIX_M>(l0c_event);

    XPU_OPS_DEBUG_PRINT("BlockMmadQKAndPV::Destructed.\n");
  }

  CATLASS_DEVICE
  void operator()(const AscendC::GlobalTensor<ElementInOut> &q_gm, const LayoutQ &layout_q,
                  const AscendC::GlobalTensor<ElementInOut> &k_gm, const LayoutK &layout_k,
                  const AscendC::GlobalTensor<ElementInOut> &v_gm, const LayoutV &layout_v,
                  const AscendC::GlobalTensor<ElementCalc> &qk_pingpong_gm,
                  const AscendC::GlobalTensor<ElementInOut> &p_pingpong_gm,
                  const AscendC::GlobalTensor<ElementCalc> &pv_pingpong_gm) {
    uint32_t valid_len_kv = layout_k.shape(1);
    uint32_t num_tile_kv = (valid_len_kv + tile_kv - 1) / tile_kv;
    XPU_OPS_DEBUG_PRINT(
        "BlockMmadQKAndPV::operator() called with layout_q = (%u, %u), layout_k = (%u, %u), layout_v = (%u, %u), "
        "num_tile_kv = %u\n",
        layout_q.shape(0), layout_q.shape(1), layout_k.shape(0), layout_k.shape(1), layout_v.shape(0),
        layout_v.shape(1), num_tile_kv);
    uint32_t preload_num = pipeline_stages - 1;
    preload_num = preload_num < num_tile_kv ? preload_num : num_tile_kv;
    // for (uint32_t tile_kv_idx = 0; tile_kv_idx < num_tile_kv; ++tile_kv_idx) {
    //   uint32_t base_kv = tile_kv_idx * tile_kv;
    //   uint32_t actual_tile_kv = tile_kv_idx == num_tile_kv - 1 ? valid_len_kv - base_kv : tile_kv;
    //   auto tile_layout_k = layout_k.GetTileLayout(Catlass::MatrixCoord(HEAD_DIM, actual_tile_kv));
    //   uint32_t offset_k = layout_k.GetOffset(Catlass::MatrixCoord(0, base_kv));
    //   auto tile_layout_v = layout_v.GetTileLayout(Catlass::MatrixCoord(actual_tile_kv, HEAD_DIM));
    //   uint32_t offset_v = layout_v.GetOffset(Catlass::MatrixCoord(base_kv, 0));
    //   auto layout_p =
    //       LayoutP::template MakeLayoutInUb<ElementCalc>(Catlass::MatrixCoord(layout_q.shape(0), actual_tile_kv));

    //   uint32_t pingpong_idx = tile_kv_idx % pipeline_stages;

    //   XPU_OPS_DEBUG_PRINT("  process QK[%u] with actual_tile_kv=%u, offset_k=%u, offset_v=%u, pingpong_idx=%u\n",
    //                       tile_kv_idx, actual_tile_kv, offset_k, offset_v, pingpong_idx);

    //   partialMmadQK(q_gm, layout_q, k_gm[offset_k], tile_layout_k, qk_pingpong_gm[pingpong_idx * qk_single_size],
    //                 tile_kv_idx == 0, tile_kv_idx == num_tile_kv - 1);

    //   XPU_OPS_DEBUG_PRINT("  QK[%u] done.\n", tile_kv_idx);
    //   AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(QK_READY);

    //   XPU_OPS_DEBUG_PRINT("  QK[%u] ready.\n", tile_kv_idx);

    //   // syncMmadQK();

    //   AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(SCORE_READY);
    //   XPU_OPS_DEBUG_PRINT("  SCORE[%u] ready.\n", tile_kv_idx);

    //   XPU_OPS_DEBUG_PRINT("  process PV[%u].\n", tile_kv_idx);
    //   partialMmadPV(p_pingpong_gm[pingpong_idx * score_single_size], layout_p, v_gm[offset_v], tile_layout_v,
    //                 pv_pingpong_gm[pingpong_idx * pv_single_size]);
    //   XPU_OPS_DEBUG_PRINT("  PV[%u] done.\n", tile_kv_idx);

    //   AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PV_READY);

    //   XPU_OPS_DEBUG_PRINT("  PV[%u] ready.\n", tile_kv_idx);

    //   // syncMmadPV();
    // }
    for (uint32_t iter = 0; iter < num_tile_kv + preload_num; ++iter) {
      if (iter < num_tile_kv) {
        uint32_t tile_kv_idx = iter;
        uint32_t base_kv = tile_kv_idx * tile_kv;
        uint32_t actual_tile_kv = tile_kv_idx == num_tile_kv - 1 ? valid_len_kv - base_kv : tile_kv;
        auto tile_layout_k = layout_k.GetTileLayout(Catlass::MatrixCoord(HEAD_DIM, actual_tile_kv));
        uint32_t offset_k = layout_k.GetOffset(Catlass::MatrixCoord(0, base_kv));
        uint32_t pingpong_idx = tile_kv_idx % pipeline_stages;

        // XPU_OPS_DEBUG_PRINT("  process QK[%u] with actual_tile_kv=%u, offset_k=%u, offset_v=%u, pingpong_idx=%u\n",
        //                     tile_kv_idx, actual_tile_kv, offset_k, offset_v, pingpong_idx);

        partialMmadQK(q_gm, layout_q, k_gm[offset_k], tile_layout_k, qk_pingpong_gm[pingpong_idx * qk_single_size],
                      tile_kv_idx == 0, tile_kv_idx == num_tile_kv - 1);

        XPU_OPS_DEBUG_PRINT("  QK[%u] done.\n", tile_kv_idx);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(QK_READY);

        XPU_OPS_DEBUG_PRINT("  QK[%u] ready.\n", tile_kv_idx);
      }

      if (iter >= preload_num) {
        uint32_t tile_kv_idx = iter - preload_num;
        uint32_t base_kv = tile_kv_idx * tile_kv;
        uint32_t actual_tile_kv = tile_kv_idx == num_tile_kv - 1 ? valid_len_kv - base_kv : tile_kv;
        auto tile_layout_v = layout_v.GetTileLayout(Catlass::MatrixCoord(actual_tile_kv, HEAD_DIM));
        uint32_t offset_v = layout_v.GetOffset(Catlass::MatrixCoord(base_kv, 0));
        uint32_t padded_tile_kv = (actual_tile_kv + BLOCK_KV - 1) / BLOCK_KV * BLOCK_KV;
        LayoutP layout_p(BLOCK_QO, padded_tile_kv, padded_tile_kv);
        uint32_t pingpong_idx = tile_kv_idx % pipeline_stages;

        XPU_OPS_DEBUG_PRINT("  process PV[%u].\n", tile_kv_idx);
        partialMmadPV(p_pingpong_gm[pingpong_idx * score_single_size], layout_p, v_gm[offset_v], tile_layout_v,
                      pv_pingpong_gm[pingpong_idx * pv_single_size]);
        XPU_OPS_DEBUG_PRINT("  PV[%u] done.\n", tile_kv_idx);

        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(PV_READY);

        XPU_OPS_DEBUG_PRINT("  PV[%u] ready.\n", tile_kv_idx);
      }
    }
  }

  CATLASS_DEVICE
  void zeroL1(const AscendC::LocalTensor<ElementInOut> &tensor, uint32_t element_count) {
    constexpr uint32_t BYTES_PER_BLOCK = 32;
    AscendC::InitConstValueParams<ElementInOut> params(
        1, static_cast<uint16_t>(element_count * sizeof(ElementInOut) / BYTES_PER_BLOCK), 0,
        static_cast<ElementInOut>(0));
    AscendC::InitConstValue(tensor, params);
    // Complete initialization before ND2NZ overwrites the valid portion.
    AscendC::PipeBarrier<PIPE_MTE2>();
  }

  CATLASS_DEVICE
  void partialMmadQK(const AscendC::GlobalTensor<ElementInOut> &q_gm, const LayoutQ &layout_q,
                     const AscendC::GlobalTensor<ElementInOut> &k_gm, const LayoutK &layout_k,
                     const AscendC::GlobalTensor<ElementCalc> &qk_pingpong_gm, bool first_tile_kv, bool last_tile_kv) {
    uint32_t valid_len_qo = layout_q.shape(0);
    uint32_t valid_len_kv = layout_k.shape(1);
    uint32_t padded_len_kv = (valid_len_kv + BLOCK_KV - 1) / BLOCK_KV * BLOCK_KV;
    LayoutQK layout_qk(BLOCK_QO, padded_len_kv, padded_len_kv);
    auto layout_q_in_l1 = LayoutQInL1::template MakeLayout<ElementInOut>(BLOCK_QO, HEAD_DIM);

    if (first_tile_kv) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_q_event);
      if (valid_len_qo < BLOCK_QO) {
        zeroL1(l1_q_buf, BLOCK_QO * HEAD_DIM);
      }
      copy_q_to_l1(l1_q_buf, q_gm, layout_q_in_l1, layout_q);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1_q_event);
    }

    uint32_t kv_loop_num = padded_len_kv / BLOCK_KV;
    uint32_t k_start_idx = ENABLE_SHUFFLE_K ? AscendC::GetBlockIdx() : 0;

    for (uint32_t kv_loop_idx = 0; kv_loop_idx < kv_loop_num; ++kv_loop_idx) {
      uint32_t shuffled_kv_idx = (kv_loop_idx + k_start_idx) % kv_loop_num;
      uint32_t block_start = shuffled_kv_idx * BLOCK_KV;
      uint32_t valid_n = block_start + BLOCK_KV <= valid_len_kv ? BLOCK_KV : valid_len_kv - block_start;
      auto layout_k_in_l1 = LayoutKInL1::template MakeLayout<ElementInOut>(HEAD_DIM, BLOCK_KV);

      if (kv_loop_idx == 0) {
        auto tile_layout_k = layout_k.GetTileLayout(Catlass::MatrixCoord(HEAD_DIM, valid_n));
        uint32_t offset_k = layout_k.GetOffset(Catlass::MatrixCoord(0, block_start));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[l1_kv_event_id]);
        if (valid_n < BLOCK_KV) {
          zeroL1(l1_k_bufs[l1_kv_event_id], HEAD_DIM * BLOCK_KV);
        }
        copy_k_to_l1(l1_k_bufs[l1_kv_event_id], k_gm[offset_k], layout_k_in_l1, tile_layout_k);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1_kv_events[l1_kv_event_id]);
      }

      if (kv_loop_idx < kv_loop_num - 1) {
        uint32_t next_l1_event_id = l1_kv_event_id + 1 < L1_STAGE_NUM ? l1_kv_event_id + 1 : 0;
        uint32_t next_shuffled_kv_idx = (shuffled_kv_idx + 1) % kv_loop_num;
        uint32_t next_block_start = next_shuffled_kv_idx * BLOCK_KV;
        uint32_t next_valid_n =
            next_block_start + BLOCK_KV <= valid_len_kv ? BLOCK_KV : valid_len_kv - next_block_start;
        auto next_layout_k_in_l1 = LayoutKInL1::template MakeLayout<ElementInOut>(HEAD_DIM, BLOCK_KV);
        auto next_tile_layout_k = layout_k.GetTileLayout(Catlass::MatrixCoord(HEAD_DIM, next_valid_n));
        uint32_t next_offset_k = layout_k.GetOffset(Catlass::MatrixCoord(0, next_block_start));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[next_l1_event_id]);
        if (next_valid_n < BLOCK_KV) {
          zeroL1(l1_k_bufs[next_l1_event_id], HEAD_DIM * BLOCK_KV);
        }
        copy_k_to_l1(l1_k_bufs[next_l1_event_id], k_gm[next_offset_k], next_layout_k_in_l1, next_tile_layout_k);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1_kv_events[next_l1_event_id]);
      }

      auto layout_qk_in_l0 = LayoutQKInL0::MakeLayoutInL0C(Catlass::MatrixCoord(BLOCK_QO, BLOCK_KV));
      constexpr uint32_t kv_inner_loop_num = HEAD_DIM / L0TileSizeQK::K;
      uint32_t l0a_event_id = 0;
      uint32_t l0b_event_id = 0;
      auto l1_k_buf = l1_k_bufs[l1_kv_event_id];

      if (!ENABLE_UNIT_FLAG) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0c_event);
      }

      for (uint32_t kv_inner_loop_idx = 0; kv_inner_loop_idx < kv_inner_loop_num; ++kv_inner_loop_idx) {
        auto l0_q_buf = l0_q_bufs[l0a_event_id];
        auto l0_k_buf = l0_k_bufs[l0b_event_id];

        auto layout_q_in_l0 = LayoutQInL0::template MakeLayout<ElementInOut>(BLOCK_QO, L0TileSizeQK::K);
        auto layout_tile_q_in_l1 =
            layout_q_in_l1.GetTileLayout(Catlass::MatrixCoord(BLOCK_QO, L0TileSizeQK::K));
        uint32_t tile_q_offset =
            layout_q_in_l1.GetOffset(Catlass::MatrixCoord(0, kv_inner_loop_idx * L0TileSizeQK::K));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_a_events[l0a_event_id]);
        if (first_tile_kv && kv_loop_idx == 0 && kv_inner_loop_idx == 0) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1_q_event);
        }
        copy_q_to_l0a(l0_q_buf, l1_q_buf[tile_q_offset], layout_q_in_l0, layout_tile_q_in_l1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0_a_events[l0a_event_id]);
        if (last_tile_kv && kv_loop_idx == kv_loop_num - 1 && kv_inner_loop_idx == kv_inner_loop_num - 1) {
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_q_event);
        }

        auto layout_k_in_l0 = LayoutKInL0::template MakeLayout<ElementInOut>(L0TileSizeQK::K, BLOCK_KV);
        auto layout_tile_k_in_l1 =
            layout_k_in_l1.GetTileLayout(Catlass::MatrixCoord(L0TileSizeQK::K, BLOCK_KV));
        uint32_t tile_k_offset =
            layout_k_in_l1.GetOffset(Catlass::MatrixCoord(kv_inner_loop_idx * L0TileSizeQK::K, 0));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_b_events[l0b_event_id]);
        if (kv_inner_loop_idx == 0) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1_kv_events[l1_kv_event_id]);
        }
        copy_k_to_l0b(l0_k_buf, l1_k_buf[tile_k_offset], layout_k_in_l0, layout_tile_k_in_l1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0_b_events[l0b_event_id]);
        if (kv_inner_loop_idx == kv_inner_loop_num - 1) {
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[l1_kv_event_id]);
        }

        uint8_t unit_flag = 0b00;
        if constexpr (ENABLE_UNIT_FLAG) {
          unit_flag = kv_inner_loop_idx == kv_inner_loop_num - 1 ? 0b11 : 0b10;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0_a_events[l0a_event_id]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0_b_events[l0b_event_id]);
        XPU_OPS_DEBUG_PRINT("      launch padded QK with (m, n, k) = (%u, %u, %u).\n", BLOCK_QO, BLOCK_KV,
                            L0TileSizeQK::K);
        mmad_qk(l0_qk_buf, l0_q_buf, l0_k_buf, BLOCK_QO, BLOCK_KV, L0TileSizeQK::K,
                kv_inner_loop_idx == 0, unit_flag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_a_events[l0a_event_id]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_b_events[l0b_event_id]);

        l0a_event_id = l0a_event_id + 1 < L0_STAGE_NUM ? l0a_event_id + 1 : 0;
        l0b_event_id = l0b_event_id + 1 < L0_STAGE_NUM ? l0b_event_id + 1 : 0;
      }

      auto tile_layout_qk = layout_qk.GetTileLayout(Catlass::MatrixCoord(BLOCK_QO, BLOCK_KV));
      uint32_t offset_qk = layout_qk.GetOffset(Catlass::MatrixCoord(0, block_start));
      if constexpr (!ENABLE_UNIT_FLAG) {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0c_event);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0c_event);
        copy_qk_to_gm(qk_pingpong_gm[offset_qk], l0_qk_buf, tile_layout_qk, layout_qk_in_l0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0c_event);
      } else {
        copy_qk_to_gm(qk_pingpong_gm[offset_qk], l0_qk_buf, tile_layout_qk, layout_qk_in_l0, 0b11);
      }
      l1_kv_event_id = l1_kv_event_id + 1 < L1_STAGE_NUM ? l1_kv_event_id + 1 : 0;
    }
  }

  // CATLASS_DEVICE
  // void syncMmadQK() {
  //   // TODO: as K/V has same L1 size, we just need to synchronize based on previous mmad launched.
  //   l1_kv_event_id = 0;
  //   l1_p_event_id = 0;
  //   for (uint32_t i = 0; i < L1_STAGE_NUM; ++i) {
  //     AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[i]);
  //     AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[i]);
  //   }
  // }

  CATLASS_DEVICE
  void partialMmadPV(const AscendC::GlobalTensor<ElementInOut> &p_gm, const LayoutP &layout_p,
                     const AscendC::GlobalTensor<ElementInOut> &v_gm, const LayoutV &layout_v,
                     const AscendC::GlobalTensor<ElementCalc> &pv_pingpong_gm) {
    uint32_t padded_len_qo = layout_p.shape(0);
    uint32_t padded_len_kv = layout_p.shape(1);
    uint32_t valid_len_kv = layout_v.shape(0);
    LayoutPV layout_pv(BLOCK_QO, HEAD_DIM, HEAD_DIM);
    auto layout_pv_in_l0 = LayoutPVInL0::MakeLayoutInL0C(Catlass::MatrixCoord(BLOCK_QO, HEAD_DIM));
    uint32_t kv_loop_num = padded_len_kv / BLOCK_KV;

    if constexpr (!ENABLE_UNIT_FLAG) {
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0c_event);
    }

    uint32_t k_start_idx = ENABLE_SHUFFLE_K ? AscendC::GetBlockIdx() : 0;
    for (uint32_t kv_loop_idx = 0; kv_loop_idx < kv_loop_num; ++kv_loop_idx) {
      uint32_t shuffled_kv_idx = (kv_loop_idx + k_start_idx) % kv_loop_num;
      uint32_t block_start = shuffled_kv_idx * BLOCK_KV;
      uint32_t valid_k = block_start + BLOCK_KV <= valid_len_kv ? BLOCK_KV : valid_len_kv - block_start;
      auto layout_p_in_l1 = LayoutPInL1::template MakeLayout<ElementInOut>(BLOCK_QO, BLOCK_KV);
      auto layout_v_in_l1 = LayoutVInL1::template MakeLayout<ElementInOut>(BLOCK_KV, HEAD_DIM);

      if (kv_loop_idx == 0) {
        auto tile_layout_p = layout_p.GetTileLayout(Catlass::MatrixCoord(BLOCK_QO, BLOCK_KV));
        uint32_t offset_p = layout_p.GetOffset(Catlass::MatrixCoord(0, block_start));
        auto tile_layout_v = layout_v.GetTileLayout(Catlass::MatrixCoord(valid_k, HEAD_DIM));
        uint32_t offset_v = layout_v.GetOffset(Catlass::MatrixCoord(block_start, 0));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[l1_kv_event_id]);
        if (valid_k < BLOCK_KV) {
          zeroL1(l1_v_bufs[l1_kv_event_id], BLOCK_KV * HEAD_DIM);
        }
        copy_v_to_l1(l1_v_bufs[l1_kv_event_id], v_gm[offset_v], layout_v_in_l1, tile_layout_v);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1_kv_events[l1_kv_event_id]);

        AscendC::CrossCoreWaitFlag(SCORE_READY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[l1_p_event_id]);
        copy_p_to_l1(l1_p_bufs[l1_p_event_id], p_gm[offset_p], layout_p_in_l1, tile_layout_p);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1_p_events[l1_p_event_id]);
      }

      if (kv_loop_idx < kv_loop_num - 1) {
        uint32_t next_l1_p_event_id = l1_p_event_id + 1 < L1_STAGE_NUM ? l1_p_event_id + 1 : 0;
        uint32_t next_l1_kv_event_id = l1_kv_event_id + 1 < L1_STAGE_NUM ? l1_kv_event_id + 1 : 0;
        uint32_t next_shuffled_kv_idx = (shuffled_kv_idx + 1) % kv_loop_num;
        uint32_t next_block_start = next_shuffled_kv_idx * BLOCK_KV;
        uint32_t next_valid_k =
            next_block_start + BLOCK_KV <= valid_len_kv ? BLOCK_KV : valid_len_kv - next_block_start;
        auto next_tile_layout_p = layout_p.GetTileLayout(Catlass::MatrixCoord(BLOCK_QO, BLOCK_KV));
        uint32_t next_offset_p = layout_p.GetOffset(Catlass::MatrixCoord(0, next_block_start));
        auto next_layout_p_in_l1 = LayoutPInL1::template MakeLayout<ElementInOut>(BLOCK_QO, BLOCK_KV);
        auto next_tile_layout_v = layout_v.GetTileLayout(Catlass::MatrixCoord(next_valid_k, HEAD_DIM));
        uint32_t next_offset_v = layout_v.GetOffset(Catlass::MatrixCoord(next_block_start, 0));
        auto next_layout_v_in_l1 = LayoutVInL1::template MakeLayout<ElementInOut>(BLOCK_KV, HEAD_DIM);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[next_l1_p_event_id]);
        copy_p_to_l1(l1_p_bufs[next_l1_p_event_id], p_gm[next_offset_p], next_layout_p_in_l1, next_tile_layout_p);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1_p_events[next_l1_p_event_id]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[next_l1_kv_event_id]);
        if (next_valid_k < BLOCK_KV) {
          zeroL1(l1_v_bufs[next_l1_kv_event_id], BLOCK_KV * HEAD_DIM);
        }
        copy_v_to_l1(l1_v_bufs[next_l1_kv_event_id], v_gm[next_offset_v], next_layout_v_in_l1, next_tile_layout_v);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1_kv_events[next_l1_kv_event_id]);
      }

      auto l1_p_stage = l1_p_bufs[l1_p_event_id];
      auto l1_v_stage = l1_v_bufs[l1_kv_event_id];
      constexpr uint32_t inner_loop_num = BLOCK_KV / L0TileSizePV::K;
      uint32_t l0a_event_id = 0;
      uint32_t l0b_event_id = 0;

      for (uint32_t kv_inner_loop_idx = 0; kv_inner_loop_idx < inner_loop_num; ++kv_inner_loop_idx) {
        auto l0_p_buf = l0_p_bufs[l0a_event_id];
        auto l0_v_buf = l0_v_bufs[l0b_event_id];
        auto layout_p_in_l0 = LayoutPInL0::template MakeLayout<ElementInOut>(BLOCK_QO, L0TileSizePV::K);
        auto layout_tile_p_in_l1 =
            layout_p_in_l1.GetTileLayout(Catlass::MatrixCoord(BLOCK_QO, L0TileSizePV::K));
        uint32_t tile_p_offset =
            layout_p_in_l1.GetOffset(Catlass::MatrixCoord(0, kv_inner_loop_idx * L0TileSizePV::K));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_a_events[l0a_event_id]);
        if (kv_inner_loop_idx == 0) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1_p_events[l1_p_event_id]);
        }
        copy_p_to_l0a(l0_p_buf, l1_p_stage[tile_p_offset], layout_p_in_l0, layout_tile_p_in_l1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0_a_events[l0a_event_id]);
        if (kv_inner_loop_idx == inner_loop_num - 1) {
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[l1_p_event_id]);
        }

        auto layout_v_in_l0 = LayoutVInL0::template MakeLayout<ElementInOut>(L0TileSizePV::K, HEAD_DIM);
        auto layout_tile_v_in_l1 =
            layout_v_in_l1.GetTileLayout(Catlass::MatrixCoord(L0TileSizePV::K, HEAD_DIM));
        uint32_t tile_v_offset =
            layout_v_in_l1.GetOffset(Catlass::MatrixCoord(kv_inner_loop_idx * L0TileSizePV::K, 0));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0_b_events[l0b_event_id]);
        if (kv_inner_loop_idx == 0) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1_kv_events[l1_kv_event_id]);
        }
        copy_v_to_l0b(l0_v_buf, l1_v_stage[tile_v_offset], layout_v_in_l0, layout_tile_v_in_l1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0_b_events[l0b_event_id]);
        if (kv_inner_loop_idx == inner_loop_num - 1) {
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[l1_kv_event_id]);
        }

        uint8_t unit_flag = 0b00;
        if constexpr (ENABLE_UNIT_FLAG) {
          unit_flag = (kv_loop_idx == kv_loop_num - 1 && kv_inner_loop_idx == inner_loop_num - 1) ? 0b11 : 0b10;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0_a_events[l0a_event_id]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0_b_events[l0b_event_id]);
        XPU_OPS_DEBUG_PRINT("      launch padded PV with (m, n, k) = (%u, %u, %u).\n", BLOCK_QO, HEAD_DIM,
                            L0TileSizePV::K);
        mmad_pv(l0_pv_buf, l0_p_buf, l0_v_buf, BLOCK_QO, HEAD_DIM, L0TileSizePV::K,
                kv_inner_loop_idx == 0 && kv_loop_idx == 0, unit_flag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_a_events[l0a_event_id]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0_b_events[l0b_event_id]);

        l0a_event_id = l0a_event_id + 1 < L0_STAGE_NUM ? l0a_event_id + 1 : 0;
        l0b_event_id = l0b_event_id + 1 < L0_STAGE_NUM ? l0b_event_id + 1 : 0;
      }

      l1_p_event_id = l1_p_event_id + 1 < L1_STAGE_NUM ? l1_p_event_id + 1 : 0;
      l1_kv_event_id = l1_kv_event_id + 1 < L1_STAGE_NUM ? l1_kv_event_id + 1 : 0;
    }

    if constexpr (!ENABLE_UNIT_FLAG) {
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0c_event);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0c_event);
      copy_pv_to_gm(pv_pingpong_gm, l0_pv_buf, layout_pv, layout_pv_in_l0);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0c_event);
    } else {
      copy_pv_to_gm(pv_pingpong_gm, l0_pv_buf, layout_pv, layout_pv_in_l0, 0b11);
    }
  }

  // CATLASS_DEVICE
  // void syncMmadPV() {
  //   l1_kv_event_id = 0;
  //   l1_p_event_id = 0;
  //   for (uint32_t i = 0; i < L1_STAGE_NUM; ++i) {
  //     AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[i]);
  //     AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_p_events[i]);
  //     AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[i]);
  //     AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1_kv_events[i]);
  //   }
  // }

private:
  uint32_t tile_kv, pipeline_stages;
  uint32_t qk_single_size, score_single_size, pv_single_size;

  AscendC::LocalTensor<ElementInOut> l1_q_buf;
  // k and p/v share the same space
  AscendC::LocalTensor<ElementInOut> l1_k_bufs[L1_STAGE_NUM];
  AscendC::LocalTensor<ElementInOut> l1_p_bufs[L1_STAGE_NUM];
  AscendC::LocalTensor<ElementInOut> l1_v_bufs[L1_STAGE_NUM];

  // q and p share the same space; v and k share the same space
  AscendC::LocalTensor<ElementInOut> l0_q_bufs[L0_STAGE_NUM];
  AscendC::LocalTensor<ElementInOut> l0_k_bufs[L0_STAGE_NUM];
  AscendC::LocalTensor<ElementInOut> l0_p_bufs[L0_STAGE_NUM];
  AscendC::LocalTensor<ElementInOut> l0_v_bufs[L0_STAGE_NUM];

  // qk and pv share the same space
  AscendC::LocalTensor<ElementAccumulateQK> l0_qk_buf;
  AscendC::LocalTensor<ElementAccumulatePV> l0_pv_buf;

  uint32_t l1_q_event;
  uint32_t l1_kv_events[L1_STAGE_NUM];
  uint32_t l1_p_events[L1_STAGE_NUM];

  uint32_t l1_kv_event_id = 0;
  uint32_t l1_p_event_id = 0;

  uint32_t l0_a_events[L0_STAGE_NUM];
  uint32_t l0_b_events[L0_STAGE_NUM];

  uint32_t l0c_event;

  CopyGmToL1AQK copy_q_to_l1;
  CopyGmToL1BQK copy_k_to_l1;
  CopyL1ToL0AQK copy_q_to_l0a;
  CopyL1ToL0BQK copy_k_to_l0b;
  CopyL0CToGmQK copy_qk_to_gm;

  TileMmadQK mmad_qk;

  CopyGmToL1APV copy_p_to_l1;
  CopyGmToL1BPV copy_v_to_l1;
  CopyL1ToL0APV copy_p_to_l0a;
  CopyL1ToL0BPV copy_v_to_l0b;
  CopyL0CToGmPV copy_pv_to_gm;

  TileMmadPV mmad_pv;
};

}  // namespace xpu_ops::kernels
