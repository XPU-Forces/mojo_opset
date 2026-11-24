// Adapted from data/xpu_ops commit 8452b29b; see group_gemm_SOURCE.json.
#pragma once

#include "group_gemm_training_utils.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "math_utils.h"

namespace mojo::native::a2::kernels {

template <typename InputType, typename ComputeType, typename Config> class GroupGemm {
public:
  __aicore__ inline GroupGemm() {}

  __aicore__ inline void init(GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
                              const GMMTrainBaseParams *__restrict gmmBaseParams, TCubeTiling &mmTilingData,
                              AscendC::TPipe &pipe);

  __aicore__ inline void compute(uint32_t groupIdx, MNConfig &mnConfig, uint32_t subBlockIdx);

  __aicore__ inline uint64_t calWeightOffset(uint32_t tailN, uint32_t k);

public:
  // gemm ffn1 type
  using GemmInputType =
      matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, InputType, Config::TRANSPOSE_A>;

#if MOJO_GROUP_WEIGHT_NZ
  constexpr static auto MATMUL_CFG = GetMDLConfig(false, false, 0, true, false, false, false);
  using GemmWeightType =
      matmul::MatmulType<matmul::TPosition::GM, CubeFormat::NZ, InputType, Config::TRANSPOSE_B>;
#else
  constexpr static auto MATMUL_CFG = CFG_MDL;
  using GemmWeightType =
      matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, InputType, Config::TRANSPOSE_B>;
#endif

  using GemmBiasType = matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, ComputeType>;
  using GemmOutputType = matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, ComputeType>;

  matmul::MatmulImpl<GemmInputType, GemmWeightType, GemmBiasType, GemmOutputType, MATMUL_CFG>
      matmul_impl_obj_;

protected:
  AscendC::GlobalTensor<InputType> input_gm_;
  AscendC::GlobalTensor<InputType> weight_gm_;
  AscendC::GlobalTensor<InputType> bias_gm_;
  AscendC::GlobalTensor<InputType> output_gm_;

  uint32_t coreNum;
  int32_t orgM;
  int32_t orgK;
};

template <typename InputType, typename ComputeType, typename Config>
__aicore__ inline void GroupGemm<InputType, ComputeType, Config>::init(
    GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
    const GMMTrainBaseParams *__restrict gmmBaseParams, TCubeTiling &mmTilingData, AscendC::TPipe &pipe) {
  input_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputType *>(input));
  bias_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputType *>(bias));
  weight_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputType *>(weight));
  output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputType *>(output));

  if ASCEND_IS_AIC {
    matmul_impl_obj_.Init(&(mmTilingData), &pipe);
  }

  coreNum = gmmBaseParams->coreNum;
  orgM = mmTilingData.M;
  orgK = mmTilingData.Ka;
}

template <typename InputType, typename ComputeType, typename Config>
__aicore__ inline uint64_t GroupGemm<InputType, ComputeType, Config>::calWeightOffset(uint32_t tailN,
                                                                                      uint32_t k) {
  uint64_t wOffset = 0;
  if constexpr (Config::IS_WEIGHT_NZ && Config::TRANSPOSE_B) {
    wOffset = tailN * 16; // 16: float16 is 16, quant is 32
  } else if constexpr (Config::IS_WEIGHT_NZ) {
    wOffset = tailN * align_up<16>(k); // 16: nz format last two dim size
  } else if constexpr (Config::TRANSPOSE_B) {
    wOffset = tailN * k;
  } else {
    wOffset = tailN;
  }
  return wOffset;
}

template <typename InputType, typename ComputeType, typename Config>
__aicore__ inline void GroupGemm<InputType, ComputeType, Config>::compute(uint32_t groupIdx,
                                                                          MNConfig &mnConfig,
                                                                          uint32_t subBlockIdx) {
  if (subBlockIdx != 0) {
    return;
  }

  uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
  uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
  uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1
                            ? mnConfig.singleM
                            : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;

  uint64_t xOffset = 0;
  if constexpr (Config::TRANSPOSE_A) {
    xOffset = mnConfig.mIdx * mnConfig.singleM;
  } else {
    xOffset = mnConfig.mIdx * mnConfig.singleM * orgK;
  }

  uint64_t wOffset = calWeightOffset(tailN, orgK);
  auto weight_gm_tmp = weight_gm_[wOffset + mnConfig.wBaseOffset];
  if (mnConfig.blockDimM == 1) {
    weight_gm_tmp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  }

  uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;

  matmul_impl_obj_.SetOrgShape(orgM, mnConfig.n, orgK);
  matmul_impl_obj_.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
  matmul_impl_obj_.SetTensorA(input_gm_[xOffset + mnConfig.xBaseOffset], Config::TRANSPOSE_A);
  matmul_impl_obj_.SetTensorB(weight_gm_tmp, Config::TRANSPOSE_B);
  if constexpr (Config::HAS_BIAS) {
    matmul_impl_obj_.SetBias(bias_gm_[tailN + mnConfig.nAxisBaseOffset]);
  }
  matmul_impl_obj_.template IterateAll(output_gm_[outOffset + mnConfig.yBaseOffset], 0);
}

} // namespace mojo::native::a2::kernels
