#pragma once
#include "group_tiling.h"
#include "torch_utils.h"
#include <climits>
#include <map>
#include <tiling/platform/platform_ascendc.h>
#include <tiling/tiling_api.h>
#include <tuple>
namespace mojo::native::a2 {
inline void validate_group(const at::Tensor &x, const at::Tensor &w, const at::Tensor &counts, bool wgrad) {
  tensor(x, "input");
  tensor(w, "weight");
  tensor(counts, "counts");
  TORCH_CHECK(x.dim() == 2 && w.dim() == (wgrad ? 2 : 3),
              "group GEMM expects a matrix and grouped weights, or two "
              "matrices for wgrad");
  TORCH_CHECK(x.scalar_type() == w.scalar_type() &&
                  (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf),
              "group GEMM requires matching BF16/FP16 inputs");
  TORCH_CHECK(counts.dim() == 1 && counts.scalar_type() == at::kInt, "expert counts must be int32");
  TORCH_CHECK(x.device() == w.device() && x.device() == counts.device(),
              "group GEMM tensors must share device");
  TORCH_CHECK(wgrad || w.size(0) == counts.numel(), "one count per expert is required");
  for (auto d : x.sizes())
    TORCH_CHECK(d <= INT32_MAX, "group GEMM dimension exceeds int32");
  for (auto d : w.sizes())
    TORCH_CHECK(d <= INT32_MAX, "group GEMM dimension exceeds int32");
}
inline GroupTiling make_group_tiling(int64_t m, int64_t n, int64_t k, int64_t groups, at::ScalarType dtype,
                                     bool wgrad, uint32_t cores) {
  // Tiling depends only on tensor metadata; cache it on the submitting thread.
  using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, at::ScalarType, bool, uint32_t>;
  thread_local std::map<Key, GroupTiling> cache;
  Key key{m, n, k, groups, dtype, wgrad, cores};
  if (auto it = cache.find(key); it != cache.end())
    return it->second;
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform, "cannot initialize A2 matmul platform");
  uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0;
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  const uint32_t bn = 256;
  const uint32_t bk = (l0b / 2 / (bn * 2)) / 16 * 16;
  uint32_t bm = std::min<uint64_t>(l0a / 2 / (bk * 2), l0c / (bn * 4));
  bm = std::min<uint32_t>(256, bm > m ? (m + 15) / 16 * 16 : bm / 16 * 16);
  TORCH_CHECK(bm && bk, "invalid A2 matmul block sizes");
  matmul_tiling::PlatformInfo info;
  info.l1Size = l1;
  info.l0CSize = l0c;
  info.ubSize = ub;
  info.l0ASize = l0a;
  info.l0BSize = l0b;
  info.socVersion = platform_ascendc::SocVersion::ASCEND910B;
  matmul_tiling::MultiCoreMatmulTiling mm(info);
  const auto type =
      dtype == at::kBFloat16 ? matmul_tiling::DataType::DT_BF16 : matmul_tiling::DataType::DT_FLOAT16;
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, type, false);
  mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, type, false);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
              matmul_tiling::DataType::DT_FLOAT16);
  mm.SetBias(false);
  mm.SetOrgShape(m, n, k);
  mm.SetShape(m, bn, k);
  mm.SetFixSplit(bm, bn, bk);
  mm.SetBufferSpace(l1, l0c, ub);
  optiling::TCubeTiling data;
  TORCH_CHECK(mm.GetTiling(data) != -1, "cannot compute native grouped matmul tiling");
  uint32_t ska = 64 * 1024 / (bm * bk * 2), skb = 128 * 1024 / (bn * bk * 2);
  TORCH_CHECK(ska && skb, "invalid native grouped matmul L1 steps");
  if (ska > skb)
    ska = ska / skb * skb;
  else if (ska < skb)
    skb = skb / ska * ska;
  data.set_dbL0C(1);
  data.set_usedCoreNum(cores);
  data.set_baseM(bm);
  data.set_baseN(bn);
  data.set_baseK(bk);
  data.set_stepKa(ska);
  data.set_depthA1(ska * 2);
  data.set_stepKb(skb);
  data.set_depthB1(skb * 2);
  data.set_stepM(1);
  data.set_stepN(1);
  GroupTiling tiling{};
  tiling.base.groupNum = groups;
  tiling.base.coreNum = cores;
  tiling.base.groupType = wgrad ? 2 : 0;
  TORCH_CHECK(data.GetDataSize() <= sizeof(tiling.matmul), "increase native matmul tiling payload");
  data.SaveToBuffer(tiling.matmul, sizeof(tiling.matmul));
  if (cache.size() >= 256)
    cache.clear();
  cache.emplace(key, tiling);
  return tiling;
}
} // namespace mojo::native::a2
