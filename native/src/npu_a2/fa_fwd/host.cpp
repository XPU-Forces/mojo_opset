#include "host.h"
#include "host_tiling.h"
#include "../common/fa_host.h"
#include <array>
#include <tiling/platform/platform_ascendc.h>
#include "aclrtlaunch_mojo_a2_fa_fwd_bf16_0.h"
#include "aclrtlaunch_mojo_a2_fa_fwd_bf16_1.h"
#include "aclrtlaunch_mojo_a2_fa_fwd_fp16_0.h"
#include "aclrtlaunch_mojo_a2_fa_fwd_fp16_1.h"
namespace mojo::native::a2 {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
FlashAttentionForward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &mask,
                      const at::Tensor &cuq, const at::Tensor &cuk, int64_t maxq, int64_t maxk, double scale,
                      bool causal) {
  validate_fa(q, k, v, mask, cuq, cuk, causal);
  TORCH_CHECK(maxq > 0 && maxk > 0 && maxq <= INT_MAX && maxk <= INT_MAX,
              "FA maximum lengths must be positive int32");
  c10::DeviceGuard guard(q.device());
  const auto ns = c10_npu::getCurrentNPUStream(q.get_device());
  const auto stream = ns.stream(false);
  const auto cores = core_counts(q.get_device());
  const int64_t batch = cuq.numel(), n1 = q.size(1), n2 = k.size(1), d = q.size(2), g = n1 / n2;
  const int64_t aq = (maxq + 15) / 16 * 16, ak = (maxk + 15) / 16 * 16;
  const int64_t bq = std::min<int64_t>(256, aq), bk = std::min<int64_t>(128, ak), bd = d;
  const int64_t nr = std::min<int64_t>(8, (maxk + bk - 1) / bk);
  optiling::FlashAttentionTrainingGeneralTilingData t;
  auto &ip = t.inputParams;
  ip.set_bSize(batch);
  ip.set_n2Size(n2);
  ip.set_gSize(g);
  ip.set_s1Size(maxq);
  ip.set_s2Size(maxk);
  ip.set_alignedS2(ak);
  ip.set_dSize(d);
  ip.set_keepProb(1.0f);
  ip.set_scaleValue(scale);
  ip.set_preTokens(maxk);
  ip.set_nextTokens(causal ? 0 : INT_MAX);
  ip.set_layoutType(4);
  ip.set_attenMaskShapeType(2);
  ip.set_attenMaskDataType(1);
  ip.set_attenMaskCompressMode(causal ? 2 : 0);
  ip.set_implMode(0);
  ip.set_sparseType(causal ? 7 : 0);
  ip.set_attenMaskS2Size(2048);
  auto &cp = t.coreParams;
  cp.set_s1BaseSize(bq);
  cp.set_s1BaseTailSize(maxq % bq ? maxq % bq : bq);
  cp.set_s1OuterSize((maxq + bq - 1) / bq);
  cp.set_s2BaseSize(bk);
  cp.set_s2BaseTailSize(maxk % bk ? maxk % bk : bk);
  cp.set_s2OuterSize(((maxk + bk - 1) / bk + nr - 1) / nr);
  cp.set_nRatio(nr);
  cp.set_dBaseSize(bd);
  cp.set_dBaseTailSize(bd);
  cp.set_dOuterSize(1);
  const auto vec2 = std::min<int64_t>(512, 8192 / d);
  cp.set_s1Vec2BaseSize(vec2);
  cp.set_s1Vec2BaseTailSize(maxq % vec2);
  cp.set_bBaseSize(1);
  cp.set_bBaseTailSize(1);
  cp.set_bOuterSize(batch);
  cp.set_n2BaseSize(1);
  cp.set_n2BaseTailSize(1);
  cp.set_n2OuterSize(n2);
  cp.set_gBaseSize(1);
  cp.set_gBaseTailSize(1);
  cp.set_gOuterSize(g);
  cp.set_s1SparseValidSize(maxk);
  cp.set_s2SparseValidSize(causal ? 0 : maxk);
  t.multiCoreParams.set_coreNum(cores.cube * 2);
  auto &ts = t.tensorSizeParams;
  ts.set_bmm1ResUbSize(bq * bk);
  ts.set_attenMaskUbSize(bq * bk);
  ts.set_castUbSize(bq * std::max(bk, bd));
  ts.set_softmaxMaxUbSize(bq * 8);
  ts.set_softmaxSumUbSize(bq * 8);
  ts.set_softmaxExpUbSize(bq * 8);
  ts.set_apiTmpBufferBytes(2 * bq * bk * 4);
  ts.set_bmm2ResUbSize(bq * bd);
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  uint64_t l1 = 0, l0c = 0, l0a = 0, l0b = 0, ub = 0;
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  matmul_tiling::PlatformInfo info;
  info.l1Size = l1;
  info.l0CSize = l0c;
  info.l0ASize = l0a;
  info.l0BSize = l0b;
  info.ubSize = ub;
  info.socVersion = platform_ascendc::SocVersion::ASCEND910B;
  const auto dtype = q.scalar_type() == at::kBFloat16 ? matmul_tiling::DataType::DT_BF16
                                                      : matmul_tiling::DataType::DT_FLOAT16;
  for (int i = 0; i < 2; ++i) {
    matmul_tiling::MatmulApiTiling mm(info);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dtype, i == 0);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT);
    if (!i) {
      mm.SetShape(std::min(bq, maxq), std::min(bk * nr, maxk), d);
      mm.SetOrgShape(maxq, bk * nr, n1 * d, n2 * d);
    } else {
      mm.SetShape(std::min(bq, maxq), d, std::min(bk * nr, maxk));
      mm.SetOrgShape(maxq, n2 * d, std::min(bk * nr, maxk), n2 * d);
    }
    mm.SetBias(false);
    mm.SetBufferSpace(l1, l0c);
    auto &mt = i ? t.bmm2TilingData : t.bmm1TilingData;
    TORCH_CHECK(mm.GetTiling(mt) != -1, "FA tiling failure");
    mt.set_shareMode(0);
    mt.set_shareL1Size(l1);
    mt.set_shareL0CSize(l0c);
  }
  AscendC::SoftMaxFlashV2TilingFunc(ge::Shape({bq / 8, bk * 8}), 4, 4, 2 * bq * bk * 4,
                                    t.softmaxFlashTilingData, true, (bq / 8) % 8 == 0);
  const auto size = t.GetDataSize();
  // Pageable H2D copies drain the task queue even with non_blocking=true.
  auto cpu = at::zeros({static_cast<int64_t>(size)},
                       at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
  t.SaveToBuffer(cpu.data_ptr(), size);
  auto tiling = cpu.to(q.device(), true);
  const int64_t workspace_size = (3 * nr * bq * bk * 4 + 4 * bq * d * 4) * cores.vector + 16 * 1024 * 1024;
  auto workspace = at::empty({workspace_size}, q.options().dtype(at::kByte));
  auto output = at::empty_like(q, q.options().dtype(at::kFloat));
  auto mx = at::empty({q.size(0), n1, 8}, q.options().dtype(at::kFloat)), sm = at::empty_like(mx);
  for (const auto &x : {q, k, v, mask, cuq, cuk, tiling, workspace})
    record(x, ns);
  // The nonblocking copy records the pinned host buffer; release it on the submitting thread.
  // Keep storage alive without moving Python-owned Tensor metadata to the queue thread.
  const std::array<c10::Storage, 11> keep{
      q.storage(),      k.storage(),         v.storage(),      mask.storage(), cuq.storage(), cuk.storage(),
      tiling.storage(), workspace.storage(), output.storage(), mx.storage(),   sm.storage()};
  const std::array<void *, 11> args{q.data_ptr(),   k.data_ptr(),         v.data_ptr(),      mask.data_ptr(),
                                    cuq.data_ptr(), cuk.data_ptr(),       output.data_ptr(), mx.data_ptr(),
                                    sm.data_ptr(),  workspace.data_ptr(), tiling.data_ptr()};
  auto launch = q.scalar_type() == at::kBFloat16
                    ? (causal ? aclrtlaunch_mojo_a2_fa_fwd_bf16_1 : aclrtlaunch_mojo_a2_fa_fwd_bf16_0)
                    : (causal ? aclrtlaunch_mojo_a2_fa_fwd_fp16_1 : aclrtlaunch_mojo_a2_fa_fwd_fp16_0);
  submit("mojo_a2_flash_attention_forward", [args, keep, cores, stream, launch]() {
    mojo::native::CheckAcl(launch(cores.cube, stream, args[0], args[1], args[2], args[3], args[4], args[5],
                                  args[6], args[7], args[8], args[9], args[10]),
                           "FA forward");
  });
  return {output, mx, sm};
}

} // namespace mojo::native::a2
