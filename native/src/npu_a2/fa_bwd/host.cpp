#include "host.h"
#include "host_tiling.h"
#include "../common/fa_host.h"
#include <array>
#include <tiling/platform/platform_ascendc.h>
#include "aclrtlaunch_mojo_a2_fa_bwd_bf16.h"
#include "aclrtlaunch_mojo_a2_fa_bwd_fp16.h"
namespace mojo::native::a2 {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
FlashAttentionBackward(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &dy,
                       const at::Tensor &mask, const at::Tensor &out, const at::Tensor &mx,
                       const at::Tensor &sm, const at::Tensor &cuq, const at::Tensor &cuk, double scale,
                       bool causal) {
  validate_fa(q, k, v, mask, cuq, cuk, causal);
  for (const auto &x : {dy, out, mx, sm}) {
    tensor(x, "FA gradient/cache");
    TORCH_CHECK(x.device() == q.device(), "FA gradient/cache must share the input device");
  }
  TORCH_CHECK(dy.sizes() == q.sizes() && dy.scalar_type() == q.scalar_type(), "FA gradient must match q");
  TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == at::kFloat,
              "FA backward requires the FP32 forward cache");
  TORCH_CHECK(mx.dim() == 3 && mx.size(0) == q.size(0) && mx.size(1) == q.size(1) && mx.size(2) == 8 &&
                  sm.sizes() == mx.sizes() && mx.scalar_type() == at::kFloat &&
                  sm.scalar_type() == at::kFloat,
              "FA statistics must be FP32 [tokens,heads,8]");
  c10::DeviceGuard guard(q.device());
  const auto ns = c10_npu::getCurrentNPUStream(q.get_device());
  const auto stream = ns.stream(false);
  const auto cores = core_counts(q.get_device());
  const int64_t d = q.size(2), n1 = q.size(1), n2 = k.size(1), t1 = q.size(0), t2 = k.size(0);
  optiling::FlashAttentionTrainingGradTilingData t;
  t.set_coreNum(cores.vector);
  t.set_scaleValue(scale);
  t.set_keepProb(1.0);
  t.set_preTockens(t2);
  t.set_nextTockens(causal ? 0 : INT_MAX);
  t.set_sparseMode(causal ? 3 : 0);
  t.set_dqPostAbsorb(0);
  t.set_b(cuq.numel());
  t.set_t1(t1);
  t.set_t2(t2);
  t.set_n2(n2);
  t.set_g(n1 / n2);
  t.set_d(d);
  t.set_qSize(q.numel());
  t.set_kvSize(k.numel());
  AscendC::SoftMaxTilingFunc(ge::Shape({64, 128}), 4, 33 * 1024, t.softmaxTilingData);
  const uint32_t output_len = (48 * 1024 + d - 1) / d * 8;
  AscendC::SoftMaxGradTilingFunc(ge::Shape({24 * 1024 / 4 / d, d}), 4, 40 * 1024 - output_len,
                                 t.softmaxGradTilingData, true);
  size_t offset = 16 * 1024 * 1024;
  auto aligned = [&](size_t n) { offset = (offset + n + 512) / 512 * 512; };
  t.set_sfmgWorkspaceOffset(offset);
  aligned(t1 * n1 * 8 * 4);
  t.set_mm1WorkspaceOffset(offset);
  aligned(cores.cube * 16 * 128 * 128 * 4 * 2);
  t.set_mm2WorkspaceOffset(offset);
  aligned(cores.cube * 16 * 128 * 128 * 4 * 2);
  {
    t.set_dqDetWorkspaceOffset(offset);
    offset += cores.cube * 512 * d * 4;
    t.set_dkDetWorkspaceOffset(offset);
    offset += cores.cube * 512 * d * 4;
    t.set_dvDetWorkspaceOffset(offset);
    offset += cores.cube * 512 * d * 4;
  }
  t.set_dqWorkSpaceOffset(offset);
  aligned(q.numel() * 4);
  t.set_dkWorkSpaceOffset(offset);
  aligned(k.numel() * 4);
  t.set_dvWorkSpaceOffset(offset);
  aligned(v.numel() * 4);
  offset += 16 * 1024 * 1024;
  auto cpu = at::zeros({static_cast<int64_t>(t.GetDataSize())},
                       at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
  t.SaveToBuffer(cpu.data_ptr(), t.GetDataSize());
  auto tiling = cpu.to(q.device(), true),
       workspace = at::empty({static_cast<int64_t>(offset)}, q.options().dtype(at::kByte));
  auto dq = at::empty_like(q), dk = at::empty_like(k), dv = at::empty_like(v);
  for (const auto &x : {q, k, v, dy, mask, out, mx, sm, cuq, cuk, tiling, workspace})
    record(x, ns);
  // The nonblocking copy records the pinned host buffer; release it on the submitting thread.
  // Keep storage alive without moving Python-owned Tensor metadata to the queue thread.
  const std::array<c10::Storage, 15> keep{
      q.storage(),      k.storage(),         v.storage(),  dy.storage(),  mask.storage(),
      out.storage(),    mx.storage(),        sm.storage(), cuq.storage(), cuk.storage(),
      tiling.storage(), workspace.storage(), dq.storage(), dk.storage(),  dv.storage()};
  const std::array<void *, 15> args{q.data_ptr(),    k.data_ptr(),         v.data_ptr(),     dy.data_ptr(),
                                    mask.data_ptr(), mx.data_ptr(),        sm.data_ptr(),    out.data_ptr(),
                                    cuq.data_ptr(),  cuk.data_ptr(),       dq.data_ptr(),    dk.data_ptr(),
                                    dv.data_ptr(),   workspace.data_ptr(), tiling.data_ptr()};
  auto launch =
      q.scalar_type() == at::kBFloat16 ? aclrtlaunch_mojo_a2_fa_bwd_bf16 : aclrtlaunch_mojo_a2_fa_bwd_fp16;
  submit("mojo_a2_flash_attention_backward", [args, keep, cores, stream, launch]() {
    mojo::native::CheckAcl(launch(cores.cube, stream, args[0], args[1], args[2], args[3], args[4], args[5],
                                  args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13],
                                  args[14]),
                           "FA backward");
  });
  return {dq, dk, dv};
}
} // namespace mojo::native::a2
