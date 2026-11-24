#include "../common/group_gemm_host.h"
#include "host.h"
#include "aclrtlaunch_mojo_a2_group_wgrad_bf16.h"
#include "aclrtlaunch_mojo_a2_group_wgrad_fp16.h"
namespace mojo::native::a2 {
at::Tensor GroupGemmWgrad(const at::Tensor &x, const at::Tensor &w, const at::Tensor &counts) {
  validate_group(x, w, counts, true);
  c10::DeviceGuard guard(x.device());
  const int64_t m = x.size(1), n = w.size(1), k = x.size(0);
  TORCH_CHECK(w.size(0) == k, "group GEMM contraction dimensions differ");
  auto out = at::zeros({counts.numel(), m, n}, x.options());
  if (!m || !n || !k)
    return out;
  const auto ns = c10_npu::getCurrentNPUStream(x.get_device());
  const auto stream = ns.stream(false);
  const auto cores = core_counts(x.get_device()).cube;
  auto tiling = make_group_tiling(m, n, k, counts.numel(), x.scalar_type(), true, cores);
  for (const auto &t : {x, w, counts, out})
    record(t, ns);
  submit("mojo_a2_group_wgrad", [x, w, counts, out, tiling, cores, stream]() mutable {
    if (x.scalar_type() == at::kBFloat16) {
      CheckAcl(aclrtlaunch_mojo_a2_group_wgrad_bf16(cores, stream, x.data_ptr(), w.data_ptr(),
                                                    counts.data_ptr(), out.data_ptr(), &tiling),
               "A2 grouped matmul launch");
      return;
    }
    if (x.scalar_type() == at::kHalf) {
      CheckAcl(aclrtlaunch_mojo_a2_group_wgrad_fp16(cores, stream, x.data_ptr(), w.data_ptr(),
                                                    counts.data_ptr(), out.data_ptr(), &tiling),
               "A2 grouped matmul launch");
      return;
    }
    TORCH_CHECK(false, "unsupported grouped matmul specialization");
  });
  return out;
}
} // namespace mojo::native::a2
