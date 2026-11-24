#include "../common/group_gemm_host.h"
#include "host.h"
#include "aclrtlaunch_mojo_a2_group_forward_bf16.h"
#include "aclrtlaunch_mojo_a2_group_forward_fp16.h"
#include "aclrtlaunch_mojo_a2_group_dgrad_bf16.h"
#include "aclrtlaunch_mojo_a2_group_dgrad_fp16.h"
namespace mojo::native::a2 {
at::Tensor GroupGemm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &counts,
                     bool transpose_weight) {
  validate_group(x, w, counts, false);
  c10::DeviceGuard guard(x.device());
  const int64_t m = x.size(0), n = w.size(transpose_weight ? 1 : 2), k = x.size(1);
  TORCH_CHECK(w.size(transpose_weight ? 2 : 1) == k, "group GEMM contraction dimensions differ");
  auto out = !k ? at::zeros({m, n}, x.options()) : at::empty({m, n}, x.options());
  if (!m || !n || !k)
    return out;
  const auto ns = c10_npu::getCurrentNPUStream(x.get_device());
  const auto stream = ns.stream(false);
  const auto cores = core_counts(x.get_device()).cube;
  auto tiling = make_group_tiling(m, n, k, counts.numel(), x.scalar_type(), false, cores);
  for (const auto &t : {x, w, counts, out})
    record(t, ns);
  submit("mojo_a2_group_gemm", [x, w, counts, out, tiling, cores, stream, transpose_weight]() mutable {
    if (x.scalar_type() == at::kBFloat16 && transpose_weight) {
      CheckAcl(aclrtlaunch_mojo_a2_group_forward_bf16(cores, stream, x.data_ptr(), w.data_ptr(),
                                                      counts.data_ptr(), out.data_ptr(), &tiling),
               "A2 grouped matmul launch");
      return;
    }
    if (x.scalar_type() == at::kHalf && transpose_weight) {
      CheckAcl(aclrtlaunch_mojo_a2_group_forward_fp16(cores, stream, x.data_ptr(), w.data_ptr(),
                                                      counts.data_ptr(), out.data_ptr(), &tiling),
               "A2 grouped matmul launch");
      return;
    }
    if (x.scalar_type() == at::kBFloat16 && !transpose_weight) {
      CheckAcl(aclrtlaunch_mojo_a2_group_dgrad_bf16(cores, stream, x.data_ptr(), w.data_ptr(),
                                                    counts.data_ptr(), out.data_ptr(), &tiling),
               "A2 grouped matmul launch");
      return;
    }
    if (x.scalar_type() == at::kHalf && !transpose_weight) {
      CheckAcl(aclrtlaunch_mojo_a2_group_dgrad_fp16(cores, stream, x.data_ptr(), w.data_ptr(),
                                                    counts.data_ptr(), out.data_ptr(), &tiling),
               "A2 grouped matmul launch");
      return;
    }
    TORCH_CHECK(false, "unsupported grouped matmul specialization");
  });
  return out;
}
} // namespace mojo::native::a2
