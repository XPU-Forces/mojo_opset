#include "../common/swiglu_host.h"
#include "host.h"
#include "aclrtlaunch_mojo_a2_swiglu_backward_bf16.h"
#include "aclrtlaunch_mojo_a2_swiglu_backward_fp16.h"
#include "aclrtlaunch_mojo_a2_swiglu_backward_fp32.h"

namespace mojo::native::a2 {
std::tuple<at::Tensor, at::Tensor> SwigluBackward(const at::Tensor &grad, const at::Tensor &a,
                                                  const at::Tensor &b) {
  validate_swiglu(a, b);
  validate_swiglu(a, grad);
  auto da = at::empty_like(a), db = at::empty_like(b);
  if (!a.numel())
    return {da, db};
  c10::DeviceGuard guard(a.device());
  const auto npu_stream = c10_npu::getCurrentNPUStream(a.get_device());
  const auto stream = npu_stream.stream(false);
  const uint32_t cores = core_counts(a.get_device()).vector;
  record(a, npu_stream);
  record(b, npu_stream);
  record(grad, npu_stream);
  const uint64_t elements = a.numel();
  submit("mojo_a2_swiglu", [a, b, grad, da, db, elements, stream, cores]() {
    aclError status;
    if (a.scalar_type() == at::kBFloat16) {
      status = aclrtlaunch_mojo_a2_swiglu_backward_bf16(
          cores, stream, a.data_ptr(), b.data_ptr(), grad.data_ptr(), da.data_ptr(), db.data_ptr(), elements);
    } else if (a.scalar_type() == at::kFloat) {
      status = aclrtlaunch_mojo_a2_swiglu_backward_fp32(
          cores, stream, a.data_ptr(), b.data_ptr(), grad.data_ptr(), da.data_ptr(), db.data_ptr(), elements);
    } else {
      status = aclrtlaunch_mojo_a2_swiglu_backward_fp16(
          cores, stream, a.data_ptr(), b.data_ptr(), grad.data_ptr(), da.data_ptr(), db.data_ptr(), elements);
    }
    CheckAcl(status, "A2 SwiGLU");
  });
  return {da, db};
}
} // namespace mojo::native::a2
