#include "../common/swiglu_host.h"
#include "host.h"
#include "aclrtlaunch_mojo_a2_swiglu_forward_bf16.h"
#include "aclrtlaunch_mojo_a2_swiglu_forward_fp16.h"
#include "aclrtlaunch_mojo_a2_swiglu_forward_fp32.h"

namespace mojo::native::a2 {
at::Tensor SwigluForward(const at::Tensor &a, const at::Tensor &b) {
  validate_swiglu(a, b);
  auto output = at::empty_like(a);
  if (!a.numel())
    return output;
  c10::DeviceGuard guard(a.device());
  const auto npu_stream = c10_npu::getCurrentNPUStream(a.get_device());
  const auto stream = npu_stream.stream(false);
  const uint32_t cores = core_counts(a.get_device()).vector;
  record(a, npu_stream);
  record(b, npu_stream);
  const uint64_t elements = a.numel();
  submit("mojo_a2_swiglu", [a, b, output, elements, stream, cores]() {
    aclError status;
    if (a.scalar_type() == at::kBFloat16) {
      status = aclrtlaunch_mojo_a2_swiglu_forward_bf16(cores, stream, a.data_ptr(), b.data_ptr(), nullptr,
                                                       output.data_ptr(), nullptr, elements);
    } else if (a.scalar_type() == at::kFloat) {
      status = aclrtlaunch_mojo_a2_swiglu_forward_fp32(cores, stream, a.data_ptr(), b.data_ptr(), nullptr,
                                                       output.data_ptr(), nullptr, elements);
    } else {
      status = aclrtlaunch_mojo_a2_swiglu_forward_fp16(cores, stream, a.data_ptr(), b.data_ptr(), nullptr,
                                                       output.data_ptr(), nullptr, elements);
    }
    CheckAcl(status, "A2 SwiGLU");
  });
  return output;
}
} // namespace mojo::native::a2
