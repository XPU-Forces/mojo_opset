#include "../common/rope_host.h"
#include "host.h"
#include "aclrtlaunch_mojo_a2_rope_backward_bf16.h"
#include "aclrtlaunch_mojo_a2_rope_backward_bf16_fp32_table.h"
#include "aclrtlaunch_mojo_a2_rope_backward_fp16.h"
#include "aclrtlaunch_mojo_a2_rope_backward_fp16_fp32_table.h"
#include "aclrtlaunch_mojo_a2_rope_backward_fp32.h"

namespace mojo::native::a2 {
void RopeBackward(const at::Tensor &input, const at::Tensor &sin, const at::Tensor &cos, int64_t rotary_dim) {
  validate_rope(input, sin, cos, rotary_dim);
  TORCH_CHECK(sin.size(0) >= input.size(0), "tables must cover all input tokens");
  if (!input.numel())
    return;
  c10::DeviceGuard guard(input.device());
  const auto npu_stream = c10_npu::getCurrentNPUStream(input.get_device());
  const auto stream = npu_stream.stream(false);
  uint32_t cores;
  auto tiling = make_rope_tiling(input, rotary_dim, cores);
  record(input, npu_stream);
  record(sin, npu_stream);
  record(cos, npu_stream);
  submit("mojo_a2_rope_bwd", [input, sin, cos, tiling, cores, stream]() mutable {
    if (input.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16) {
      CheckAcl(aclrtlaunch_mojo_a2_rope_backward_bf16(cores, stream, input.data_ptr(), sin.data_ptr(),
                                                      cos.data_ptr(), &tiling),
               "A2 rope backward");
      return;
    }
    if (input.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kFloat) {
      CheckAcl(aclrtlaunch_mojo_a2_rope_backward_bf16_fp32_table(cores, stream, input.data_ptr(),
                                                                 sin.data_ptr(), cos.data_ptr(), &tiling),
               "A2 rope backward");
      return;
    }
    if (input.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf) {
      CheckAcl(aclrtlaunch_mojo_a2_rope_backward_fp16(cores, stream, input.data_ptr(), sin.data_ptr(),
                                                      cos.data_ptr(), &tiling),
               "A2 rope backward");
      return;
    }
    if (input.scalar_type() == at::kHalf && sin.scalar_type() == at::kFloat) {
      CheckAcl(aclrtlaunch_mojo_a2_rope_backward_fp16_fp32_table(cores, stream, input.data_ptr(),
                                                                 sin.data_ptr(), cos.data_ptr(), &tiling),
               "A2 rope backward");
      return;
    }
    if (input.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat) {
      CheckAcl(aclrtlaunch_mojo_a2_rope_backward_fp32(cores, stream, input.data_ptr(), sin.data_ptr(),
                                                      cos.data_ptr(), &tiling),
               "A2 rope backward");
      return;
    }
    TORCH_CHECK(false, "unsupported A2 rope specialization");
  });
}
} // namespace mojo::native::a2
