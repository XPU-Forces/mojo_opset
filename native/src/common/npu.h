#pragma once

#include <ATen/ATen.h>
#include <acl/acl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mojo::native {

inline void CheckAcl(aclError ret, const char *what) {
  if (ret == ACL_ERROR_NONE) {
    return;
  }
  const char *recent = aclGetRecentErrMsg();
  std::string message = std::string(what) + " failed: " + std::to_string(static_cast<int>(ret));
  if (recent && recent[0]) {
    message += ", recent acl error: ";
    message += recent;
  }
  throw std::runtime_error(message);
}

// Callers validate that cumulative lengths are contiguous int32 tensors.
inline std::vector<int64_t> ReadCuLens(const at::Tensor &cu_lens) {
  auto cu_cpu = cu_lens.to(at::kCPU).contiguous();
  const auto *data = cu_cpu.data_ptr<int32_t>();
  std::vector<int64_t> result(cu_cpu.numel());
  for (int64_t i = 0; i < cu_cpu.numel(); ++i) {
    result[i] = data[i];
  }
  return result;
}

}  // namespace mojo::native
