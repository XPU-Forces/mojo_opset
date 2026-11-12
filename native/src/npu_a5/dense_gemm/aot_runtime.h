#pragma once
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <host/checks.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace deep_gemm {
inline uint32_t get_num_cube_cores() {
    int device = 0;
    int64_t cores = 0;
    DG_HOST_ASSERT(aclrtGetDevice(&device) == ACL_SUCCESS);
    DG_HOST_ASSERT(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &cores) == ACL_SUCCESS);
    DG_HOST_ASSERT(cores > 0);
    return static_cast<uint32_t>(cores);
}
inline aclrtStream get_current_npu_stream() { return c10_npu::getCurrentNPUStream().stream(); }
} // namespace deep_gemm
