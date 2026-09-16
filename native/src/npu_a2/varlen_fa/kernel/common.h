#pragma once

#include <cstdint>

namespace xpu_ops::kernels {

#define EPSILON 1e-5f
#define UB_ALIGNMENT_UNIT ((uint32_t)32)
#define VEC_BLOCK_BYTES ((uint32_t)32)
#define VEC_BLOCK_NUMS ((uint32_t)8)
#define VEC_BLOCK_STRIDE_DEFAULT ((uint32_t)1)
#define VEC_BLOCK_STRIDE_REPEAT ((uint32_t)0)
#define VEC_REPEAT_STRIDE_DEFAULT ((uint32_t)8)
#define CUBE_BLOCK_BYTES ((uint32_t)512)
#define VEC_REPEAT_MAX_BYTES ((uint32_t)256)
#define VEC_REPEAT_FP32_ELEMENTS ((uint32_t)64)
#define UB_TOTAL_SIZE (uint32_t(192 * 1024))
#define VEC_BLK_FP32_ELEMENTS (uint32_t(8))
#define VEC_ALIGN_NUM (uint32_t(32))
}  // namespace xpu_ops::kernels
