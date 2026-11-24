// Adapted from data/xpu_ops commit 8452b29b; see group_gemm_SOURCE.json.
#pragma once
#include "common.h"
#include "data_copy_utils.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace mojo::native::a2 {
namespace kernels {

struct GroupGemmEpilogueNone {};

} // namespace kernels
} // namespace mojo::native::a2
