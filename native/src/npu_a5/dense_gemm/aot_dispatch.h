#pragma once
#include "host/config.h"
namespace deep_gemm {
void launch_aot_gemm(const GemmDesc& desc, int kind);
}
