#pragma once

#include <cstdint>

#include "kernel_operator.h"

// #define XPU_OPS_ENABLE_DEBUG_LOG

#ifndef XPU_OPS_DEBUG_PRINT
#if defined(XPU_OPS_ENABLE_DEBUG_LOG)
#define XPU_OPS_DEBUG_PRINT(...)                                                                                  \
  do {                                                                                                            \
    AscendC::printf("[Core %d/%d] ", AscendC::GetBlockIdx(), AscendC::GetBlockNum() * AscendC::GetSubBlockNum()); \
    AscendC::printf(__VA_ARGS__);                                                                                 \
  } while (0)
#else
#define XPU_OPS_DEBUG_PRINT(...) ((void)0)
#endif
#endif
