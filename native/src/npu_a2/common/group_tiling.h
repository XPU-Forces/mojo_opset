#pragma once
#include <cstdint>
struct GMMTrainBaseParams {
  uint32_t groupNum, coreNum, activeType, ubBaseK, ubBaseN, ubCalSize, ubRestBytes;
  int32_t groupType;
  uint32_t singleN;
  uint64_t workspaceSize;
};
struct GroupTiling {
  GMMTrainBaseParams base;
  // Serialized CANN TCubeTiling, passed inline in kernel arguments. The host
  // and device both verify its size; no OPP registration or device copy.
  alignas(8) uint8_t matmul[512];
};
