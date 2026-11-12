#pragma once

#include "structs.h"

namespace mojo::native::a5 {
void LaunchTopK(const TopkSelectArgs& args);
void LaunchRescanTopK(const TopkSelectArgs& args, uint32_t cores);
}
