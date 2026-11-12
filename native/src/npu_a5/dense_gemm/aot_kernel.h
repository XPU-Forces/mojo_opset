#pragma once
#include <cstdint>
#include <kernel/common.h>
namespace mojo_gemm {
struct Args {
    uintptr_t a, b, d, sfa, sfb;
    uint64_t sa, sb, sd;
    uint32_t m, n, k;
    deep_gemm::epilogue::Args epilogue;
};
struct ScaleArgs {
    uintptr_t src, dst;
    uint64_t stride, batch_stride;
    uint32_t src_m, src_k, dst_m, dst_k;
    int key;
};
}
void mojo_gemm_transform_sf(const mojo_gemm::ScaleArgs&, uint32_t, void*);
