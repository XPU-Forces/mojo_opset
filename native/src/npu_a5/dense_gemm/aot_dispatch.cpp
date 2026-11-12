#include "aot_kernel.h"
#include "host/cost_model.h"
#include "aot_runtime.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>

namespace deep_gemm {
struct AotEntry {
    int key;
    const char* tag;
    bool fallback;
    GemmConfig config;
    void (*launch)(const mojo_gemm::Args&, uint32_t, void*);
};
struct AotTuning {
    const char* soc;
    uint32_t cores, key, m, n, k;
    size_t entry;
};
}
#include "registry.inc"
namespace deep_gemm {
#include "tuning.inc"

void launch_aot_gemm(const GemmDesc& d, int kind) {
    const bool bias = dynamic_cast<const BiasEpilogue*>(d.epilogue_class.get()) != nullptr;
    const bool alpha = dynamic_cast<const AlphaEpilogue*>(d.epilogue_class.get()) != nullptr;
    DG_HOST_ASSERT(bias || alpha || dynamic_cast<const IdentityEpilogue*>(d.epilogue_class.get()));
    const int mode = bias ? 4 : (alpha ? 2 : 0) + static_cast<int>(d.acc);
    const int key = kind * 40 + static_cast<int>(d.major_a == Major::MN) * 20
        + static_cast<int>(d.major_b == Major::MN) * 10 + static_cast<int>(d.cd_dtype == at::kFloat) * 5 + mode;
    const auto dtype = get_gemm_dtype(d);
    const auto candidates = std::span(aot_entries + aot_key_offsets[key], aot_key_offsets[key + 1] - aot_key_offsets[key]);
    const auto legal = [&](const AotEntry& entry) {
        const auto& c = entry.config;
        return entry.key == key && (!c.shape_n || c.shape_n == d.n) && (!c.shape_k || c.shape_k == d.k)
            && (!c.num_cores || c.num_cores == d.num_cores)
            && (!(c.single_m_block || c.resident_a) || d.m <= c.block_m)
            && (!c.resident_b || d.n <= c.block_n)
            && (!c.direct_store || can_direct_store(d, dtype, c));
    };
    const AotEntry* best = nullptr;
    const char* source = "model";
    // Developer-only override for offline profiling; cannot bypass layout or safety checks.
    if (const char* tag = std::getenv("MOJO_GEMM_FORCE_CONFIG")) {
        for (const auto& entry : candidates)
            if (legal(entry) && std::strcmp(tag, entry.tag) == 0) { best = &entry; break; }
        DG_HOST_ASSERT(best, "Requested AOT configuration is unavailable or unsafe for this GEMM: {}", tag);
        source = "forced";
    }
    // Measurements apply only to the exact SKU/core count, metadata, shape and
    // tightly packed matrix strides. Unmeasured/strided inputs use the cost model.
    const auto inner_a = d.major_a == Major::K ? d.k : d.m;
    const auto inner_b = d.major_b == Major::K ? d.k : d.n;
    const bool tight = d.outer_stride_a == inner_a / (kind == 3 ? 2 : 1)
        && d.outer_stride_b == inner_b / (kind >= 2 ? 2 : 1) && d.outer_stride_d == d.n;
    if (!best && tight && !std::getenv("MOJO_GEMM_DISABLE_TUNING")) {
        const char* soc = aclrtGetSocName();
        for (const auto& tuned : aot_tuning) {
            if (tuned.soc && soc && tuned.key == key && tuned.cores == d.num_cores &&
                tuned.m == d.m && tuned.n == d.n && tuned.k == d.k && std::strcmp(soc, tuned.soc) == 0) {
                const auto& entry = aot_entries[tuned.entry];
                if (legal(entry)) { best = &entry; source = "tuned"; }
                break;
            }
        }
    }
    if (!best) {
        auto best_cost = std::numeric_limits<int64_t>::max();
        for (const auto& entry : candidates) {
            if (!entry.fallback || !legal(entry)) continue;
            const auto cost = get_gemm_k_cost(d, dtype, entry.config);
            if (!best || cost < best_cost) { best = &entry; best_cost = cost; }
        }
    }
    DG_HOST_ASSERT(best != nullptr, "No AOT GEMM entry for key {}", key);
    if (std::getenv("MOJO_GEMM_PRINT_CONFIG"))
        fmt::print("AOT key={} shape={},{},{} tag={} source={} {}\n", key, d.m, d.n, d.k, best->tag, source, best->config);
    mojo_gemm::Args args{reinterpret_cast<uintptr_t>(d.a_ptr), reinterpret_cast<uintptr_t>(d.b_ptr),
        reinterpret_cast<uintptr_t>(d.d_ptr), reinterpret_cast<uintptr_t>(d.sfa_ptr), reinterpret_cast<uintptr_t>(d.sfb_ptr),
        d.outer_stride_a, d.outer_stride_b, d.outer_stride_d, d.m, d.n, d.k,
        d.epilogue_class->make_epilogue_operator_args(d.n)};
    best->launch(args, d.num_cores, get_current_npu_stream());
}
}
