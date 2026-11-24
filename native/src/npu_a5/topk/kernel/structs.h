#pragma once

#include <cstdint>

#include <acl/acl_base_rt.h>

// `__gm__` qualifies the pointers that live in Ascend global memory. It only
// exists in the device pass of a `.asc` translation unit and is empty in every
// other pass (host pass of a `.asc`, plain `.cpp`).
#if defined(__NPU_DEVICE__)
#define TOPK_SELECT_GM __gm__
#else
#define TOPK_SELECT_GM
#endif

using topk_select_stream_t = aclrtStream;

static constexpr uint32_t INPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes
static constexpr uint32_t OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes

static constexpr uint32_t MAX_VOCAB_SIZE = 2 * 1024 * 1024;

struct TopkSelectArgs {
    uint32_t batch_size;
    uint32_t vocab_size;
    uint32_t topk;

    // The typed kernel casts the input and output pointers once on entry.
    TOPK_SELECT_GM void* input;
    TOPK_SELECT_GM void* output_value;
    TOPK_SELECT_GM void* output_index;
    TOPK_SELECT_GM int* begin_ptr;
    TOPK_SELECT_GM int* end_ptr;
    TOPK_SELECT_GM int* output_idx_offset;

    // All strides are in number of elements, not bytes
    uint64_t stride_input_batch;
    uint64_t stride_output_value_batch;
    uint64_t stride_output_index_batch;

    bool sorted_value;
    bool sorted_index;
    bool return_value;
    int idx_oob_fill_value;
    float value_oob_fill_value;
    bool abort_when_nan_found;

    // BF16 stores use the host-rounded fill bits.
    uint16_t value_oob_fill_bits;

    topk_select_stream_t stream;
};
