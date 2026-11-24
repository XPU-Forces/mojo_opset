#pragma once

#include "kernel_operator.h"
#include "kernel/task_scheduler.h"

namespace SWA_v5::Backward {

static constexpr uint32_t FINALIZE_CHUNK_ELEMS = 16384U;
static constexpr uint32_t FINALIZE_TILE_ROWS = 128U;
static constexpr uint32_t FINALIZE_DIM = 128U;
static constexpr uint32_t FINALIZE_FP32_ROWS = 64U;
static constexpr float ATTENTION_SCALE = 0.08838834764831845F;

class BackwardWorkspaceZero {
public:
    __aicore__ inline void init() {
        pipe_.InitBuffer(fp32_queue_, 1U, FINALIZE_CHUNK_ELEMS * sizeof(float));
    }

    __aicore__ inline void zero(
        GM_ADDR workspace, uint64_t start_offset,
        uint64_t total_elems, uint32_t workers) {
        AscendC::GlobalTensor<float> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

        const uint64_t chunks =
            (total_elems + FINALIZE_CHUNK_ELEMS - 1U) / FINALIZE_CHUNK_ELEMS;
        const uint32_t worker = AscendC::GetBlockIdx();
        for (uint64_t chunk = worker; chunk < chunks; chunk += workers) {
            const uint64_t offset = chunk * FINALIZE_CHUNK_ELEMS;
            const uint32_t count = static_cast<uint32_t>(
                total_elems - offset < FINALIZE_CHUNK_ELEMS ?
                    total_elems - offset : FINALIZE_CHUNK_ELEMS);
            AscendC::LocalTensor<float> zeros = fp32_queue_.AllocTensor<float>();
            AscendC::Duplicate(zeros, 0.0F, static_cast<int32_t>(count));
            fp32_queue_.EnQue(zeros);
            zeros = fp32_queue_.DeQue<float>();
            AscendC::DataCopyExtParams copy{
                1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            AscendC::DataCopyPad(dst[start_offset + offset], zeros, copy);
            fp32_queue_.FreeTensor(zeros);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    template<class CuLens>
    __aicore__ inline void zero_global_dkv_partials(
        GM_ADDR workspace, uint64_t partial_section_offset,
        uint64_t plane_elems, uint32_t plane_count,
        CuLens& cu_q_lens, CuLens& cu_kv_lens, uint32_t batch,
        uint32_t num_kv_heads, uint32_t kv_stride,
        uint32_t window_left, uint32_t global_window_size,
        uint32_t workers) {
        AscendC::GlobalTensor<float> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

        constexpr uint32_t rows_per_job =
            FINALIZE_CHUNK_ELEMS / FINALIZE_DIM;
        constexpr uint32_t grad_kinds = 2U;
        const uint32_t worker = AscendC::GetBlockIdx();
        uint64_t global_job_base = 0U;
        uint64_t partial_row_base = 0U;
        for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
            const uint32_t q_len =
                cu_q_lens.GetValue(batch_idx + 1U) -
                cu_q_lens.GetValue(batch_idx);
            const uint32_t kv_len =
                cu_kv_lens.GetValue(batch_idx + 1U) -
                cu_kv_lens.GetValue(batch_idx);
            const uint32_t global_rows = min_u32(
                global_window_size,
                q_len > 0U && kv_len >= q_len &&
                        kv_len > window_left ?
                    kv_len - window_left : 0U);
            const uint32_t row_parts =
                ceil_div(global_rows, rows_per_job);
            const uint64_t batch_jobs =
                static_cast<uint64_t>(row_parts) * num_kv_heads *
                grad_kinds * plane_count;
            const uint64_t first_job =
                (static_cast<uint64_t>(worker) + workers -
                    global_job_base % workers) % workers;
            for (uint64_t job = first_job; job < batch_jobs;
                 job += workers) {
                uint64_t remaining = job;
                const uint32_t row_part =
                    static_cast<uint32_t>(remaining % row_parts);
                remaining /= row_parts;
                const uint32_t kv_head =
                    static_cast<uint32_t>(remaining % num_kv_heads);
                remaining /= num_kv_heads;
                const uint32_t grad_kind =
                    static_cast<uint32_t>(
                        remaining % grad_kinds);
                const uint32_t plane =
                    static_cast<uint32_t>(
                        remaining / grad_kinds);
                const uint32_t row_offset = row_part * rows_per_job;
                const uint32_t rows =
                    min_u32(rows_per_job, global_rows - row_offset);
                const uint32_t count = rows * FINALIZE_DIM;
                AscendC::LocalTensor<float> zeros =
                    fp32_queue_.AllocTensor<float>();
                AscendC::Duplicate(
                    zeros, 0.0F, static_cast<int32_t>(count));
                fp32_queue_.EnQue(zeros);
                zeros = fp32_queue_.DeQue<float>();
                const uint64_t dst_offset =
                    partial_section_offset +
                    2U * static_cast<uint64_t>(plane) * plane_elems +
                    static_cast<uint64_t>(grad_kind) * plane_elems +
                    (partial_row_base + row_offset) * kv_stride +
                    static_cast<uint64_t>(kv_head) * FINALIZE_DIM;
                AscendC::DataCopyExtParams copy{
                    static_cast<uint16_t>(rows),
                    FINALIZE_DIM *
                        static_cast<uint32_t>(sizeof(float)),
                    0,
                    static_cast<int64_t>(
                        (kv_stride - FINALIZE_DIM) * sizeof(float)),
                    0U};
                AscendC::DataCopyPad(dst[dst_offset], zeros, copy);
                fp32_queue_.FreeTensor(zeros);
            }
            partial_row_base +=
                static_cast<uint64_t>(row_parts) * FINALIZE_TILE_ROWS;
            global_job_base += batch_jobs;
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void destroy() {
        pipe_.Destroy();
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> fp32_queue_;
};

class BackwardFinalize {
public:
    __aicore__ inline void init() {
        pipe_.InitBuffer(fp32_queue_, 1U, FINALIZE_CHUNK_ELEMS * sizeof(float));
        pipe_.InitBuffer(reduce_input_queue_, 1U, FINALIZE_CHUNK_ELEMS * sizeof(float));
        pipe_.InitBuffer(bf16_queue_, 1U, FINALIZE_CHUNK_ELEMS * sizeof(bfloat16_t));
        pipe_.InitBuffer(
            bf16_input_queue_, 2U, FINALIZE_CHUNK_ELEMS * sizeof(bfloat16_t));
    }

    __aicore__ inline void destroy() {
        pipe_.Destroy();
    }

    template<class PartialT>
    __aicore__ inline void reduce_cast_dkv_colors(
        AscendC::GlobalTensor<PartialT>& workspace, uint64_t partial_section_offset,
        uint64_t dk_storage_elems, AscendC::GlobalTensor<bfloat16_t>& dk_dst,
        AscendC::GlobalTensor<bfloat16_t>& dv_dst,
        AscendC::GlobalTensor<uint32_t>& cu_q_lens,
        AscendC::GlobalTensor<uint32_t>& cu_kv_lens,
        uint32_t batch, uint32_t num_kv_heads, uint32_t kv_stride,
        uint32_t window_left, int32_t min_delta, int32_t max_delta, float dk_scale,
        uint32_t workers) {
        constexpr uint32_t rows_per_job =
            AscendC::IsSameType<PartialT, bfloat16_t>::value ?
            FINALIZE_CHUNK_ELEMS / FINALIZE_DIM :
            FINALIZE_FP32_ROWS;
        constexpr uint32_t row_parts =
            (FINALIZE_TILE_ROWS + rows_per_job - 1U) /
            rows_per_job;
        const uint32_t worker = AscendC::GetBlockIdx();
        uint64_t global_job_base = 0U;
        for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
            const uint32_t q_begin = cu_q_lens.GetValue(batch_idx);
            const uint32_t q_end = cu_q_lens.GetValue(batch_idx + 1U);
            const uint32_t kv_begin = cu_kv_lens.GetValue(batch_idx);
            const uint32_t kv_end = cu_kv_lens.GetValue(batch_idx + 1U);
            const uint32_t q_len = q_end - q_begin;
            const uint32_t kv_len = kv_end - kv_begin;
            const uint32_t q_tiles = ceil_div(q_len, FINALIZE_TILE_ROWS);
            const uint32_t kv_tiles = ceil_div(kv_len, FINALIZE_TILE_ROWS);
            const uint32_t kv_prefix = kv_len - q_len;
            const uint32_t prefix_tile = kv_prefix / FINALIZE_TILE_ROWS;
            const uint64_t batch_jobs =
                static_cast<uint64_t>(kv_tiles) * num_kv_heads * 2U *
                row_parts;
            const uint64_t first_job =
                (static_cast<uint64_t>(worker) + workers -
                    global_job_base % workers) % workers;
            for (uint64_t job = first_job; job < batch_jobs; job += workers) {
                uint64_t remaining = job;
                const uint32_t row_half =
                    static_cast<uint32_t>(remaining % row_parts);
                remaining /= row_parts;
                const uint32_t kv_head =
                    static_cast<uint32_t>(remaining % num_kv_heads);
                remaining /= num_kv_heads;
                const uint32_t kv_tile =
                    static_cast<uint32_t>(remaining % kv_tiles);
                const uint32_t grad_kind =
                    static_cast<uint32_t>(remaining / kv_tiles);
                const uint32_t tile_rows = min_u32(
                    FINALIZE_TILE_ROWS,
                    kv_len - kv_tile * FINALIZE_TILE_ROWS);
                const uint32_t row_offset = row_half * rows_per_job;
                if (row_offset >= tile_rows) {
                    continue;
                }
                const uint32_t rows =
                    min_u32(rows_per_job, tile_rows - row_offset);
                const uint32_t count = rows * FINALIZE_DIM;
                const uint64_t padded_row =
                    static_cast<uint64_t>(kv_begin) +
                    static_cast<uint64_t>(batch_idx) *
                        (FINALIZE_TILE_ROWS - 1U) +
                    static_cast<uint64_t>(kv_tile) * FINALIZE_TILE_ROWS +
                    row_offset;
                const uint64_t packed_row =
                    static_cast<uint64_t>(kv_begin) +
                    static_cast<uint64_t>(kv_tile) * FINALIZE_TILE_ROWS +
                    row_offset;
                const uint64_t tile_offset =
                    padded_row * kv_stride +
                    static_cast<uint64_t>(kv_head) * FINALIZE_DIM;
                const bool contiguous = kv_stride == FINALIZE_DIM;
                AscendC::DataCopyExtParams load{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : FINALIZE_DIM) *
                        static_cast<uint32_t>(sizeof(PartialT)),
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (kv_stride - FINALIZE_DIM) * sizeof(PartialT)),
                    0, 0U};
                AscendC::DataCopyPadExtParams<PartialT> no_pad{
                    false, 0U, 0U, 0.0F};
                AscendC::LocalTensor<float> accum =
                    fp32_queue_.AllocTensor<float>();
                bool has_partial = false;
                bool has_pending_bf16 = false;
                for (int32_t delta = min_delta;
                     delta <= max_delta; ++delta) {
                    const int32_t q_tile_signed =
                        delta + static_cast<int32_t>(kv_tile) -
                        static_cast<int32_t>(prefix_tile);
                    if (q_tile_signed < 0 ||
                        static_cast<uint32_t>(q_tile_signed) >= q_tiles) {
                        continue;
                    }
                    const uint32_t q_tile =
                        static_cast<uint32_t>(q_tile_signed);
                    const uint32_t q_row = q_tile * FINALIZE_TILE_ROWS;
                    const uint32_t q_rows = min_u32(
                        FINALIZE_TILE_ROWS, q_len - q_row);
                    const KvTileRange range =
                        make_kv_tile_range<FINALIZE_TILE_ROWS, true>(
                            kv_prefix + q_row, q_rows, kv_len,
                            window_left, 0U, 0U);
                    if (kv_tile < range.first_tile ||
                        kv_tile >= range.first_tile +
                            range.local_tile_count) {
                        continue;
                    }
                    const uint64_t color =
                        static_cast<uint32_t>(delta - min_delta);
                    const uint64_t src_offset =
                        partial_section_offset +
                        color * 2U * dk_storage_elems +
                        static_cast<uint64_t>(grad_kind) *
                            dk_storage_elems +
                        tile_offset;
                    if constexpr (
                        AscendC::IsSameType<PartialT, float>::value) {
                        if (!has_partial) {
                            AscendC::DataCopyPad(
                                accum, workspace[src_offset], load, no_pad);
                            fp32_queue_.EnQue(accum);
                            accum = fp32_queue_.DeQue<float>();
                        } else {
                            AscendC::LocalTensor<float> input =
                                reduce_input_queue_.AllocTensor<float>();
                            AscendC::DataCopyPad(
                                input, workspace[src_offset], load, no_pad);
                            reduce_input_queue_.EnQue(input);
                            input = reduce_input_queue_.DeQue<float>();
                            AscendC::Add(accum, accum, input, count);
                            AscendC::PipeBarrier<PIPE_V>();
                            reduce_input_queue_.FreeTensor(input);
                        }
                    } else {
                        AscendC::LocalTensor<bfloat16_t> partial =
                            bf16_input_queue_.AllocTensor<bfloat16_t>();
                        AscendC::DataCopyPad(
                            partial, workspace[src_offset], load, no_pad);
                        bf16_input_queue_.EnQue(partial);
                        if (has_pending_bf16) {
                            partial =
                                bf16_input_queue_.DeQue<bfloat16_t>();
                            accumulate_bf16_partial(
                                accum, partial, count, has_partial);
                            bf16_input_queue_.FreeTensor(partial);
                            has_partial = true;
                        }
                        has_pending_bf16 = true;
                    }
                    if constexpr (
                        AscendC::IsSameType<PartialT, float>::value) {
                        has_partial = true;
                    }
                }
                if constexpr (
                    AscendC::IsSameType<PartialT, bfloat16_t>::value) {
                    if (has_pending_bf16) {
                        AscendC::LocalTensor<bfloat16_t> partial =
                            bf16_input_queue_.DeQue<bfloat16_t>();
                        accumulate_bf16_partial(
                            accum, partial, count, has_partial);
                        bf16_input_queue_.FreeTensor(partial);
                        has_partial = true;
                    }
                }
                if (!has_partial) {
                    AscendC::Duplicate(
                        accum, 0.0F, static_cast<int32_t>(count));
                }
                if (grad_kind == 0U && dk_scale != 1.0F) {
                    AscendC::Muls(accum, accum, dk_scale, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::LocalTensor<bfloat16_t> bf16 =
                    bf16_queue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(
                    bf16, accum, AscendC::RoundMode::CAST_RINT, count);
                fp32_queue_.FreeTensor(accum);
                bf16_queue_.EnQue(bf16);
                bf16 = bf16_queue_.DeQue<bfloat16_t>();
                AscendC::DataCopyExtParams store{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : FINALIZE_DIM) *
                        static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0,
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (kv_stride - FINALIZE_DIM) *
                            sizeof(bfloat16_t)),
                    0U};
                const uint64_t dst_offset =
                    packed_row * kv_stride +
                    static_cast<uint64_t>(kv_head) * FINALIZE_DIM;
                if (grad_kind == 0U) {
                    AscendC::DataCopyPad(dk_dst[dst_offset], bf16, store);
                } else {
                    AscendC::DataCopyPad(dv_dst[dst_offset], bf16, store);
                }
                bf16_queue_.FreeTensor(bf16);
            }
            global_job_base += batch_jobs;
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void reduce_cast_dq_colors(
        AscendC::GlobalTensor<bfloat16_t>& workspace,
        uint64_t dq_storage_elems,
        AscendC::GlobalTensor<bfloat16_t>& dq_dst,
        AscendC::GlobalTensor<uint32_t>& cu_q_lens,
        uint32_t batch, uint32_t num_q_heads, uint32_t q_stride,
        uint32_t window_left, int32_t min_delta, int32_t max_delta,
        uint32_t workers) {
        constexpr uint32_t rows_per_job =
            FINALIZE_CHUNK_ELEMS / FINALIZE_DIM / 2U;
        constexpr uint32_t row_parts =
            FINALIZE_TILE_ROWS / rows_per_job;
        const uint32_t worker = AscendC::GetBlockIdx();
        uint64_t global_job_base = 0U;
        for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
            const uint32_t q_begin = cu_q_lens.GetValue(batch_idx);
            const uint32_t q_len =
                cu_q_lens.GetValue(batch_idx + 1U) - q_begin;
            const uint32_t q_tiles =
                ceil_div(q_len, FINALIZE_TILE_ROWS);
            const uint64_t batch_jobs =
                static_cast<uint64_t>(q_tiles) * num_q_heads *
                row_parts;
            const uint64_t first_job =
                (static_cast<uint64_t>(worker) + workers -
                    global_job_base % workers) % workers;
            for (uint64_t job = first_job; job < batch_jobs;
                 job += workers) {
                uint64_t remaining = job;
                const uint32_t row_part =
                    static_cast<uint32_t>(remaining % row_parts);
                remaining /= row_parts;
                const uint32_t q_head =
                    static_cast<uint32_t>(remaining % num_q_heads);
                const uint32_t q_tile =
                    static_cast<uint32_t>(remaining / num_q_heads);
                const uint32_t tile_q_row =
                    q_tile * FINALIZE_TILE_ROWS;
                const uint32_t tile_rows =
                    min_u32(FINALIZE_TILE_ROWS, q_len - tile_q_row);
                const uint32_t row_offset =
                    row_part * rows_per_job;
                if (row_offset >= tile_rows) {
                    continue;
                }
                const uint32_t rows =
                    min_u32(rows_per_job, tile_rows - row_offset);
                const uint32_t count = rows * FINALIZE_DIM;
                const uint64_t padded_row =
                    static_cast<uint64_t>(q_begin) +
                    static_cast<uint64_t>(batch_idx) *
                        (FINALIZE_TILE_ROWS - 1U) +
                    tile_q_row + row_offset;
                const uint64_t tile_offset =
                    padded_row * q_stride +
                    static_cast<uint64_t>(q_head) * FINALIZE_DIM;
                const bool contiguous = q_stride == FINALIZE_DIM;
                AscendC::DataCopyExtParams load{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : FINALIZE_DIM) *
                        static_cast<uint32_t>(sizeof(bfloat16_t)),
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (q_stride - FINALIZE_DIM) *
                            sizeof(bfloat16_t)),
                    0, 0U};
                AscendC::DataCopyPadExtParams<bfloat16_t> no_pad{
                    false, 0U, 0U, 0.0F};
                AscendC::LocalTensor<float> accum =
                    fp32_queue_.AllocTensor<float>();
                bool has_partial = false;
                uint32_t packed_partials = 0U;
                AscendC::LocalTensor<bfloat16_t> packed;
                for (int32_t delta = min_delta;
                     delta <= max_delta; ++delta) {
                    const int32_t kv_tile_signed =
                        static_cast<int32_t>(q_tile) - delta;
                    if (kv_tile_signed < 0 ||
                        static_cast<uint32_t>(kv_tile_signed) >= q_tiles) {
                        continue;
                    }
                    const KvTileRange range =
                        make_kv_tile_range<FINALIZE_TILE_ROWS, true>(
                            tile_q_row, tile_rows, q_len,
                            window_left, 0U, 0U);
                    const uint32_t kv_tile =
                        static_cast<uint32_t>(kv_tile_signed);
                    if (kv_tile < range.first_tile ||
                        kv_tile >=
                            range.first_tile + range.local_tile_count) {
                        continue;
                    }
                    const uint64_t color =
                        static_cast<uint32_t>(delta - min_delta);
                    if (packed_partials == 0U) {
                        packed =
                            bf16_input_queue_.AllocTensor<bfloat16_t>();
                    }
                    AscendC::DataCopyPad(
                        packed[packed_partials * count],
                        workspace[
                            color * dq_storage_elems + tile_offset],
                        load, no_pad);
                    ++packed_partials;
                    if (packed_partials == 2U) {
                        bf16_input_queue_.EnQue(packed);
                        packed =
                            bf16_input_queue_.DeQue<bfloat16_t>();
                        accumulate_bf16_pair(
                            accum, packed, count,
                            packed_partials, has_partial);
                        bf16_input_queue_.FreeTensor(packed);
                        has_partial = true;
                        packed_partials = 0U;
                    }
                }
                if (packed_partials != 0U) {
                    bf16_input_queue_.EnQue(packed);
                    packed =
                        bf16_input_queue_.DeQue<bfloat16_t>();
                    accumulate_bf16_pair(
                        accum, packed, count,
                        packed_partials, has_partial);
                    bf16_input_queue_.FreeTensor(packed);
                    has_partial = true;
                }
                if (!has_partial) {
                    AscendC::Duplicate(
                        accum, 0.0F, static_cast<int32_t>(count));
                }
                AscendC::LocalTensor<bfloat16_t> bf16 =
                    bf16_queue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(
                    bf16, accum, AscendC::RoundMode::CAST_RINT, count);
                fp32_queue_.FreeTensor(accum);
                bf16_queue_.EnQue(bf16);
                bf16 = bf16_queue_.DeQue<bfloat16_t>();
                AscendC::DataCopyExtParams store{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : FINALIZE_DIM) *
                        static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0,
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (q_stride - FINALIZE_DIM) *
                            sizeof(bfloat16_t)),
                    0U};
                const uint64_t dst_offset =
                    (static_cast<uint64_t>(q_begin) +
                        tile_q_row + row_offset) *
                        q_stride +
                    static_cast<uint64_t>(q_head) * FINALIZE_DIM;
                AscendC::DataCopyPad(
                    dq_dst[dst_offset], bf16, store);
                bf16_queue_.FreeTensor(bf16);
            }
            global_job_base += batch_jobs;
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void reduce_cast_dq_planes(
        AscendC::GlobalTensor<float>& workspace,
        uint64_t partial_section_offset, uint64_t dq_storage_elems,
        uint32_t plane_count, AscendC::GlobalTensor<bfloat16_t>& dq_dst,
        AscendC::GlobalTensor<uint32_t>& cu_q_lens,
        uint32_t batch, uint32_t num_q_heads, uint32_t q_stride,
        uint32_t window_left, uint32_t kv_core_chunk_size,
        uint32_t workers) {
        constexpr uint32_t rows_per_job = FINALIZE_FP32_ROWS;
        constexpr uint32_t row_parts =
            FINALIZE_TILE_ROWS / rows_per_job;
        const uint32_t worker = AscendC::GetBlockIdx();
        uint64_t global_job_base = 0U;
        uint32_t kv_group_base = 0U;
        for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
            const uint32_t q_begin = cu_q_lens.GetValue(batch_idx);
            const uint32_t q_len =
                cu_q_lens.GetValue(batch_idx + 1U) - q_begin;
            const uint32_t q_tiles =
                ceil_div(q_len, FINALIZE_TILE_ROWS);
            const uint64_t batch_jobs =
                static_cast<uint64_t>(q_tiles) * num_q_heads *
                row_parts;
            const uint64_t first_job =
                (static_cast<uint64_t>(worker) + workers -
                    global_job_base % workers) % workers;
            for (uint64_t job = first_job; job < batch_jobs;
                 job += workers) {
                uint64_t remaining = job;
                const uint32_t row_part =
                    static_cast<uint32_t>(remaining % row_parts);
                remaining /= row_parts;
                const uint32_t q_head =
                    static_cast<uint32_t>(remaining % num_q_heads);
                const uint32_t q_tile =
                    static_cast<uint32_t>(remaining / num_q_heads);
                const uint32_t tile_q_row =
                    q_tile * FINALIZE_TILE_ROWS;
                const uint32_t tile_rows =
                    min_u32(FINALIZE_TILE_ROWS, q_len - tile_q_row);
                const uint32_t row_offset =
                    row_part * rows_per_job;
                if (row_offset >= tile_rows) {
                    continue;
                }
                const uint32_t rows =
                    min_u32(rows_per_job, tile_rows - row_offset);
                const uint32_t count = rows * FINALIZE_DIM;
                const uint64_t padded_row =
                    static_cast<uint64_t>(q_begin) +
                    static_cast<uint64_t>(batch_idx) *
                        (FINALIZE_TILE_ROWS - 1U) +
                    tile_q_row + row_offset;
                const uint64_t tile_offset =
                    padded_row * q_stride +
                    static_cast<uint64_t>(q_head) * FINALIZE_DIM;
                const bool contiguous = q_stride == FINALIZE_DIM;
                AscendC::DataCopyExtParams load{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : FINALIZE_DIM) *
                        static_cast<uint32_t>(sizeof(float)),
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (q_stride - FINALIZE_DIM) * sizeof(float)),
                    0, 0U};
                AscendC::DataCopyPadExtParams<float> no_pad{
                    false, 0U, 0U, 0.0F};
                AscendC::LocalTensor<float> accum =
                    fp32_queue_.AllocTensor<float>();
                const KvTileRange range =
                    make_kv_tile_range<FINALIZE_TILE_ROWS, true>(
                        tile_q_row, tile_rows, q_len,
                        window_left, 0U, 0U);
                const uint32_t first_group =
                    kv_group_base + range.first_tile;
                const uint32_t last_group =
                    first_group + range.local_tile_count;
                const uint32_t first_chunk =
                    first_group / kv_core_chunk_size;
                const uint32_t last_chunk =
                    ceil_div(last_group, kv_core_chunk_size);
                bool has_partial = false;
                for (uint32_t chunk = first_chunk;
                     chunk < last_chunk; ++chunk) {
                    const uint32_t plane = chunk % plane_count;
                    const uint64_t src_offset =
                        partial_section_offset +
                        static_cast<uint64_t>(plane) *
                            dq_storage_elems +
                        tile_offset;
                    if (!has_partial) {
                        AscendC::DataCopyPad(
                            accum, workspace[src_offset],
                            load, no_pad);
                        fp32_queue_.EnQue(accum);
                        accum = fp32_queue_.DeQue<float>();
                        has_partial = true;
                    } else {
                        AscendC::LocalTensor<float> input =
                            reduce_input_queue_.AllocTensor<float>();
                        AscendC::DataCopyPad(
                            input, workspace[src_offset],
                            load, no_pad);
                        reduce_input_queue_.EnQue(input);
                        input = reduce_input_queue_.DeQue<float>();
                        AscendC::Add(accum, accum, input, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        reduce_input_queue_.FreeTensor(input);
                    }
                }
                if (!has_partial) {
                    AscendC::Duplicate(
                        accum, 0.0F, static_cast<int32_t>(count));
                }
                AscendC::LocalTensor<bfloat16_t> bf16 =
                    bf16_queue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(
                    bf16, accum, AscendC::RoundMode::CAST_RINT, count);
                fp32_queue_.FreeTensor(accum);
                bf16_queue_.EnQue(bf16);
                bf16 = bf16_queue_.DeQue<bfloat16_t>();
                AscendC::DataCopyExtParams store{
                    static_cast<uint16_t>(contiguous ? 1U : rows),
                    (contiguous ? count : FINALIZE_DIM) *
                        static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0,
                    contiguous ? 0 :
                        static_cast<int64_t>(
                            (q_stride - FINALIZE_DIM) *
                            sizeof(bfloat16_t)),
                    0U};
                const uint64_t dst_offset =
                    (static_cast<uint64_t>(q_begin) +
                        tile_q_row + row_offset) *
                        q_stride +
                    static_cast<uint64_t>(q_head) * FINALIZE_DIM;
                AscendC::DataCopyPad(
                    dq_dst[dst_offset], bf16, store);
                bf16_queue_.FreeTensor(bf16);
            }
            global_job_base += batch_jobs;
            kv_group_base += q_tiles;
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void reduce_cast_add_dkv_planes(
        AscendC::GlobalTensor<float>& workspace,
        uint64_t partial_section_offset, uint64_t plane_elems,
        uint32_t plane_count, uint64_t dq_storage_elems,
        uint64_t dk_storage_elems,
        AscendC::GlobalTensor<bfloat16_t>& dk_dst,
        AscendC::GlobalTensor<bfloat16_t>& dv_dst,
        AscendC::GlobalTensor<uint32_t>& cu_q_lens,
        AscendC::GlobalTensor<uint32_t>& cu_kv_lens,
        uint32_t batch, uint32_t num_kv_heads, uint32_t kv_stride,
        uint32_t window_left, uint32_t global_window_size, float scale,
        uint32_t workers) {
        constexpr uint32_t rows_per_job = FINALIZE_FP32_ROWS;
        const uint32_t worker = AscendC::GetBlockIdx();
        uint64_t global_job_base = 0U;
        uint64_t partial_row_base = 0U;
        for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
            const uint32_t q_len =
                cu_q_lens.GetValue(batch_idx + 1U) -
                cu_q_lens.GetValue(batch_idx);
            const uint32_t kv_begin = cu_kv_lens.GetValue(batch_idx);
            const uint32_t kv_len =
                cu_kv_lens.GetValue(batch_idx + 1U) - kv_begin;
            const uint32_t global_rows =
                min_u32(global_window_size,
                    q_len > 0U && kv_len >= q_len &&
                            kv_len > window_left ?
                        kv_len - window_left : 0U);
            const uint32_t row_parts =
                ceil_div(global_rows, rows_per_job);
            const uint64_t batch_jobs =
                static_cast<uint64_t>(row_parts) *
                num_kv_heads * 2U;
            const uint64_t first_job =
                (static_cast<uint64_t>(worker) + workers -
                    global_job_base % workers) % workers;
            for (uint64_t job = first_job; job < batch_jobs;
                 job += workers) {
                uint64_t remaining = job;
                const uint32_t row_part =
                    static_cast<uint32_t>(remaining % row_parts);
                remaining /= row_parts;
                const uint32_t kv_head =
                    static_cast<uint32_t>(remaining % num_kv_heads);
                const uint32_t grad_kind =
                    static_cast<uint32_t>(remaining / num_kv_heads);
                const uint32_t row_offset = row_part * rows_per_job;
                const uint32_t rows =
                    min_u32(rows_per_job, global_rows - row_offset);
                const uint32_t count = rows * FINALIZE_DIM;
                const uint64_t padded_row =
                    static_cast<uint64_t>(kv_begin) +
                    static_cast<uint64_t>(batch_idx) *
                        (FINALIZE_TILE_ROWS - 1U) +
                    row_offset;
                const uint64_t base_offset =
                    dq_storage_elems +
                    static_cast<uint64_t>(grad_kind) *
                        dk_storage_elems +
                    padded_row * kv_stride +
                    static_cast<uint64_t>(kv_head) * FINALIZE_DIM;
                AscendC::DataCopyExtParams load{
                    static_cast<uint16_t>(rows),
                    FINALIZE_DIM * static_cast<uint32_t>(sizeof(float)),
                    static_cast<int64_t>(
                        (kv_stride - FINALIZE_DIM) * sizeof(float)),
                    0, 0U};
                AscendC::DataCopyPadExtParams<float> no_pad{
                    false, 0U, 0U, 0.0F};
                AscendC::LocalTensor<float> accum =
                    fp32_queue_.AllocTensor<float>();
                AscendC::DataCopyPad(
                    accum, workspace[base_offset], load, no_pad);
                fp32_queue_.EnQue(accum);
                accum = fp32_queue_.DeQue<float>();
                const uint64_t partial_row =
                    partial_row_base + row_offset;
                const uint64_t partial_offset =
                    static_cast<uint64_t>(grad_kind) * plane_elems +
                    partial_row * kv_stride +
                    static_cast<uint64_t>(kv_head) * FINALIZE_DIM;
                for (uint32_t plane = 0U; plane < plane_count; ++plane) {
                    AscendC::LocalTensor<float> input =
                        reduce_input_queue_.AllocTensor<float>();
                    AscendC::DataCopyPad(
                        input,
                        workspace[
                            partial_section_offset +
                            2U * static_cast<uint64_t>(plane) *
                                plane_elems +
                            partial_offset],
                        load, no_pad);
                    reduce_input_queue_.EnQue(input);
                    input = reduce_input_queue_.DeQue<float>();
                    AscendC::Add(accum, accum, input, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    reduce_input_queue_.FreeTensor(input);
                }
                if (grad_kind == 0U) {
                    AscendC::Muls(accum, accum, scale, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::LocalTensor<bfloat16_t> output =
                    bf16_queue_.AllocTensor<bfloat16_t>();
                AscendC::Cast(
                    output, accum, AscendC::RoundMode::CAST_RINT, count);
                fp32_queue_.FreeTensor(accum);
                bf16_queue_.EnQue(output);
                output = bf16_queue_.DeQue<bfloat16_t>();
                AscendC::DataCopyExtParams store{
                    static_cast<uint16_t>(rows),
                    FINALIZE_DIM *
                        static_cast<uint32_t>(sizeof(bfloat16_t)),
                    0,
                    static_cast<int64_t>(
                        (kv_stride - FINALIZE_DIM) *
                        sizeof(bfloat16_t)),
                    0U};
                const uint64_t dst_offset =
                    (static_cast<uint64_t>(kv_begin) + row_offset) *
                        kv_stride +
                    static_cast<uint64_t>(kv_head) * FINALIZE_DIM;
                if (grad_kind == 0U) {
                    AscendC::DataCopyPad(
                        dk_dst[dst_offset], output, store);
                } else {
                    AscendC::DataCopyPad(
                        dv_dst[dst_offset], output, store);
                }
                bf16_queue_.FreeTensor(output);
            }
            partial_row_base +=
                ceil_div(global_rows, FINALIZE_TILE_ROWS) *
                FINALIZE_TILE_ROWS;
            global_job_base += batch_jobs;
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void cast_packed_q_tails(
        AscendC::GlobalTensor<float>& src, uint64_t section_offset,
        AscendC::GlobalTensor<bfloat16_t>& dst,
        AscendC::GlobalTensor<uint32_t>& cu_lens, uint32_t batch,
        uint32_t num_heads, uint32_t stride, uint32_t workers,
        bool compact_workspace = false) {
        const uint32_t worker = AscendC::GetBlockIdx();
        const uint64_t jobs = static_cast<uint64_t>(batch) * num_heads;
        for (uint64_t job = worker; job < jobs; job += workers) {
            const uint32_t batch_idx = static_cast<uint32_t>(job / num_heads);
            const uint32_t head = static_cast<uint32_t>(job % num_heads);
            const uint32_t begin = cu_lens.GetValue(batch_idx);
            const uint32_t length = cu_lens.GetValue(batch_idx + 1U) - begin;
            const uint32_t rows = length % FINALIZE_TILE_ROWS;
            if (rows == 0U) {
                continue;
            }
            const uint64_t src_offset = section_offset +
                (compact_workspace ?
                    static_cast<uint64_t>(batch_idx) *
                        FINALIZE_TILE_ROWS :
                    static_cast<uint64_t>(begin) +
                        static_cast<uint64_t>(batch_idx) *
                            (FINALIZE_TILE_ROWS - 1U) +
                        length - rows) * stride +
                static_cast<uint64_t>(head) * FINALIZE_DIM;
            const uint64_t dst_offset =
                (static_cast<uint64_t>(begin) + length - rows) * stride +
                static_cast<uint64_t>(head) * FINALIZE_DIM;
            const uint32_t count = rows * FINALIZE_DIM;
            AscendC::DataCopyExtParams load{
                static_cast<uint16_t>(rows), FINALIZE_DIM * sizeof(float),
                static_cast<int64_t>((stride - FINALIZE_DIM) * sizeof(float)),
                0U, 0U};
            AscendC::DataCopyPadExtParams<float> no_pad{false, 0U, 0U, 0.0F};
            AscendC::LocalTensor<float> fp32 = fp32_queue_.AllocTensor<float>();
            AscendC::DataCopyPad(fp32, src[src_offset], load, no_pad);
            fp32_queue_.EnQue(fp32);
            fp32 = fp32_queue_.DeQue<float>();
            AscendC::LocalTensor<bfloat16_t> bf16 =
                bf16_queue_.AllocTensor<bfloat16_t>();
            AscendC::Cast(bf16, fp32, AscendC::RoundMode::CAST_RINT, count);
            fp32_queue_.FreeTensor(fp32);
            bf16_queue_.EnQue(bf16);
            bf16 = bf16_queue_.DeQue<bfloat16_t>();
            AscendC::DataCopyExtParams store{
                static_cast<uint16_t>(rows),
                FINALIZE_DIM * sizeof(bfloat16_t), 0U,
                static_cast<int64_t>((stride - FINALIZE_DIM) *
                    sizeof(bfloat16_t)), 0U};
            AscendC::DataCopyPad(dst[dst_offset], bf16, store);
            bf16_queue_.FreeTensor(bf16);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void cast_packed_range(
        AscendC::GlobalTensor<float>& src, uint64_t section_offset,
        AscendC::GlobalTensor<bfloat16_t>& dst,
        AscendC::GlobalTensor<uint32_t>& cu_lens, uint32_t batch,
        uint32_t stride, float scale, uint32_t workers,
        uint64_t& global_chunk_base) {
        const uint32_t worker = AscendC::GetBlockIdx();
        for (uint32_t batch_idx = 0U; batch_idx < batch; ++batch_idx) {
            const uint32_t token_begin = cu_lens.GetValue(batch_idx);
            const uint32_t token_end = cu_lens.GetValue(batch_idx + 1U);
            const uint64_t total_elems =
                static_cast<uint64_t>(token_end - token_begin) * stride;
            const uint64_t src_offset = section_offset +
                (static_cast<uint64_t>(token_begin) +
                    static_cast<uint64_t>(batch_idx) * (FINALIZE_TILE_ROWS - 1U)) * stride;
            const uint64_t dst_offset = static_cast<uint64_t>(token_begin) * stride;
            const uint64_t chunks =
                (total_elems + FINALIZE_CHUNK_ELEMS - 1U) / FINALIZE_CHUNK_ELEMS;
            const uint64_t first_chunk =
                (static_cast<uint64_t>(worker) + workers -
                    global_chunk_base % workers) % workers;
            for (uint64_t chunk = first_chunk; chunk < chunks; chunk += workers) {
                const uint64_t offset = chunk * FINALIZE_CHUNK_ELEMS;
                const uint32_t count = static_cast<uint32_t>(
                    total_elems - offset < FINALIZE_CHUNK_ELEMS ?
                        total_elems - offset : FINALIZE_CHUNK_ELEMS);
                AscendC::DataCopyExtParams load{
                    1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                AscendC::DataCopyPadExtParams<float> no_pad{false, 0U, 0U, 0.0F};
                AscendC::LocalTensor<float> fp32 = fp32_queue_.AllocTensor<float>();
                AscendC::DataCopyPad(fp32, src[src_offset + offset], load, no_pad);
                fp32_queue_.EnQue(fp32);
                fp32 = fp32_queue_.DeQue<float>();
                AscendC::LocalTensor<bfloat16_t> bf16 =
                    bf16_queue_.AllocTensor<bfloat16_t>();
                if (scale != 1.0F) {
                    AscendC::Muls(fp32, fp32, scale, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Cast(bf16, fp32, AscendC::RoundMode::CAST_RINT, count);
                fp32_queue_.FreeTensor(fp32);
                bf16_queue_.EnQue(bf16);
                bf16 = bf16_queue_.DeQue<bfloat16_t>();
                AscendC::DataCopyExtParams store{
                    1U, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
                AscendC::DataCopyPad(dst[dst_offset + offset], bf16, store);
                bf16_queue_.FreeTensor(bf16);
            }
            global_chunk_base += chunks;
        }
    }

private:
    __aicore__ inline void accumulate_bf16_pair(
        AscendC::LocalTensor<float>& accum,
        AscendC::LocalTensor<bfloat16_t>& packed,
        uint32_t count, uint32_t partials, bool has_partial) {
        AscendC::LocalTensor<float> input =
            reduce_input_queue_.AllocTensor<float>();
        AscendC::Cast(
            input, packed, AscendC::RoundMode::CAST_NONE,
            partials * count);
        AscendC::PipeBarrier<PIPE_V>();
        if (partials == 2U) {
            AscendC::Add(
                input, input, input[count], count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (has_partial) {
            AscendC::Add(accum, accum, input, count);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(accum, input, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        reduce_input_queue_.FreeTensor(input);
    }

    __aicore__ inline void accumulate_bf16_partial(
        AscendC::LocalTensor<float>& accum,
        AscendC::LocalTensor<bfloat16_t>& partial,
        uint32_t count, bool has_partial) {
        if (!has_partial) {
            AscendC::Cast(
                accum, partial, AscendC::RoundMode::CAST_NONE, count);
            return;
        }
        AscendC::LocalTensor<float> input =
            reduce_input_queue_.AllocTensor<float>();
        AscendC::Cast(
            input, partial, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(accum, accum, input, count);
        AscendC::PipeBarrier<PIPE_V>();
        reduce_input_queue_.FreeTensor(input);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> fp32_queue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> reduce_input_queue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> bf16_queue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> bf16_input_queue_;
};

}  // namespace SWA_v5::Backward
