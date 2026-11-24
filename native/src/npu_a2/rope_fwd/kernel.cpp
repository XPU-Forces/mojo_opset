// Adapted from data/xpu_ops commit 8452b29b; see SOURCE.json.
#include "../common/rope_tiling.h"
#include "kernel.h"

#include "../common/common.h"
#include "../common/data_copy_utils.h"

namespace mojo::native::a2 {
namespace kernels {

template <typename T, typename EmbedT>
__aicore__ inline void RopeForward<T, EmbedT>::init(GM_ADDR input, GM_ADDR sin_embeds, GM_ADDR cos_embeds,
                                                    const RopeForwardTiling &tiling) {
  input_global_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
  sin_embeds_global_.SetGlobalBuffer(reinterpret_cast<__gm__ EmbedT *>(sin_embeds));
  cos_embeds_global_.SetGlobalBuffer(reinterpret_cast<__gm__ EmbedT *>(cos_embeds));

  tiling_ = tiling;

  skip_head_dim_ = tiling_.head_dim - tiling_.cut_head_dim;

  pipe_.InitBuffer(input_queue_, 1, tiling_.chunk_size * tiling_.cut_head_dim * sizeof(float));
  pipe_.InitBuffer(sin_embeds_queue_, 1, tiling_.chunk_size * tiling_.cut_head_dim * sizeof(float));
  pipe_.InitBuffer(cos_embeds_queue_, 1, tiling_.chunk_size * tiling_.cut_head_dim * sizeof(float));
  pipe_.InitBuffer(output_queue_, 1, tiling_.chunk_size * tiling_.cut_head_dim * sizeof(T));
  input_buffer_size_ = tiling_.chunk_size * tiling_.cut_head_dim;
}

template <typename T, typename EmbedT>
__aicore__ inline void
RopeForward<T, EmbedT>::do_rope_mul(AscendC::LocalTensor<float> &input_float_local,
                                    AscendC::LocalTensor<float> &sin_embeds_float_local,
                                    AscendC::LocalTensor<float> &cos_embeds_float_local,
                                    AscendC::LocalTensor<T> &output_local, int32_t valid_copy_rows) {
  // x * cos
  AscendC::Mul(cos_embeds_float_local, input_float_local, cos_embeds_float_local,
               valid_copy_rows * tiling_.cut_head_dim);
  // FP32 masks cover at most 64 channels. Keep the row stride while
  // splitting larger rotary halves into independent vector operations.
  const int32_t half = tiling_.cut_head_dim / 2;
  const auto stride = static_cast<uint8_t>(tiling_.cut_head_dim / (VEC_BLOCK_BYTES / sizeof(float)));
  for (int32_t column = 0; column < half; column += 64) {
    const uint32_t width = min(half - column, 64);
    AscendC::Muls(sin_embeds_float_local[column], sin_embeds_float_local[column], -1.0f, width,
                  valid_copy_rows, {1, 1, stride, stride});
    AscendC::Mul(sin_embeds_float_local[column], input_float_local[half + column],
                 sin_embeds_float_local[column], width, valid_copy_rows, {1, 1, 1, stride, stride, stride});
    AscendC::Mul(sin_embeds_float_local[half + column], input_float_local[column],
                 sin_embeds_float_local[half + column], width, valid_copy_rows,
                 {1, 1, 1, stride, stride, stride});
  }
  if constexpr (!std::is_same<T, float>::value) {
    // x * cos + rotate_half(x) * sin
    AscendC::Add(cos_embeds_float_local, cos_embeds_float_local, sin_embeds_float_local,
                 valid_copy_rows * tiling_.cut_head_dim);
    AscendC::Cast(output_local, cos_embeds_float_local, AscendC::RoundMode::CAST_RINT,
                  valid_copy_rows * tiling_.cut_head_dim);
  } else {
    AscendC::Add(output_local, cos_embeds_float_local, sin_embeds_float_local,
                 valid_copy_rows * tiling_.cut_head_dim);
  }
}

template <typename T, typename EmbedT> __aicore__ inline void RopeForward<T, EmbedT>::process() {
  int64_t block_id = AscendC::GetBlockIdx();
  int64_t cur_blocks =
      (block_id < tiling_.normal_cores ? tiling_.blocks_per_core : tiling_.blocks_per_core_tail);
  int64_t pre_blocks =
      (block_id < tiling_.normal_cores ? (block_id * tiling_.blocks_per_core)
                                       : (tiling_.normal_cores * tiling_.blocks_per_core +
                                          (block_id - tiling_.normal_cores) * tiling_.blocks_per_core_tail));

  int64_t cur_block_idx = pre_blocks;
  for (int n_block = 0; n_block < cur_blocks; n_block++) {
    int32_t head_id = cur_block_idx % tiling_.head_num;
    int32_t batch_id = cur_block_idx / tiling_.head_num;
    // get valid kv&q lens
    int32_t cur_cumsum_len = batch_id * tiling_.seqlen_per_block; // cumsum_len_global_.GetValue(batch_id);
    int32_t next_cumsum_len = (batch_id < tiling_.batch_size - 1)
                                  ? (batch_id + 1) * tiling_.seqlen_per_block
                                  : tiling_.token_num; // cumsum_len_global_.GetValue(batch_id + 1);
    int32_t cur_query_len = next_cumsum_len - cur_cumsum_len;

    int64_t token_num_offset = cur_cumsum_len * tiling_.head_num * tiling_.head_dim;
    int64_t head_num_offset = token_num_offset + head_id * tiling_.head_dim;

    int32_t row_id = 0;
    int32_t vec_copy_nums = (cur_query_len + tiling_.chunk_size - 1) / tiling_.chunk_size;
    for (int copy_num = 0; copy_num < vec_copy_nums; copy_num++) {
      int32_t valid_copy_rows = min(cur_query_len - row_id, tiling_.chunk_size);
      // seqlen offset
      int32_t seqlen_chunk_id = copy_num * tiling_.chunk_size;
      int64_t seq_len_offset = head_num_offset + seqlen_chunk_id * tiling_.token_num_stride;
      int64_t input_offset = seq_len_offset;

      AscendC::LocalTensor<T> input_local = input_queue_.AllocTensor<T>();
      AscendC::LocalTensor<float> input_float_local;
      if constexpr (!std::is_same<T, float>::value) {
        auto input_local_tmp = input_local[input_buffer_size_];
        mojo::native::a2::kernels::aligned_datacopy_gm2ub<T>(
            /*rows*/ valid_copy_rows, /*cols*/ tiling_.cut_head_dim,
            /*src_stride*/ tiling_.token_num_stride,
            /*dst_stride*/ tiling_.cut_head_dim, input_global_[input_offset + skip_head_dim_],
            input_local_tmp);
        input_queue_.EnQue(input_local);
        input_local = input_queue_.DeQue<T>();
        input_float_local = input_local.template ReinterpretCast<float>();
        AscendC::Cast(input_float_local, input_local[input_buffer_size_], AscendC::RoundMode::CAST_NONE,
                      valid_copy_rows * tiling_.cut_head_dim);
      } else {
        mojo::native::a2::kernels::aligned_datacopy_gm2ub<T>(
            /*rows*/ valid_copy_rows, /*cols*/ tiling_.cut_head_dim,
            /*src_stride*/ tiling_.token_num_stride,
            /*dst_stride*/ tiling_.cut_head_dim, input_global_[input_offset + skip_head_dim_], input_local);
        input_queue_.EnQue(input_local);
        input_local = input_queue_.DeQue<T>();
        input_float_local = input_local;
      }

      AscendC::LocalTensor<T> output_local = output_queue_.AllocTensor<T>();
      int64_t output_offset = input_offset;

      int32_t embeds_offset = (cur_cumsum_len + seqlen_chunk_id) * tiling_.cut_head_dim;

      AscendC::LocalTensor<float> cos_embeds_float_local;
      AscendC::LocalTensor<float> sin_embeds_float_local;
      if constexpr (!std::is_same<EmbedT, float>::value) {
        // sin
        AscendC::LocalTensor<EmbedT> sin_embeds_local = sin_embeds_queue_.AllocTensor<EmbedT>();
        auto sin_embeds_local_tmp = sin_embeds_local[input_buffer_size_];
        mojo::native::a2::kernels::aligned_datacopy_gm2ub<EmbedT>(
            /*rows*/ valid_copy_rows, /*cols*/ tiling_.cut_head_dim, /*src_stride*/ tiling_.cut_head_dim,
            /*dst_stride*/ tiling_.cut_head_dim, sin_embeds_global_[embeds_offset], sin_embeds_local_tmp);
        sin_embeds_queue_.EnQue(sin_embeds_local);
        sin_embeds_local = sin_embeds_queue_.DeQue<EmbedT>();

        // cos
        AscendC::LocalTensor<EmbedT> cos_embeds_local = cos_embeds_queue_.AllocTensor<EmbedT>();
        auto cos_embeds_local_tmp = cos_embeds_local[input_buffer_size_];
        mojo::native::a2::kernels::aligned_datacopy_gm2ub<EmbedT>(
            /*rows*/ valid_copy_rows, /*cols*/ tiling_.cut_head_dim, /*src_stride*/ tiling_.cut_head_dim,
            /*dst_stride*/ tiling_.cut_head_dim, cos_embeds_global_[embeds_offset], cos_embeds_local_tmp);
        cos_embeds_queue_.EnQue(cos_embeds_local);
        cos_embeds_local = cos_embeds_queue_.DeQue<EmbedT>();

        // cast
        cos_embeds_float_local = cos_embeds_local.template ReinterpretCast<float>();
        sin_embeds_float_local = sin_embeds_local.template ReinterpretCast<float>();

        AscendC::Cast(cos_embeds_float_local, cos_embeds_local[input_buffer_size_],
                      AscendC::RoundMode::CAST_NONE, valid_copy_rows * tiling_.cut_head_dim);
        AscendC::Cast(sin_embeds_float_local, sin_embeds_local[input_buffer_size_],
                      AscendC::RoundMode::CAST_NONE, valid_copy_rows * tiling_.cut_head_dim);

      } else {
        // sin
        sin_embeds_float_local = sin_embeds_queue_.AllocTensor<EmbedT>();
        mojo::native::a2::kernels::aligned_datacopy_gm2ub<EmbedT>(
            /*rows*/ valid_copy_rows, /*cols*/ tiling_.cut_head_dim, /*src_stride*/ tiling_.cut_head_dim,
            /*dst_stride*/ tiling_.cut_head_dim, sin_embeds_global_[embeds_offset], sin_embeds_float_local);
        sin_embeds_queue_.EnQue(sin_embeds_float_local);
        sin_embeds_float_local = sin_embeds_queue_.DeQue<EmbedT>();

        // cos
        cos_embeds_float_local = cos_embeds_queue_.AllocTensor<EmbedT>();
        mojo::native::a2::kernels::aligned_datacopy_gm2ub<EmbedT>(
            /*rows*/ valid_copy_rows, /*cols*/ tiling_.cut_head_dim, /*src_stride*/ tiling_.cut_head_dim,
            /*dst_stride*/ tiling_.cut_head_dim, cos_embeds_global_[embeds_offset], cos_embeds_float_local);
        cos_embeds_queue_.EnQue(cos_embeds_float_local);
        cos_embeds_float_local = cos_embeds_queue_.DeQue<EmbedT>();
      }

      do_rope_mul(input_float_local, sin_embeds_float_local, cos_embeds_float_local, output_local,
                  valid_copy_rows);

      // sync out
      output_queue_.EnQue(output_local);
      output_local = output_queue_.DeQue<T>();
      auto qkv_input_global = input_global_[output_offset + skip_head_dim_];
      mojo::native::a2::kernels::aligned_datacopy_ub2gm<T>(
          /*rows*/ valid_copy_rows, /*cols*/ tiling_.cut_head_dim, /*src_stride*/ tiling_.cut_head_dim,
          /*dst_stride*/ tiling_.token_num_stride, output_local, qkv_input_global);

      cos_embeds_queue_.FreeTensor(cos_embeds_float_local);
      input_queue_.FreeTensor(input_float_local);
      sin_embeds_queue_.FreeTensor(sin_embeds_float_local);
      output_queue_.FreeTensor(output_local);
      row_id += tiling_.chunk_size;
    }
    cur_block_idx++;
  }
}

} // namespace kernels
} // namespace mojo::native::a2

extern "C" __global__ __aicore__ void MOJO_ROPE_KERNEL(GM_ADDR input, GM_ADDR sin_embeds, GM_ADDR cos_embeds,
                                                       RopeForwardTiling tiling_data_in) {
  mojo::native::a2::kernels::RopeForward<DTYPE_INPUT, DTYPE_SIN_EMBEDS> rotary_embeds;

  const RopeForwardTiling *tiling_data = &tiling_data_in;
  rotary_embeds.init(input, sin_embeds, cos_embeds, *tiling_data);
  rotary_embeds.process();
}
