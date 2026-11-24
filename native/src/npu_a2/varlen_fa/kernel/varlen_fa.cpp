#if defined(__CCE_AICORE__) && (__CCE_AICORE__ >= 200)
#ifndef TILING_KEY_VAR
#define TILING_KEY_VAR 9
#endif
#include "varlen_fa_tiling.h"
#include "varlen_fa.h"

#ifndef VARLEN_FA_KERNEL_NAME
#define VARLEN_FA_KERNEL_NAME varlen_fa
#endif

extern "C" CATLASS_GLOBAL void VARLEN_FA_KERNEL_NAME(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR mask,
                                         GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k, GM_ADDR attn_out,
                                         GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  using namespace xpu_ops::kernels;
  using ArchTag = Catlass::Arch::AtlasA2;

  auto tiling_gm = reinterpret_cast<__gm__ VarlenFaTiling *>(tiling);
  VarlenFaTiling tiling_data;
  tiling_data.batch_size = tiling_gm->batch_size;
  tiling_data.total_q_tokens = tiling_gm->total_q_tokens;
  tiling_data.total_k_tokens = tiling_gm->total_k_tokens;
  tiling_data.head_qo = tiling_gm->head_qo;
  tiling_data.head_kv = tiling_gm->head_kv;
  tiling_data.gqa_layout = tiling_gm->gqa_layout;
  tiling_data.seq_len_q = tiling_gm->seq_len_q;
  tiling_data.seq_len_k = tiling_gm->seq_len_k;
  tiling_data.max_seq_len_q = tiling_gm->max_seq_len_q;
  tiling_data.max_seq_len_k = tiling_gm->max_seq_len_k;
  tiling_data.head_dim = tiling_gm->head_dim;
  tiling_data.mask_type = tiling_gm->mask_type;
  tiling_data.mask_layout = tiling_gm->mask_layout;
  tiling_data.num_mask_q_tiles = tiling_gm->num_mask_q_tiles;
  tiling_data.scale = tiling_gm->scale;
  tiling_data.aligned_seq_len_k = tiling_gm->aligned_seq_len_k;
  tiling_data.tile_qo = tiling_gm->tile_qo;
  tiling_data.tile_kv = tiling_gm->tile_kv;
  tiling_data.pipeline_stages = tiling_gm->pipeline_stages;

  // if (TILING_KEY_IS(0)) {
  //   using TileSizeFa = Catlass::GemmShape<128, 256, 64>;
  //   using FaKernel = CustomFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, true>;
  //   FaKernel kernel;
  //   kernel.Init(query, key, value, mask, attn_out, workspace, tiling_data);
  //   kernel.Process();
  // } else if (TILING_KEY_IS(1)) {
  //   using TileSizeFa = Catlass::GemmShape<128, 256, 64>;
  //   using FaKernel = CustomFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, false>;
  //   FaKernel kernel;
  //   kernel.Init(query, key, value, mask, attn_out, workspace, tiling_data);
  //   kernel.Process();
  // } else if (TILING_KEY_IS(2)) {
  //   using TileSizeFa = Catlass::GemmShape<128, 256, 128>;
  //   using FaKernel = CustomFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, true>;
  //   FaKernel kernel;
  //   kernel.Init(query, key, value, mask, attn_out, workspace, tiling_data);
  //   kernel.Process();
  // } else if (TILING_KEY_IS(3)) {
  //   using TileSizeFa = Catlass::GemmShape<128, 256, 128>;
  //   using FaKernel = CustomFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, false>;
  //   FaKernel kernel;
  //   kernel.Init(query, key, value, mask, attn_out, workspace, tiling_data);
  //   kernel.Process();
  // } else if (TILING_KEY_IS(4)) {
  //   using TileSizeFa = Catlass::GemmShape<64, 256, 256>;
  //   using FaKernel = CustomFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, true>;
  //   FaKernel kernel;
  //   kernel.Init(query, key, value, mask, attn_out, workspace, tiling_data);
  //   kernel.Process();
  // } else if (TILING_KEY_IS(5)) {
  //   using TileSizeFa = Catlass::GemmShape<64, 256, 256>;
  //   using FaKernel = CustomFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, false>;
  //   FaKernel kernel;
  //   kernel.Init(query, key, value, mask, attn_out, workspace, tiling_data);
  //   kernel.Process();
  // }
  if (TILING_KEY_IS(2)) {
    using TileSizeFa = Catlass::GemmShape<128, 128, 128>;
    using FaKernel =
        VarlenFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, true, 2, CUSTOM_FA_MMAD_L0_STAGE_NUM>;
    FaKernel kernel;
    kernel.Init(query, key, value, mask, cu_seqlens_q, cu_seqlens_k, attn_out, workspace, tiling_data);
    kernel.pipelinedProcess();
  } else if (TILING_KEY_IS(3)) {
    using TileSizeFa = Catlass::GemmShape<128, CUSTOM_FA_FULL_BLOCK_KV, 128>;
    using FaKernel =
        VarlenFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, false, 2, CUSTOM_FA_MMAD_L0_STAGE_NUM>;
    FaKernel kernel;
    kernel.Init(query, key, value, mask, cu_seqlens_q, cu_seqlens_k, attn_out, workspace, tiling_data);
    kernel.pipelinedProcess();
  } else if (TILING_KEY_IS(9)) {
    using TileSizeFa = Catlass::GemmShape<128, CUSTOM_FA_FULL_BLOCK_KV, 128>;
    using FaKernel = VarlenFa<ArchTag, DTYPE_QUERY, float, DTYPE_MASK, TileSizeFa, false, 3, 2>;
    FaKernel kernel;
    kernel.Init(query, key, value, mask, cu_seqlens_q, cu_seqlens_k, attn_out, workspace, tiling_data);
    kernel.pipelinedProcess();
  }
}

#else

extern "C" void VARLEN_FA_KERNEL_NAME(void *query, void *key, void *value, void *mask, void *cu_seqlens_q, void *cu_seqlens_k,
                         void *attn_out, void *workspace, void *tiling) {}

#endif
