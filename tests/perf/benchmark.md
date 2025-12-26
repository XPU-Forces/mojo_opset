| Name | Parameters | Device Latency (us) | Host Latency (ms) |
|------|------------|---------------------|-------------------|
| TTXGelu_TORCH_REF | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 1.9480 us | 0.0317 ms |
| TTXGelu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 4.9322 us | 0.2851 ms |
| TTXSilu_TORCH_REF | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 1.8560 us | 0.0323 ms |
| TTXSilu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 4.6240 us | 0.2933 ms |
| TTXSwiGLU_TORCH_REF | gate_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0)<br>  up_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 4.8642 us | 0.0349 ms |
| TTXSwiGLU | gate_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0)<br>  up_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 4.6122 us | 0.2916 ms |
| TTXGatedDeltaRule_TORCH_REF | beta: Tensor(shape=(1, 512, 4), dtype=torch.float32, device=npu:0)<br>  cu_seqlens: Tensor(shape=(3,), dtype=torch.int64, device=npu:0)<br>  g: Tensor(shape=(1, 512, 4), dtype=torch.float16, device=npu:0)<br>  k: Tensor(shape=(1, 512, 4, 128), dtype=torch.float16, device=npu:0)<br>  q: Tensor(shape=(1, 512, 24, 128), dtype=torch.float16, device=npu:0)<br>  v: Tensor(shape=(1, 512, 4, 256), dtype=torch.float16, device=npu:0) | 16529.9012 us | 32.3648 ms |
| TTXGatedDeltaRule | beta: Tensor(shape=(1, 512, 4), dtype=torch.float32, device=npu:0)<br>  cu_seqlens: Tensor(shape=(3,), dtype=torch.int64, device=npu:0)<br>  g: Tensor(shape=(1, 512, 4), dtype=torch.float16, device=npu:0)<br>  k: Tensor(shape=(1, 512, 4, 128), dtype=torch.float16, device=npu:0)<br>  q: Tensor(shape=(1, 512, 24, 128), dtype=torch.float16, device=npu:0)<br>  v: Tensor(shape=(1, 512, 4, 256), dtype=torch.float16, device=npu:0) | 17270.4888 us | 10.7182 ms |
| TTXPagedDecodeGQA_TORCH_REF | block_tables: Tensor(shape=(8, 31), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(181, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(8, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  seqlens: Tensor(shape=(8,), dtype=torch.int32, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(181, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 5795.8914 us | 14.9972 ms |
| TTXPagedDecodeGQA | block_tables: Tensor(shape=(8, 31), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(181, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(8, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  seqlens: Tensor(shape=(8,), dtype=torch.int32, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(181, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 107.3506 us | 0.3710 ms |
| TTXPagedPrefillGQA_TORCH_REF | block_tables: Tensor(shape=(2, 32), dtype=torch.int64, device=npu:0)<br>  cu_seqlens_q: Tensor(shape=(3,), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(68, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(1813, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(68, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 4714.5946 us | 7.4142 ms |
| TTXPagedPrefillGQA | block_tables: Tensor(shape=(2, 32), dtype=torch.int64, device=npu:0)<br>  cu_seqlens_q: Tensor(shape=(3,), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(68, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(1813, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(68, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 22840.5044 us | 23.2753 ms |
| TTXNorm_TORCH_REF | x: Tensor(shape=(1, 32, 2048), dtype=torch.float32, device=npu:0) | 29.5762 us | 0.0705 ms |
| TTXNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.float32, device=npu:0) | 6.0244 us | 0.2900 ms |
| TTXNorm_TORCH_REF | x: Tensor(shape=(1, 32, 2048), dtype=torch.float16, device=npu:0) | 37.9888 us | 0.0927 ms |
| TTXNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.float16, device=npu:0) | 5.7040 us | 0.3018 ms |
| TTXNorm_TORCH_REF | x: Tensor(shape=(1, 32, 2048), dtype=torch.bfloat16, device=npu:0) | 37.3290 us | 0.1269 ms |
| TTXNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.bfloat16, device=npu:0) | 6.4840 us | 0.2948 ms |
| TTXNorm_TORCH_REF | x: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 84.0818 us | 0.1393 ms |
| TTXNorm | x: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 6.2920 us | 0.3024 ms |
| TTXNorm_TORCH_REF | x: Tensor(shape=(256, 128), dtype=torch.float16, device=npu:0) | 65.9896 us | 0.1175 ms |
| TTXNorm | x: Tensor(shape=(256, 128), dtype=torch.float16, device=npu:0) | 5.7122 us | 0.2694 ms |
| TTXNorm_TORCH_REF | x: Tensor(shape=(256, 128), dtype=torch.bfloat16, device=npu:0) | 68.1374 us | 0.1316 ms |
| TTXNorm | x: Tensor(shape=(256, 128), dtype=torch.bfloat16, device=npu:0) | 5.8482 us | 0.2748 ms |
| TTXRoPE_TORCH_REF | cos: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0)<br>  k: Tensor(shape=(1, 8, 1024, 32), dtype=torch.float32, device=npu:0)<br>  q: Tensor(shape=(1, 32, 1024, 32), dtype=torch.float32, device=npu:0)<br>  sin: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0) | 337.5382 us | 0.2701 ms |
| TTXRoPE | cos: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0)<br>  k: Tensor(shape=(1, 8, 1024, 32), dtype=torch.float32, device=npu:0)<br>  q: Tensor(shape=(1, 32, 1024, 32), dtype=torch.float32, device=npu:0)<br>  sin: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0) | 26.0166 us | 0.3366 ms |
| RefGelu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 2.0440 us | 0.0336 ms |
| TTXGelu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 4.7880 us | 0.2819 ms |
| RefSilu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 1.8120 us | 0.0318 ms |
| TTXSilu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 4.2442 us | 0.3295 ms |
| RefSwiGLU | gate_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0)<br>  up_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 4.9524 us | 0.1121 ms |
| TTXSwiGLU | gate_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0)<br>  up_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 4.9322 us | 0.2801 ms |
| RefGelu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 1.9120 us | 0.0304 ms |
| TTXGelu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 5.5320 us | 0.2657 ms |
| RefSilu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 1.9680 us | 0.0287 ms |
| TTXSilu | x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 5.2162 us | 0.2944 ms |
| RefSwiGLU | gate_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0)<br>  up_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 4.9440 us | 0.0341 ms |
| TTXSwiGLU | gate_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0)<br>  up_out: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 5.1042 us | 0.2757 ms |
| RefPagedDecodeGQA | block_tables: Tensor(shape=(8, 30), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(176, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(8, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  seqlens: Tensor(shape=(8,), dtype=torch.int32, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(176, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 5369.2910 us | 14.3556 ms |
| TTXPagedDecodeGQA | block_tables: Tensor(shape=(8, 30), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(176, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(8, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  seqlens: Tensor(shape=(8,), dtype=torch.int32, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(176, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 106.2382 us | 0.4742 ms |
| RefPagedPrefillGQA | block_tables: Tensor(shape=(2, 26), dtype=torch.int64, device=npu:0)<br>  cu_seqlens_q: Tensor(shape=(3,), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(55, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(1415, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(55, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 3055.9096 us | 5.1515 ms |
| TTXPagedPrefillGQA | block_tables: Tensor(shape=(2, 26), dtype=torch.int64, device=npu:0)<br>  cu_seqlens_q: Tensor(shape=(3,), dtype=torch.int64, device=npu:0)<br>  k_cache: Tensor(shape=(55, 4, 32, 128), dtype=torch.bfloat16, device=npu:0)<br>  query: Tensor(shape=(1415, 16, 128), dtype=torch.bfloat16, device=npu:0)<br>  sm_scale: 0.08838834764831843<br>  v_cache: Tensor(shape=(55, 4, 32, 128), dtype=torch.bfloat16, device=npu:0) | 14270.1610 us | 14.8702 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 28.2086 us | 0.0642 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 6.8722 us | 0.2866 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 37.9566 us | 0.0734 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 6.9842 us | 0.2952 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 39.0170 us | 0.0952 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 7.9922 us | 0.2913 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 28.1766 us | 0.0709 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 7.5362 us | 0.3024 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 36.5050 us | 0.0770 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 7.6442 us | 0.2945 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 38.7170 us | 0.0800 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 7.6362 us | 0.2970 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 7.8802 us | 0.0537 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 8.3362 us | 0.3297 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 10.6202 us | 0.0448 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 7.4842 us | 0.3175 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 10.8402 us | 0.0456 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 7.8522 us | 0.2967 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 7.6842 us | 0.0461 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float32, device=npu:0) | 8.0122 us | 0.2937 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 10.3922 us | 0.0437 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.float16, device=npu:0) | 8.3202 us | 0.2943 ms |
| RefResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 10.1962 us | 0.0526 ms |
| TTXResidualAddNorm | residual: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0)<br>  x: Tensor(shape=(128, 128), dtype=torch.bfloat16, device=npu:0) | 7.5202 us | 0.2922 ms |
| RefNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.float32, device=npu:0) | 30.1964 us | 0.0783 ms |
| TTXNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.float32, device=npu:0) | 6.9322 us | 0.2590 ms |
| RefNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.float16, device=npu:0) | 36.7410 us | 0.0705 ms |
| TTXNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.float16, device=npu:0) | 6.7842 us | 0.2450 ms |
| RefNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.bfloat16, device=npu:0) | 38.3130 us | 0.0752 ms |
| TTXNorm | x: Tensor(shape=(1, 32, 2048), dtype=torch.bfloat16, device=npu:0) | 6.5082 us | 0.2440 ms |
| RefNorm | x: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 81.2616 us | 0.1220 ms |
| TTXNorm | x: Tensor(shape=(256, 128), dtype=torch.float32, device=npu:0) | 6.2602 us | 0.2867 ms |
| RefNorm | x: Tensor(shape=(256, 128), dtype=torch.float16, device=npu:0) | 75.1378 us | 0.1284 ms |
| TTXNorm | x: Tensor(shape=(256, 128), dtype=torch.float16, device=npu:0) | 6.4642 us | 0.3203 ms |
| RefNorm | x: Tensor(shape=(256, 128), dtype=torch.bfloat16, device=npu:0) | 76.7934 us | 0.1307 ms |
| TTXNorm | x: Tensor(shape=(256, 128), dtype=torch.bfloat16, device=npu:0) | 5.8922 us | 0.2764 ms |
| RefRoPE | cos: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0)<br>  k: Tensor(shape=(1, 8, 1024, 32), dtype=torch.float32, device=npu:0)<br>  q: Tensor(shape=(1, 32, 1024, 32), dtype=torch.float32, device=npu:0)<br>  sin: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0) | 265.7056 us | 0.2787 ms |
| TTXRoPE | cos: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0)<br>  k: Tensor(shape=(1, 8, 1024, 32), dtype=torch.float32, device=npu:0)<br>  q: Tensor(shape=(1, 32, 1024, 32), dtype=torch.float32, device=npu:0)<br>  sin: Tensor(shape=(1, 1, 1024, 32), dtype=torch.float32, device=npu:0) | 26.4126 us | 0.3510 ms |
| RefTopPFilter | logits: Tensor(shape=(120, 151936), dtype=torch.float32, device=npu:0)<br>  min_tokens_to_keep: 1<br>  topk: 1000<br>  topp: 0.7 | 1570.3276 us | 0.9938 ms |
| TTXTopPFilter | logits: Tensor(shape=(120, 151936), dtype=torch.float32, device=npu:0)<br>  min_tokens_to_keep: 1<br>  topk: 1000<br>  topp: 0.7 | 1227.4888 us | 0.9086 ms |
