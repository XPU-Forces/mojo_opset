# A5 TopK inference

This operator provides DeepSelect BF16 TopK and FP32 rescan TopK through
`functions.topk_infer` on 950PR and 950DT. It is forward-only; A5 defaults to
`native`, with `implementation="torch"` selecting the independent reference.

BF16 supports `[B,N]`, `N < 2**21`, K=1–4096, `sorted=False` and int32 indices.
Options include `end`, `output_idx_offset`, `sorted_index`, `return_value=False`
and K>N with padding. FP32 supports `[B,N]`, N=1–1024 and K=1–min(N,32), with
`sorted=True`, int64 indices and `return_value=True`; end, offsets and index
sorting are unsupported. FP32 ties choose the smaller original index.

Inputs require ND storage and a contiguous last dimension. BF16 input row
strides and all output row strides must be aligned to 32 bytes; FP32 input row
strides need not be aligned. Outputs may be noncontiguous.

Build with:

    bash native/build.sh npu_a5 topk

Both paths compile ahead of time into `libmojo_topk.so` for the shared A5
`dav-3510` architecture. FP32 queries the vector core count at runtime; there
is no PR/DT-specific launch path. The ASC target uses C++20 and supports CANN
9.1.T103/9.2.0 with GCC 12.2 default headers. Torch/torch-npu/CANN must match the
build environment. Inference uses packaged binaries without a compiler or device
headers; release wheels require a complete `npu_a5 all` provider build.

BF16 uses threshold filtering and radix selection with 512/1024/4096 capacity
tiers. FP32 caches each row in a 32-lane warp and selects K winners through
repeated max/index reductions. Full-bucket and fixed-prefix loads reduce boundary
checks across the 256/512/640/1024 capacity tiers.

Functional calls (`output_idx=None`) support fake tensors, forward fullgraph
tracing and NPU Graph capture/replay. Import and fake execution do not load the
native library. Warm dispatch, the library and Dynamo before capture; shapes,
dtypes, strides, K, options and buffer addresses stay fixed during replay.
Preallocated `output_idx` is eager-only. No backward is registered.

NaNs in the valid range abort native execution by default. Setting
`abort_when_nan_found=False` writes index sentinel `0x3f3f3f3f` for the affected
row; its values are unspecified.
