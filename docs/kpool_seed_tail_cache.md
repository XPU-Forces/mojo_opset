# KPool tail cache seeding on Ascend A2 and 950PR

`mojo_opset.functions.kpool_seed_tail_cache(cache, key, gate_score, tslot)`
seeds the paged tail cache for GLM-5.3-Flash prefill. It writes K and gate
in place and returns `None`. The public API follows the
[vLLM GPU implementation](https://github.com/vllm-project/vllm/blob/3d81a510f9ebc802a7864ced38832182295f4cad/vllm/models/glm5next/nvidia/ops/kpool_compress.py#L435-L463)
for BF16 payloads. The Ascend port uses KPool=4 and head dimension 128.

The cache has shape `[blocks, 2, ring, 128]` with contiguous rows and may
have padding between blocks or channels. K and gate are contiguous
`[tokens, 128]` tensors. All three tensors use BF16. The contiguous slot
tensor uses signed int32 or int64. Negative slots skip the row. A valid
row is omitted if the row four positions later maps to the same cache
block. Nonnegative slots must be in range, destinations of selected
rows must be unique, and cache storage must be separate from the inputs.

## Kernel design

- A2 checks 256 rows at a time for two four-row selected runs. It copies
  each run together when its destinations follow ring order. Other layouts
  use the 16-row bitmap fallback. The high word of an INT64 slot provides
  the sign check and the low word supplies the valid slot address.
- 950PR uses one JIT with paths for tiny, short, and general inputs. Its
  general path autotunes the tile size.

## Accuracy and performance cases

Both test entry points use the same 12 BF16 cases from the GPU comparison
profile. The fixtures preserve the original request lengths, slot maps,
and cache layout. Accuracy checks the complete backing storage, including
padding and guards, and checks that the three inputs are unchanged.

| Case | Workload | Total rows | Ring | Slot dtype |
| --- | --- | ---: | ---: | --- |
| 02 | Single request | 1 | 8 | int64 |
| 05 | Single request | 4 | 8 | int32 |
| 06 | Single request | 5 | 16 | int64 |
| 10 | Single request | 9 | 4 | int64 |
| 13 | Single request | 33 | 4 | int32 |
| 16 | Single request | 129 | 4 | int64 |
| 17 | Mixed request lengths | 320 | 4 | int32 |
| 19 | Padded cache and skipped slots | 183 | 16 | int32 |
| 21 | 32 requests × 33 tokens | 1056 | 8 | int32 |
| 22 | 128 requests × 128 tokens | 16384 | 16 | int64 |
| 31 | One request × 16384 tokens | 16384 | 4 | int32 |
| 32 | Four requests with lengths 255, 256, 257, 256 | 1024 | 8 | int64 |

## Reproduce

```sh
pytest tests/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a2
pytest tests/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a5.950pr
pytest tests/performance/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a5.950pr \
  --perf-warmup 5 --perf-repeats 20 --perf-output /tmp/kpool-a5.json
```

The repository performance test uses rotating input instances. The explicit
cold-L2 A2 measurement in the PR uses a separate protocol.
