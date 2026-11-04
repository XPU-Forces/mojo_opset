# KPool tail cache seeding on Ascend 950PR

`mojo_opset.functions.kpool_seed_tail_cache(cache, key, gate_score, tslot)`
updates the tail cache in place and returns `None`. This operator handles the
fixed KPool factor of 4 and head dimension of 128. The Triton backend is enabled
for `npu.a5.950pr`. Select `implementation="torch"` to run the reference.

The cache has shape `[blocks, 2, ring, 128]`. It accepts padding between blocks
or channels, with contiguous rows. The key and gate inputs have shape
`[tokens, 128]` and must match the cache dtype (BF16 or FP16). Slot indices are
signed int32 or int64. Negative slots skip the row. A valid row is omitted if
the row four positions later maps to the same cache block. The caller must
supply in-range nonnegative slots and unique destinations for selected rows.
The cache and inputs must use separate storage.

## Run the tests

Use the repository's NPU environment on a 950PR device. Initialize the NPU
backend as required by that environment before running pytest.

```sh
pytest tests/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a5.950pr
pytest tests/performance/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a5.950pr \
  --perf-warmup 5 --perf-repeats 20 --perf-output /tmp/kpool-a5.json
```

The accuracy cases compare the whole backing storage bit for bit, including
padding that must remain untouched. They exercise both data and index dtypes,
empty inputs, launch boundaries, negative slots and non-power-of-two rings.
Performance calls reuse fixed slot maps and overwrite the same cache locations.
