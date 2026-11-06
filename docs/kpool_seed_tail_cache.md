# KPool tail cache seeding on Ascend A2 and 950PR

`mojo_opset.functions.kpool_seed_tail_cache(cache, key, gate_score, tslot)`
updates the tail cache in place and returns `None`. This operator handles the
fixed KPool factor of 4 and head dimension of 128. The Triton backend is enabled
for `npu.a2` and `npu.a5.950pr`. Select `implementation="torch"` to run the reference.

The cache has shape `[blocks, 2, ring, 128]`. It accepts padding between blocks
or channels, with contiguous rows. The key and gate inputs have shape
`[tokens, 128]` and must match the cache dtype (BF16 or FP16). Slot indices are
signed int32 or int64. Negative slots skip the row. A valid row is omitted if
the row four positions later maps to the same cache block. The caller must
supply in-range nonnegative slots and unique destinations for selected rows.
The cache and inputs must use separate storage.

## Platform implementations

- A2 uses 256-row batches with runtime recognition of two four-row selected
  runs. It copies each run together when its destinations are consecutive
  along the ring; otherwise it uses the 16-row bitmap fallback. INT64 slots
  are read as high/low INT32 words, keeping the sign check separate from
  address calculation.
- 950PR retains its existing single-JIT tiny, short and general paths with
  autotuning. The A2 algorithm is not selected on A5.

Both implementations use the same API and input contract. Cache views with
nonzero storage offsets are accepted; their pointers already include the
offset.

## Run the tests

Use the repository's NPU environment on the target device. Initialize the NPU
backend as required by that environment before running pytest. For A2, use:

```sh
pytest tests/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a2
```

For 950PR:

```sh
pytest tests/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a5.950pr
pytest tests/performance/functions/test_kpool_seed_tail_cache.py --mojo-target npu.a5.950pr \
  --perf-warmup 5 --perf-repeats 20 --perf-output /tmp/kpool-a5.json
```

The accuracy cases compare the whole backing storage bit for bit, including
padding that must remain untouched. They exercise both data and index dtypes,
empty inputs, launch boundaries, negative slots and non-power-of-two rings.
Performance calls reuse fixed slot maps and overwrite the same cache locations.


## Performance evidence

The A2 standalone final short implementation measured case22 (BF16,
128 requests x 128 tokens, D=128, RING=16) at 9.277 us on Ascend910B3.
Every warmup/sample was preceded by a synchronized 192 MiB FP32 reduction
on the same stream; eviction tasks were excluded from target task timing.
Effective MBU was 4.469%, using a same-device, same-cold-policy measured
bandwidth denominator of 1580.873 GB/s. This is one representative case,
not a full-shape throughput or hardware-acceptance claim.

A5 retains its separately validated implementation. Its historical 24-case
ABBA measurements use a different cache/timing protocol and must not be
used as a matched A2/A5 speed comparison.

The repository performance entry point uses the shared rotating-input
harness. Its results are a separate metric from the explicit cold-L2 A2
measurements above.


## Integration validation

The A2 integration passed all 33 repository pytest cases on Ascend910B3
without skips. A separate run through the public API passed 128/128
bitwise checks (64 fixture cases, two trials each), comparing the complete
backing storage, guards and all inputs against the independent reference.
The A2 JIT AST matches the standalone final short implementation. The A5
kernel source is unchanged; its historical standalone validation is retained,
and the expanded repository tests have not been rerun on 950PR.
