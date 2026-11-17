# vllm-ascend integration: `fused_qk_norm_rope` dispatch

`qk_norm_rope_infer` is the Ascend Triton port of vLLM-Omni's `fused_qk_norm_rope`
(`vllm_omni/diffusion/layers/fused_qk_norm_rope.py`), the MiniMax-H3 contract:
`head_dim=128`, `rotary_dim=96`, BF16, `[tokens, heads, head_dim]`, half-split pairing.

## Design

Default production path for H3 half-split:

1. **Persist schedule** — `grid ≈ num_vectorcore`, each core strides tokens with `tl.range`
   (triton-ascend HighPerfGuide / profiling Method 2).
2. **extract/insert half-split** — Ascend `tl.extra.cann.extension.extract_slice` /
   `insert_slice` for the two rotary halves (rotary_half=48), avoiding partner re-loads.
3. One launch per Q/K stream.

Interleaved (GPT-NeoX) pairing and non-power-of-two `head_dim` keep the general multi-head
tile kernel for coverage.

Numerical contract (unchanged vs upstream):

1. RMSNorm statistics once per head; rotary pairs share `inv_rms`.
2. Normalize → round to BF16 → rotate in FP32 → store.
3. Scale left-to-right: `x * inv_rms * weight`.
4. Dims `>= rotary_dim` pass through as the BF16-rounded norm value.

## HiAscend acceptance evidence (taskbook MBU)

Calibration uses on-device `copy_` bandwidth; RW bytes follow the taskbook formula
`2*(4*T*H*128 + T*96 + 2*128)`. Warmup 5 + active 30 profiler samples.

### A5 / Ascend950PR — **MBU gate PASS**

| tokens×heads | gate | latency | MBU | result |
| --- | --- | --- | --- | --- |
| 32×8 | relative_A | 0.003 ms | 6.7% | PASS |
| 320×8 | relative_A | 0.009 ms | 20.7% | PASS |
| 320×56 | mbu≥60% | 0.021 ms | **64.6%** | PASS |
| 1024×56 | mbu≥60% | 0.057 ms | **74.3%** | PASS |
| 4096×56 | mbu≥60% | 0.212 ms | **80.0%** | PASS |
| 16384×56 | mbu≥60% | 0.828 ms | **81.9%** | PASS |
| 69888×56 | mbu≥60% | 3.513 ms | **82.3%** | PASS |
| 4096×57 | mbu≥50% | 0.217 ms | **79.5%** | PASS |

`copy_` ≈ 1390 GB/s. Stack: triton-ascend 3.6.0 + CANN 9.2 + torch_npu 2.10.

### A3 / 910_9382 — precision + relative_A PASS; MBU ~56% (gap ~4pt)

Persist default: 4096×56 **~56.9%** MBU (`copy_` ≈ 1163 GB/s). Tail 4096×57 ≥50% PASS.
Compile-option / autotune sweeps did not close the remaining gap; profile shows high
`aiv_vec_ratio` (~0.87–0.90) with residual descriptor / conversion cost.

### A2 / 910B3 — precision + relative_A PASS; MBU ~47%

Persist vs classic: **+1.4pt** on large shapes (`copy_` ≈ 1143 GB/s, 40 vector cores).
Still below the 60% line; gap analysis attached in the private delivery package.

## Dispatch note

`mojo_opset` config selects `triton` for this op. Consumers that also ship the
`npu_rms_norm` + `npu_rotary_mul` composition may still prefer that path on A2/A3 for
bandwidth-bound shapes until the remaining MBU gap is closed; on A5/950PR the Triton
persist path meets the taskbook MBU line and is the preferred route.

## Escape hatches

| env | effect |
| --- | --- |
| `MOJO_T1_TILE=1` | force the general tile kernel (disable persist extract) |
| `MOJO_T1_HEADS` / `MOJO_T1_ROWS` / `MOJO_T1_WAVES` | tile-path tuning only |
