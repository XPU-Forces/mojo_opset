| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 block=16 gqa_layout=AABB dtype=bf16 | 2.4275 | 0.4669 | 0.1923 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 block=16 gqa_layout=AABB dtype=bf16 | 8.3286 | 1.4420 | 0.1731 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 block=16 gqa_layout=AABB dtype=bf16 | 17.8349 | 2.9626 | 0.1661 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 block=16 gqa_layout=AABB dtype=bf16 | 30.8535 | 4.9980 | 0.1620 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 block=16 gqa_layout=AABB dtype=bf16 | 1.6727 | 0.3308 | 0.1978 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 block=16 gqa_layout=AABB dtype=bf16 | 5.6245 | 1.0011 | 0.1780 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 block=16 gqa_layout=AABB dtype=bf16 | 12.0426 | 2.0239 | 0.1681 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 block=16 gqa_layout=AABB dtype=bf16 | 20.7368 | 3.4066 | 0.1643 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 block=128 gqa_layout=AABB dtype=bf16 | 5.2449 | 0.4664 | 0.0889 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 block=128 gqa_layout=AABB dtype=bf16 | 18.1618 | 1.4525 | 0.0800 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 block=128 gqa_layout=AABB dtype=bf16 | 39.0583 | 2.9457 | 0.0754 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 block=128 gqa_layout=AABB dtype=bf16 | 67.5022 | 4.9900 | 0.0739 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 block=128 gqa_layout=AABB dtype=bf16 | 3.5358 | 0.3310 | 0.0936 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 block=128 gqa_layout=AABB dtype=bf16 | 12.5194 | 1.0015 | 0.0800 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 block=128 gqa_layout=AABB dtype=bf16 | 25.9825 | 2.0209 | 0.0778 |
| paged_prefill_gqa_with_kv_dequant | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 block=128 gqa_layout=AABB dtype=bf16 | 45.0916 | 3.3911 | 0.0752 |
