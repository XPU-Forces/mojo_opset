| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| paged_prefill_gqa | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 2.0723 | 0.4450 | 0.2147 |
| paged_prefill_gqa | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 7.1227 | 1.4138 | 0.1985 |
| paged_prefill_gqa | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 14.7398 | 2.9101 | 0.1974 |
| paged_prefill_gqa | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 24.9199 | 4.9414 | 0.1983 |
| paged_prefill_gqa | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 1.4130 | 0.3078 | 0.2178 |
| paged_prefill_gqa | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 4.8033 | 0.9574 | 0.1993 |
| paged_prefill_gqa | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 9.8578 | 1.9679 | 0.1996 |
| paged_prefill_gqa | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=16 gqa_layout=AABB dtype=bf16 | 16.7358 | 3.3243 | 0.1986 |
| paged_prefill_gqa | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 4.2969 | 0.4441 | 0.1034 |
| paged_prefill_gqa | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 15.4786 | 1.4103 | 0.0911 |
| paged_prefill_gqa | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 33.5500 | 2.9196 | 0.0870 |
| paged_prefill_gqa | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 58.5443 | 4.9481 | 0.0845 |
| paged_prefill_gqa | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 2.8618 | 0.3091 | 0.1080 |
| paged_prefill_gqa | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 10.3548 | 0.9602 | 0.0927 |
| paged_prefill_gqa | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 22.3207 | 1.9737 | 0.0884 |
| paged_prefill_gqa | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=128 gqa_layout=AABB dtype=bf16 | 39.1105 | 3.3381 | 0.0854 |
