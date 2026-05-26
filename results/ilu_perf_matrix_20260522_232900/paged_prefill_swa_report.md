| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| paged_prefill_swa | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 1.7182 | 0.4892 | 0.2847 |
| paged_prefill_swa | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 4.6472 | 1.4149 | 0.3045 |
| paged_prefill_swa | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 7.5474 | 2.3408 | 0.3101 |
| paged_prefill_swa | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 10.5736 | 3.2785 | 0.3101 |
| paged_prefill_swa | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 1.1362 | 0.3344 | 0.2943 |
| paged_prefill_swa | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 3.0370 | 0.9577 | 0.3153 |
| paged_prefill_swa | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 5.0004 | 1.5731 | 0.3146 |
| paged_prefill_swa | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 7.0151 | 2.1975 | 0.3133 |
| paged_prefill_swa | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 2.3058 | 0.4896 | 0.2123 |
| paged_prefill_swa | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 6.2609 | 1.4173 | 0.2264 |
| paged_prefill_swa | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 10.2247 | 2.3414 | 0.2290 |
| paged_prefill_swa | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 14.4399 | 3.2778 | 0.2270 |
| paged_prefill_swa | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 1.5530 | 0.3400 | 0.2189 |
| paged_prefill_swa | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 4.2204 | 0.9574 | 0.2269 |
| paged_prefill_swa | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 6.9272 | 1.5828 | 0.2285 |
| paged_prefill_swa | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 kv_extra=0 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 9.6261 | 2.2024 | 0.2288 |
