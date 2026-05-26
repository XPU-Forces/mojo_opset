| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| paged_prefill_swa_with_kv_dequant | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 100.3948 | 0.6601 | 0.0066 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 401.7306 | 1.9684 | 0.0049 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 901.5690 | 3.0812 | 0.0034 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 1604.8556 | 4.2930 | 0.0027 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 67.1010 | 0.4604 | 0.0069 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 267.1180 | 1.2813 | 0.0048 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 603.3144 | 2.0930 | 0.0035 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 block=16 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 1068.7187 | 2.9091 | 0.0027 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=2048 Hq=48 Hkv=8 D=128 q_len/seq=1024 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 99.6538 | 0.6597 | 0.0066 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=4096 Hq=48 Hkv=8 D=128 q_len/seq=2048 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 398.7133 | 1.8751 | 0.0047 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=6144 Hq=48 Hkv=8 D=128 q_len/seq=3072 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 897.7432 | 3.0804 | 0.0034 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=8192 Hq=48 Hkv=8 D=128 q_len/seq=4096 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 1596.5517 | 4.2823 | 0.0027 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=2048 Hq=32 Hkv=8 D=128 q_len/seq=1024 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 66.6517 | 0.4594 | 0.0069 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=4096 Hq=32 Hkv=8 D=128 q_len/seq=2048 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 265.2161 | 1.2801 | 0.0048 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=6144 Hq=32 Hkv=8 D=128 q_len/seq=3072 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 596.7047 | 2.0892 | 0.0035 |
| paged_prefill_swa_with_kv_dequant | B=2 Tq=8192 Hq=32 Hkv=8 D=128 q_len/seq=4096 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 1061.4838 | 2.8973 | 0.0027 |
