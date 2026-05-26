| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| paged_decode_swa | B=8 Hq=16 Hkv=4 D=128 kv_len/seq=1024 block=32 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.3282 | 0.1684 | 0.5131 |
| paged_decode_swa | B=8 Hq=16 Hkv=4 D=96 kv_len/seq=1024 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.1627 | 0.1510 | 0.9281 |
| paged_decode_swa | B=8 Hq=16 Hkv=4 D=128 kv_len/seq=8192 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.1708 | 0.3464 | 2.0281 |
| paged_decode_swa | B=1 Hq=8 Hkv=4 D=128 kv_len/seq=1024 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.0489 | 0.1303 | 2.6646 |
| paged_decode_swa | B=1 Hq=8 Hkv=4 D=128 kv_len/seq=20480 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.0507 | 0.1920 | 3.7870 |
| paged_decode_swa | B=1 Hq=16 Hkv=4 D=128 kv_len/seq=1024 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.2409 | 0.1347 | 0.5592 |
| paged_decode_swa | B=1 Hq=16 Hkv=4 D=128 kv_len/seq=20480 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.0521 | 0.2057 | 3.9482 |
| paged_decode_swa | B=1 Hq=24 Hkv=8 D=128 kv_len/seq=1024 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.0598 | 0.1408 | 2.3545 |
| paged_decode_swa | B=1 Hq=24 Hkv=8 D=128 kv_len/seq=20480 block=128 gqa_layout=AABB global_w=4 local_w=1023 dtype=bf16 | 0.0633 | 0.2646 | 4.1801 |
