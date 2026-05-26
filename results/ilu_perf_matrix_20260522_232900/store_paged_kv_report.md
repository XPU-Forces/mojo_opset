| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| store_paged_kv | B=2 tokens=2048 Hkv=4 D=128 q_len/seq=1024 block=16 dtype=bf16 | 0.0280 | 0.0532 | 1.9000 |
| store_paged_kv | B=2 tokens=4096 Hkv=4 D=128 q_len/seq=2048 block=16 dtype=bf16 | 0.0468 | 0.0910 | 1.9444 |
| store_paged_kv | B=2 tokens=6144 Hkv=4 D=128 q_len/seq=3072 block=16 dtype=bf16 | 0.0866 | 0.1303 | 1.5046 |
| store_paged_kv | B=2 tokens=8192 Hkv=4 D=128 q_len/seq=4096 block=16 dtype=bf16 | 0.1218 | 0.1727 | 1.4179 |
| store_paged_kv | B=2 tokens=2048 Hkv=8 D=128 q_len/seq=1024 block=16 dtype=bf16 | 0.0452 | 0.0687 | 1.5199 |
| store_paged_kv | B=2 tokens=4096 Hkv=8 D=128 q_len/seq=2048 block=16 dtype=bf16 | 0.2513 | 0.1270 | 0.5054 |
| store_paged_kv | B=2 tokens=6144 Hkv=8 D=128 q_len/seq=3072 block=16 dtype=bf16 | 0.1914 | 0.1963 | 1.0256 |
| store_paged_kv | B=2 tokens=8192 Hkv=8 D=128 q_len/seq=4096 block=16 dtype=bf16 | 0.2495 | 0.2654 | 1.0637 |
| store_paged_kv | B=2 tokens=2048 Hkv=4 D=128 q_len/seq=1024 block=128 dtype=bf16 | 0.0524 | 0.0584 | 1.1145 |
| store_paged_kv | B=2 tokens=4096 Hkv=4 D=128 q_len/seq=2048 block=128 dtype=bf16 | 0.0655 | 0.0967 | 1.4763 |
| store_paged_kv | B=2 tokens=6144 Hkv=4 D=128 q_len/seq=3072 block=128 dtype=bf16 | 0.0972 | 0.1362 | 1.4012 |
| store_paged_kv | B=2 tokens=8192 Hkv=4 D=128 q_len/seq=4096 block=128 dtype=bf16 | 0.2737 | 0.1785 | 0.6522 |
| store_paged_kv | B=2 tokens=2048 Hkv=8 D=128 q_len/seq=1024 block=128 dtype=bf16 | 0.0985 | 0.0791 | 0.8030 |
| store_paged_kv | B=2 tokens=4096 Hkv=8 D=128 q_len/seq=2048 block=128 dtype=bf16 | 0.1447 | 0.1460 | 1.0090 |
| store_paged_kv | B=2 tokens=6144 Hkv=8 D=128 q_len/seq=3072 block=128 dtype=bf16 | 0.2065 | 0.2179 | 1.0552 |
| store_paged_kv | B=2 tokens=8192 Hkv=8 D=128 q_len/seq=4096 block=128 dtype=bf16 | 0.2648 | 0.2786 | 1.0521 |
