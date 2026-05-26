| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| group_gemm | G=1 G_req=8 M=1 K=128 N=512 trans_weight=False dtype=bf16 | 0.2502 | 0.0068 | 0.0272 |
| group_gemm | G=1 G_req=8 M=1 K=128 N=512 trans_weight=True dtype=bf16 | 0.2436 | 0.0066 | 0.0271 |
| group_gemm | G=1 G_req=8 M=1 K=128 N=1024 trans_weight=False dtype=bf16 | 0.2445 | 0.0087 | 0.0356 |
| group_gemm | G=1 G_req=8 M=1 K=128 N=1024 trans_weight=True dtype=bf16 | 0.2447 | 0.0083 | 0.0339 |
| group_gemm | G=1 G_req=8 M=1 K=128 N=2048 trans_weight=False dtype=bf16 | 0.2444 | 0.0086 | 0.0352 |
| group_gemm | G=1 G_req=8 M=1 K=128 N=2048 trans_weight=True dtype=bf16 | 0.2489 | 0.0084 | 0.0337 |
| group_gemm | G=1 G_req=8 M=1 K=128 N=10240 trans_weight=False dtype=bf16 | 0.2594 | 0.0104 | 0.0401 |
| group_gemm | G=1 G_req=8 M=1 K=128 N=10240 trans_weight=True dtype=bf16 | 0.2587 | 0.0099 | 0.0383 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=512 trans_weight=False dtype=bf16 | 0.2525 | 0.0177 | 0.0701 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=512 trans_weight=True dtype=bf16 | 0.2480 | 0.0170 | 0.0685 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=1024 trans_weight=False dtype=bf16 | 0.2508 | 0.0215 | 0.0857 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=1024 trans_weight=True dtype=bf16 | 0.2575 | 0.0196 | 0.0761 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=2048 trans_weight=False dtype=bf16 | 0.2634 | 0.0245 | 0.0930 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=2048 trans_weight=True dtype=bf16 | 0.2659 | 0.0234 | 0.0880 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=10240 trans_weight=False dtype=bf16 | 0.3511 | 0.0699 | 0.1991 |
| group_gemm | G=8 G_req=8 M=64 K=128 N=10240 trans_weight=True dtype=bf16 | 0.3221 | 0.0781 | 0.2425 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=512 trans_weight=False dtype=bf16 | 0.2533 | 0.0175 | 0.0691 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=512 trans_weight=True dtype=bf16 | 0.2453 | 0.0170 | 0.0693 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=1024 trans_weight=False dtype=bf16 | 0.2550 | 0.0217 | 0.0851 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=1024 trans_weight=True dtype=bf16 | 0.2511 | 0.0204 | 0.0812 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=2048 trans_weight=False dtype=bf16 | 0.2635 | 0.0241 | 0.0915 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=2048 trans_weight=True dtype=bf16 | 0.2563 | 0.0236 | 0.0921 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=10240 trans_weight=False dtype=bf16 | 0.3560 | 0.0727 | 0.2042 |
| group_gemm | G=8 G_req=8 M=128 K=128 N=10240 trans_weight=True dtype=bf16 | 0.3245 | 0.0595 | 0.1834 |
