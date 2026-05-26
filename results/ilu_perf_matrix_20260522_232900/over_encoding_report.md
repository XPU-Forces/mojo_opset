| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| over_encoding | case=parametrized-prefill mode=prefill B=2 S=64,64 vocab=10086 ori_dim=1536 oe_dim=192 max_gram=7 G=12 dtype=fp32 | 0.2483 | 0.2554 | 1.0286 |
| over_encoding | case=parametrized-decode mode=decode B=128 S=1 vocab=10086 ori_dim=1536 oe_dim=192 max_gram=7 G=12 dtype=fp32 | 0.1974 | 0.2271 | 1.1505 |
| over_encoding | case=additional-prefill mode=prefill B=3 S=5,7,9 vocab=257 ori_dim=640 oe_dim=80 max_gram=5 G=8 dtype=fp32 | 0.2035 | 0.2264 | 1.1125 |
| over_encoding | case=additional-decode mode=decode B=48 S=1 vocab=257 ori_dim=320 oe_dim=40 max_gram=5 G=8 dtype=fp32 | 0.1590 | 0.1871 | 1.1767 |
