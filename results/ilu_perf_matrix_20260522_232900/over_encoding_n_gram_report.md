| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| over_encoding_n_gram | case=parametrized-prefill mode=prefill B=2 S=64,64 vocab=10086 max_gram=7 G=12 | 0.0260 | 0.1000 | 3.8462 |
| over_encoding_n_gram | case=parametrized-decode mode=decode B=128 S=1 vocab=10086 max_gram=7 G=12 | 0.0106 | 0.0490 | 4.6226 |
| over_encoding_n_gram | case=additional-prefill mode=prefill B=3 S=5,7,9 vocab=257 max_gram=5 G=8 | 0.0805 | 0.0989 | 1.2286 |
| over_encoding_n_gram | case=additional-decode mode=decode B=48 S=1 vocab=257 max_gram=5 G=8 | 0.0076 | 0.0483 | 6.3553 |
