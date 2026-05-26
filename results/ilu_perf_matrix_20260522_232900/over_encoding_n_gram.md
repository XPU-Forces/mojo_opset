| Op | Parameters | Traffic (GB) | Time (ms) | Bandwidth (GB/s) |
|----|------------|--------------|-----------|-------------------|
| ttx::over_encoding_n_gram | impl=TTXOverEncodingNGram MOJO_BACKEND=ttx traffic_obs=TTXOverEncodingNGramObserver case=parametrized-prefill mode=prefill B=2 S=64,64 vocab=10086 max_gram=7 G=12 | 0.0000 | 0.0260 | 0.52 |
| ixformer::over_encoding_n_gram | impl=IxformerOverEncodingNGram MOJO_BACKEND=ixformer traffic_obs=IxformerOverEncodingNGramObserver case=parametrized-prefill mode=prefill B=2 S=64,64 vocab=10086 max_gram=7 G=12 | 0.0000 | 0.1000 | 0.14 |
| ttx::over_encoding_n_gram | impl=TTXOverEncodingNGram MOJO_BACKEND=ttx traffic_obs=TTXOverEncodingNGramObserver case=parametrized-decode mode=decode B=128 S=1 vocab=10086 max_gram=7 G=12 | 0.0000 | 0.0106 | 1.57 |
| ixformer::over_encoding_n_gram | impl=IxformerOverEncodingNGram MOJO_BACKEND=ixformer traffic_obs=IxformerOverEncodingNGramObserver case=parametrized-decode mode=decode B=128 S=1 vocab=10086 max_gram=7 G=12 | 0.0000 | 0.0490 | 0.40 |
| ttx::over_encoding_n_gram | impl=TTXOverEncodingNGram MOJO_BACKEND=ttx traffic_obs=TTXOverEncodingNGramObserver case=additional-prefill mode=prefill B=3 S=5,7,9 vocab=257 max_gram=5 G=8 | 0.0000 | 0.0805 | 0.02 |
| ixformer::over_encoding_n_gram | impl=IxformerOverEncodingNGram MOJO_BACKEND=ixformer traffic_obs=IxformerOverEncodingNGramObserver case=additional-prefill mode=prefill B=3 S=5,7,9 vocab=257 max_gram=5 G=8 | 0.0000 | 0.0989 | 0.02 |
| ttx::over_encoding_n_gram | impl=TTXOverEncodingNGram MOJO_BACKEND=ttx traffic_obs=TTXOverEncodingNGramObserver case=additional-decode mode=decode B=48 S=1 vocab=257 max_gram=5 G=8 | 0.0000 | 0.0076 | 0.58 |
| ixformer::over_encoding_n_gram | impl=IxformerOverEncodingNGram MOJO_BACKEND=ixformer traffic_obs=IxformerOverEncodingNGramObserver case=additional-decode mode=decode B=48 S=1 vocab=257 max_gram=5 G=8 | 0.0000 | 0.0483 | 0.11 |
