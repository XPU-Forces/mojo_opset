| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| rmsnorm | tokens=1024 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0099 | 0.0061 | 0.6162 |
| rmsnorm | tokens=2048 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0101 | 0.0063 | 0.6238 |
| rmsnorm | tokens=4096 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0105 | 0.0070 | 0.6667 |
| rmsnorm | tokens=8192 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0113 | 0.0082 | 0.7257 |
| rmsnorm | tokens=1024 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0299 | 0.0160 | 0.5351 |
| rmsnorm | tokens=2048 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0696 | 0.0556 | 0.7989 |
| rmsnorm | tokens=4096 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.1229 | 0.1065 | 0.8666 |
| rmsnorm | tokens=8192 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.2280 | 0.2070 | 0.9079 |
