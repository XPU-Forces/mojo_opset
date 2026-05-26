| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| layernorm | tokens=1024 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0446 | 0.0190 | 0.4260 |
| layernorm | tokens=2048 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0452 | 0.0096 | 0.2124 |
| layernorm | tokens=4096 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0462 | 0.0367 | 0.7944 |
| layernorm | tokens=8192 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0473 | 0.0254 | 0.5370 |
| layernorm | tokens=1024 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0272 | 0.0392 | 1.4412 |
| layernorm | tokens=2048 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0736 | 0.0750 | 1.0190 |
| layernorm | tokens=4096 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.1312 | 0.1422 | 1.0838 |
| layernorm | tokens=8192 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.2412 | 0.2763 | 1.1455 |
