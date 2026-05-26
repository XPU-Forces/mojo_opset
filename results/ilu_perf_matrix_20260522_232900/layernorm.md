| Op | Parameters | Traffic (GB) | Time (ms) | Bandwidth (GB/s) |
|----|------------|--------------|-----------|-------------------|
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=1024 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0005 | 0.0446 | 11.77 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=1024 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0005 | 0.0190 | 27.67 |
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=2048 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0010 | 0.0452 | 23.21 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=2048 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0010 | 0.0096 | 109.41 |
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=4096 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0021 | 0.0462 | 45.39 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=4096 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0021 | 0.0367 | 57.16 |
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=8192 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0042 | 0.0473 | 88.66 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=8192 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0042 | 0.0254 | 165.30 |
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=1024 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0147 | 0.0272 | 540.31 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=1024 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0147 | 0.0392 | 374.45 |
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=2048 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0294 | 0.0736 | 398.98 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=2048 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0294 | 0.0750 | 391.80 |
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=4096 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0587 | 0.1312 | 447.69 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=4096 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0587 | 0.1422 | 412.99 |
| ttx::layernorm | impl=TTXLayerNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=8192 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.1175 | 0.2412 | 486.87 |
| ixformer::layernorm | impl=IxformerLayerNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=8192 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.1175 | 0.2763 | 425.08 |
