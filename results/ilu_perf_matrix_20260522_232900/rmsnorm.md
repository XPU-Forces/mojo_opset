| Op | Parameters | Traffic (GB) | Time (ms) | Bandwidth (GB/s) |
|----|------------|--------------|-----------|-------------------|
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=1024 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0005 | 0.0099 | 53.25 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=1024 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0005 | 0.0061 | 85.77 |
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=2048 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0010 | 0.0101 | 104.28 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=2048 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0010 | 0.0063 | 165.47 |
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=4096 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0021 | 0.0105 | 199.97 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=4096 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0021 | 0.0070 | 301.45 |
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=8192 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0042 | 0.0113 | 370.44 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=8192 hidden_size=128 eps=1e-05 dtype=bf16 | 0.0042 | 0.0082 | 511.53 |
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=1024 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0147 | 0.0299 | 491.02 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=1024 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0147 | 0.0160 | 916.63 |
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=2048 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0294 | 0.0696 | 422.22 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=2048 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0294 | 0.0556 | 528.30 |
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=4096 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0587 | 0.1229 | 477.85 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=4096 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.0587 | 0.1065 | 551.35 |
| ttx::rmsnorm | impl=TTXRMSNorm MOJO_BACKEND=ttx traffic_obs=inline_norm tokens=8192 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.1174 | 0.2280 | 515.23 |
| ixformer::rmsnorm | impl=IxformerRMSNorm MOJO_BACKEND=ixformer traffic_obs=inline_norm tokens=8192 hidden_size=3584 eps=1e-05 dtype=bf16 | 0.1174 | 0.2070 | 567.32 |
