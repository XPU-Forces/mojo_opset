| Op | Parameters | Traffic (GB) | Time (ms) | Bandwidth (GB/s) |
|----|------------|--------------|-----------|-------------------|
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=1024 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0004 | 0.0078 | 51.18 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=1024 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0004 | 0.0064 | 61.94 |
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=2048 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0008 | 0.0106 | 74.82 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=2048 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0008 | 0.0083 | 95.45 |
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=4096 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0016 | 0.0162 | 97.95 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=4096 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0016 | 0.0139 | 114.56 |
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=8192 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0032 | 0.0275 | 115.41 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=8192 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0032 | 0.0250 | 127.30 |
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=1024 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0110 | 0.0326 | 338.30 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=1024 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0110 | 0.0256 | 430.02 |
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=2048 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0220 | 0.0733 | 300.54 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=2048 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0220 | 0.0499 | 441.46 |
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=4096 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0441 | 0.1327 | 332.21 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=4096 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0441 | 0.0934 | 471.93 |
| ttx::dynamic_quant | impl=TTXDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_dynamic_quant tokens=8192 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0881 | 0.2578 | 341.84 |
| ixformer::dynamic_quant | impl=IxformerDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_dynamic_quant tokens=8192 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0881 | 0.1813 | 485.98 |
