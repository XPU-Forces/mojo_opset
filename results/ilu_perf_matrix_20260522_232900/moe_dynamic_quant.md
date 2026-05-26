| Op | Parameters | Traffic (GB) | Time (ms) | Bandwidth (GB/s) |
|----|------------|--------------|-----------|-------------------|
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=1024 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0004 | 0.3732 | 1.08 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=1024 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0004 | 0.5581 | 0.72 |
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=2048 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0008 | 0.3824 | 2.09 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=2048 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0008 | 0.5879 | 1.36 |
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=4096 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0016 | 0.3942 | 4.04 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=4096 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0016 | 0.6123 | 2.60 |
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=8192 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0032 | 0.4394 | 7.24 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=8192 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0032 | 0.6849 | 4.65 |
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=1024 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0111 | 0.6480 | 17.17 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=1024 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0111 | 1.0387 | 10.71 |
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=2048 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0221 | 0.9135 | 24.24 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=2048 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0221 | 1.5460 | 14.32 |
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=4096 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0442 | 1.4496 | 30.47 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=4096 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0442 | 2.4678 | 17.90 |
| ttx::moe_dynamic_quant | impl=TTXMoEDynamicQuant MOJO_BACKEND=ttx traffic_obs=inline_moe_dynamic_quant tokens=8192 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0882 | 2.5257 | 34.93 |
| ixformer::moe_dynamic_quant | impl=IxformerMoEDynamicQuant MOJO_BACKEND=ixformer traffic_obs=inline_moe_dynamic_quant tokens=8192 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.0882 | 4.3617 | 20.23 |
