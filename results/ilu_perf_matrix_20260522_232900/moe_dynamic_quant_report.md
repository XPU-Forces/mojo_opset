| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| moe_dynamic_quant | tokens=1024 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.3732 | 0.5581 | 1.4954 |
| moe_dynamic_quant | tokens=2048 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.3824 | 0.5879 | 1.5374 |
| moe_dynamic_quant | tokens=4096 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.3942 | 0.6123 | 1.5533 |
| moe_dynamic_quant | tokens=8192 hidden_size=128 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.4394 | 0.6849 | 1.5587 |
| moe_dynamic_quant | tokens=1024 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.6480 | 1.0387 | 1.6029 |
| moe_dynamic_quant | tokens=2048 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 0.9135 | 1.5460 | 1.6924 |
| moe_dynamic_quant | tokens=4096 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 1.4496 | 2.4678 | 1.7024 |
| moe_dynamic_quant | tokens=8192 hidden_size=3584 experts=8 quant_dtype=int8 input_dtype=bf16 | 2.5257 | 4.3617 | 1.7269 |
