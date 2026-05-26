| Op | Parameters | TTX Time (ms) | Ixformer Time (ms) | SpeedUp |
|----|------------|---------------|--------------------|---------|
| dynamic_quant | tokens=1024 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0078 | 0.0064 | 0.8205 |
| dynamic_quant | tokens=2048 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0106 | 0.0083 | 0.7830 |
| dynamic_quant | tokens=4096 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0162 | 0.0139 | 0.8580 |
| dynamic_quant | tokens=8192 hidden_size=128 quant_dtype=int8 input_dtype=bf16 | 0.0275 | 0.0250 | 0.9091 |
| dynamic_quant | tokens=1024 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0326 | 0.0256 | 0.7853 |
| dynamic_quant | tokens=2048 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.0733 | 0.0499 | 0.6808 |
| dynamic_quant | tokens=4096 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.1327 | 0.0934 | 0.7038 |
| dynamic_quant | tokens=8192 hidden_size=3584 quant_dtype=int8 input_dtype=bf16 | 0.2578 | 0.1813 | 0.7033 |
