python3 inference_demo.py --model_path /data07/data/Qwen3-8B --device npu --max_new_tokens 100
pkill -9 python*
sleep 10
python3 inference_demo.py --model_path /data07/data/Qwen3-8B --device npu --max_new_tokens 100 --transformers
