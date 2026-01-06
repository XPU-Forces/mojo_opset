export ASCEND_RT_VISIBLE_DEVICES=0
export MOJO_BACKEND=ref

python3 ./mojo_opset/modeling/inference_demo.py --model_path /data07/data/Qwen3-8B --device npu --max_new_tokens 100
pkill -9 python*

# sleep 10
# python3 ./mojo_opset/modeling/inference_demo.py --model_path /data07/data/Qwen3-8B --device npu --max_new_tokens 100 --transformers
