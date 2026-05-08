source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /data00/xupan/encoder_project/mojo_opset
export MOJO_BACKEND=torch
python3 -m pytest mojo_opset/tests/accuracy/operators/test_position_embedding.py -k "vision_rotary_embedding_2d or apply_vision_rope_2d"
