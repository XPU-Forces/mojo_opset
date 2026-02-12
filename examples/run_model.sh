#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0

# Determine the project root directory (parent of examples/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

run_wan() {
    local MODEL_DIR="$1"
    local ROOT_DIR
    ROOT_DIR="$(dirname "$PROJECT_ROOT")"
    local WAN_PATH="$ROOT_DIR/Wan2.2"
    local GEN_PY="$WAN_PATH/generate.py"
    local TI2V_PY="$WAN_PATH/wan/textimage2video.py"

    if [ ! -d "$WAN_PATH" ]; then
        echo "Wan2.2 not found, cloning to: $WAN_PATH"
        git clone https://github.com/Wan-Video/Wan2.2.git "$WAN_PATH"
    else
        echo "Wan2.2 repository detected: $WAN_PATH"
    fi

    if [ ! -f "$GEN_PY" ] || [ ! -f "$TI2V_PY" ]; then
        echo "Wan2.2 files missing"
        exit 1
    fi

    if command -v npu-smi >/dev/null 2>&1 && npu-smi info >/dev/null 2>&1; then
        echo "Detected NPU backend, inject transfer_to_npu import into generate.py"
        if ! grep -q '^from torch_npu\.contrib import transfer_to_npu' "$GEN_PY"; then
            sed -i '/^[[:space:]]*import[[:space:]]\+torch[[:space:]]*$/a from torch_npu.contrib import transfer_to_npu' "$GEN_PY"
        else
            echo "generate.py already contains transfer_to_npu import, skipping"
        fi  
    fi

    echo "Begin to inject mojo_opset modeling classes into textimage2video.py"
    cp "$TI2V_PY" "$TI2V_PY.bak"
    awk '
      /import/ && /WanModel/       { print "from mojo_opset.modeling.wan2_2.mojo_wan_model import WanModel"; next }
      /import/ && /T5EncoderModel/ { print "from mojo_opset.modeling.wan2_2.mojo_t5 import T5EncoderModel"; next }
      /import/ && /Wan2_2_VAE/     { print "from mojo_opset.modeling.wan2_2.mojo_vae2_2 import Wan2_2_VAE"; next }
      { print }
    ' "$TI2V_PY.bak" > "$TI2V_PY"

    echo "Running Wan2.2 TI2V inference with model at: ${MODEL_DIR}"
    python3 "$GEN_PY" \
      --task ti2v-5B \
      --size 1280*704 \
      --ckpt_dir "$MODEL_DIR" \
      --offload_model True \
      --convert_model_dtype \
      --t5_cpu
}

# Default model settings
DEFAULT_MODEL_REPO="Qwen/Qwen3-8B"
# Default local path inside project root if not specified
DEFAULT_LOCAL_PATH="$PROJECT_ROOT/Qwen3-8B"

# Use provided path or default
MODEL_PATH="${1:-$DEFAULT_LOCAL_PATH}"

# Check if model exists, if not download it
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at ${MODEL_PATH}. Checking modelscope..."
    
    # Check if modelscope is installed
    if ! python3 -c "import modelscope" &> /dev/null; then
        echo "Installing modelscope..."
        pip install modelscope
    fi
    
    echo "Downloading ${DEFAULT_MODEL_REPO} to ${MODEL_PATH}..."
    # Use python to download to ensure we control the path
    python3 -c "from modelscope import snapshot_download; snapshot_download('${DEFAULT_MODEL_REPO}', local_dir='${MODEL_PATH}', max_workers=8)"
fi

echo "Running inference with model at: ${MODEL_PATH}"
MODEL_NAME="$(basename "$MODEL_PATH")"
if [[ "$MODEL_NAME" == *Wan* ]]; then
    echo "Detected Wan model name: ${MODEL_NAME}, running Wan pipeline"
    run_wan "$MODEL_PATH"
else
    python3 "${PROJECT_ROOT}/mojo_opset/modeling/inference_demo.py" --model_path "${MODEL_PATH}" --device npu --max_new_tokens 100
fi

# Cleanup
pkill -9 python*
