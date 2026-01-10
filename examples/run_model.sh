#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0

# Determine the project root directory (parent of examples/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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
# Run the inference script using absolute path
python3 "${PROJECT_ROOT}/mojo_opset/modeling/inference_demo.py" --model_path "${MODEL_PATH}" --device npu --max_new_tokens 100

# Cleanup
pkill -9 python*
