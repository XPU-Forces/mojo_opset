#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0

# Determine the project root directory (parent of examples/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Default dit code repo path if not specified
DEFAULT_DIT_PATH="$(dirname "$PROJECT_ROOT")/Wan2.2"

# Use provided dit code repo path or default
DIT_PATH="${1:-$DEFAULT_DIT_PATH}"

# git clone dit code repo if not exists
if [ ! -d "$DIT_PATH" ]; then
    echo "Dit code repo not found, cloning to: $DIT_PATH"
    git clone https://github.com/Wan-Video/Wan2.2.git "$DIT_PATH"
else
    echo "Dit code repo detected: $DIT_PATH"
fi
export PYTHONPATH="${DIT_PATH}:${PYTHONPATH}"

# Default model settings
DEFAULT_MODEL_REPO="Wan-AI/Wan2.2-TI2V-5B"
# Default local path inside project root if not specified
DEFAULT_LOCAL_PATH="$DIT_PATH/Wan2.2-TI2V-5B"

# Use provided path or default
MODEL_PATH="${2:-$DEFAULT_LOCAL_PATH}"

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
python3 "${PROJECT_ROOT}/examples/wan2_2_patch.py" --ckpt_dir "${MODEL_PATH}"

# Cleanup
pkill -9 python*