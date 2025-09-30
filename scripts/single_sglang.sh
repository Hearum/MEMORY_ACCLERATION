#!/bin/bash

# 配置
MODEL_PATH="/mnt/data/models/Llama-3.2-3B-Instruct"
PORT=30001
CUDA_DEVICES=2
LOG_DIR="/home/shm/document/MEMORY_ACCLERATION/log"

mkdir -p "$LOG_DIR"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/sglang_$NOW.log"

echo "Starting SGLang server..."
echo "Logs will be saved to $LOG_FILE"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model-path $MODEL_PATH \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 64000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
  --enable-mixed-chunk \
  --enable-metrics \
  > "$LOG_FILE" 2>&1
