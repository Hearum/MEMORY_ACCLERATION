#!/bin/bash

# ===============================
# 用户可修改部分
# ===============================
MODEL_PATH="/mnt/data/models/Llama-3.2-3B-Instruct"
PORT=30065
CUDA_DEVICES=4
LOG_DIR="/home/shm/document/MEMORY_ACCLERATION/log"
MODEL_NAME="HippoRAG"
DATASETS="locomo10"
# locomo10
# longmemeval_s
# longmemeval_m
# longmemeval_oracle

SERVER_ENV="sglang"
PIPELINE_ENV="rag3"
# ===============================

mkdir -p "$LOG_DIR"
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/sglang_${MODEL_NAME}_${DATASETS}_$PORT_$NOW.log"

echo "=== Starting SGLang server ==="
echo "Logs will be saved to $LOG_FILE"

# 激活 conda 并启动 SGLang server
source /home/shm/anaconda3/etc/profile.d/conda.sh
conda activate $SERVER_ENV

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -m sglang.launch_server \
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
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "SGLang server PID: $SERVER_PID"

# 等待 server 启动
echo "Waiting for SGLang server to be ready..."
TIMEOUT=120
for i in $(seq 1 $TIMEOUT); do
    if nc -z localhost $PORT; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

if ! nc -z localhost $PORT; then
    echo "ERROR: SGLang server failed to start after $TIMEOUT seconds."
    tail -n 20 "$LOG_FILE"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# ===============================
# 运行 pipeline
# ===============================
echo "=== Running pipeline in $PIPELINE_ENV environment ==="
conda activate $PIPELINE_ENV

export OPENAI_API_KEY="nope"
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export HF_ENDPOINT=https://hf-mirror.com

python3 /home/shm/document/MEMORY_ACCLERATION/run_pipeline.py \
    --models $MODEL_NAME \
    --datasets $DATASETS

# ===============================
# 停止 server
# ===============================
echo "=== Stopping SGLang server ==="
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "SGLang server stopped."
