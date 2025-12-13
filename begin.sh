
#!/bin/bash

MODEL_PATH="/mnt/data/models/zhipu-glm-4-9b-chat-1m"
# "/mnt/data/models/Llama-3.2-3B-Instruct"
# /mnt/data/models/glm-4-9b/

START_PORT=30050
PORT=$START_PORT
while ss -tuln | grep -q ":$PORT "; do
    PORT=$((PORT + 1))
done

CUDA_DEVICES=3
LOG_DIR="/home/shm/document/MEMORY_ACCLERATION/log"
MODEL_NAME="QA"
# MemoryOS
# Memo0
# HippoRAG
# QAs
# langmem
DATASETS="locomo10"
# locomo10
# longmemeval_s
# longmemeval_m
# longmemeval_oracle

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

RESULTS_DIR="/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_${MODEL_NAME}_${DATASETS}_mem"

export EXP_RESULTS_DIR="$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/sglang_${MODEL_NAME}_${DATASETS}_$PORT.log"

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
cp -v "$SCRIPT_PATH" "$RESULTS_DIR/${SCRIPT_NAME%.sh}_backup_.sh"

echo "Starting SGLang server..."
echo "Logs will be saved to $LOG_FILE"

# 1004000 
# 254000 
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model-path $MODEL_PATH \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 35000  \
  --allow-auto-truncate \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --disable-shared-experts-fusion \
  --max-running-requests 10 \
  --enable-metrics \
  --enable-mixed-chunk \
    --disable-custom-all-reduce \
  > "$LOG_FILE" 2>&1 &


SERVER_PID=$!
echo "SGLang server PID: $SERVER_PID"

echo "Waiting for SGLang server on port $PORT..."
sleep 10
while ! nc -z localhost $PORT; do
    sleep 1
done
echo "SGLang server is ready, starting pipeline..."

export OPENAI_API_KEY="nope"
export OPENROUTER_API_KEY="nope"

export OPENAI_API_METRICS="http://localhost:$PORT/metrics"

export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENROUTER_API_BASE="http://localhost:$PORT/v1"
export HF_ENDPOINT=https://hf-mirror.com
# python /home/shm/document/MEMORY_ACCLERATION/run_pipeline.py
# python pipeline.py --models MemoryOS Memo0 --datasets locomo10 longmemeval_s
# python3 /home/shm/document/MEMORY_ACCLERATION/run_pipeline.py --models memo0 --datasets locomo10
# python pipeline.py --config config.yaml
# python3 /home/shm/document/MEMORY_ACCLERATION/run_pipeline.py --models simplerag --datasets longmemeval_oracle
# python3 /home/shm/document/MEMORY_ACCLERATION/run_pipeline.py --models HippoRAG --datasets longmemeval_oracle
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/MemoryOS_longmemeval_oracle_results.jsonl --dataset_type longmemeval
python3 /home/shm/document/MEMORY_ACCLERATION/run_pipeline.py \
  --models $MODEL_NAME \
  --datasets $DATASETS 

METRICS_DIR="/home/shm/document/temp_log"
mkdir -p "$METRICS_DIR"


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
METRICS_FILE="$METRICS_DIR/metrics_sglang_${MODEL_NAME}_${DATASETS}_$PORT_$TIMESTAMP.txt"

echo "Dumping SGLang metrics before shutdown..."
curl -s "http://localhost:$PORT/metrics" -o "$METRICS_FILE"

if [ $? -eq 0 ]; then
    echo "Metrics saved to $METRICS_FILE"
else
    echo "Failed to fetch metrics, server may not expose /metrics"
fi

echo "Stopping SGLang server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "SGLang server stopped."
# curl -s "http://localhost:30086/metrics" -o "/home/shm/document/temp_log/metrics_sglang_langmem_longmemeval_m_20251117_073247.txt"