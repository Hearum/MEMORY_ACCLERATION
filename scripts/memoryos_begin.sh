
#!/bin/bash
cd /home/shm/document/MEMORY_ACCLERATION
MODEL_PATH="/mnt/data/models/zhipu-glm-4-9b-chat-1m"
START_PORT=30000
PORT=$START_PORT
while ss -tuln | grep -q ":$PORT "; do
    PORT=$((PORT + 1))
done

CUDA_DEVICES=2

MODEL_NAME="MemoryOS"
DATASETS="locomo10"

# locomo10 longmemeval_s

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

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate sglang
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model-path $MODEL_PATH \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 30000  \
  --allow-auto-truncate \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
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

# LLM Sever
export OPENAI_API_KEY="nope"
export OPENROUTER_API_KEY="nope"
export OPENAI_API_METRICS="http://localhost:$PORT/metrics"
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENROUTER_API_BASE="http://localhost:$PORT/v1"

# EMBEDDING Sever
export EMBED_API_BASE="http://localhost:8000/v1"
export EMBED_MODEL_NAME="/mnt/data3/models/Qwen3-Embedding-4B"

export HF_ENDPOINT=https://hf-mirror.com

# lme
# python3 /home/shm/document/MEMORY_ACCLERATION/model/Nemori/nemori/evaluation/longmemeval/add.py
# python3 /home/shm/document/MEMORY_ACCLERATION/model/Nemori/nemori/evaluation/locomo/search.py

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate sglang
python3 /home/shm/document/MEMORY_ACCLERATION/run_pipeline.py \
  --models $MODEL_NAME \
  --datasets $DATASETS 

# pip install requests 

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
METRICS_FILE="$RESULTS_DIR/metrics_sglang_${MODEL_NAME}_${DATASETS}_$PORT_$TIMESTAMP.txt"

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