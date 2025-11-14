
#!/bin/bash

MODEL_PATH="/mnt/data/models/zhipu-glm-4-9b-chat-1m"
export SGLANG_SKIP_MEMORY_CHECK=1
# "/mnt/data/models/Llama-3.2-3B-Instruct"
# /mnt/data/models/glm-4-9b/
PORT=30085
CUDA_DEVICES=0,1,2,3,
LOG_DIR="/home/shm/document/MEMORY_ACCLERATION/log"
MODEL_NAME="QA"
# MemoryOS
# memo0
# HippoRAG
# QA
DATASETS="longmemeval_m"
# locomo10
# longmemeval_s
# longmemeval_m
# longmemeval_oracle

mkdir -p "$LOG_DIR"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/sglang_${MODEL_NAME}_${DATASETS}_$PORT_$NOW.log"

echo "Starting SGLang server..."
echo "Logs will be saved to $LOG_FILE"
# python -m sglang.launch_server \
#   --host 0.0.0.0 \
#   --port $PORT \
#   --model-path $MODEL_PATH \
#   --served-model-name LLAMA \
#   --attention-backend triton \
#   --chunked-prefill-size 4096 \
#   --max-total-tokens 1000000 \
#   --tensor-parallel-size 2 \
#   --trust-remote-code \
#   --mem-fraction-static 0.98 \
#   --disable-shared-experts-fusion \
#   --max-running-requests 50 \
#   --enable-mixed-chunk \
#   --enable-metrics \
#   > "$LOG_FILE" 2>&1 &
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model-path $MODEL_PATH \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 1024000 \
  --allow-auto-truncate \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --disable-shared-experts-fusion \
  --max-running-requests 10 \
  --enable-metrics \
  --enable-mixed-chunk \
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
export OPENAI_API_BASE="http://localhost:$PORT/v1"
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

echo "Stopping SGLang server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "SGLang server stopped."