MODEL_PATH="/mnt/data/models/zhipu-glm-4-9b-chat-1m"
# "/mnt/data/models/Llama-3.2-3B-Instruct"
# /mnt/data/models/glm-4-9b/
PORT=30087
CUDA_DEVICES=4

mkdir -p "$LOG_DIR"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/sglang_${MODEL_NAME}_${DATASETS}_$PORT_$NOW.log"

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
  --max-total-tokens 500000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
  --enable-mixed-chunk \
  --enable-metrics \

export OPENAI_API_KEY="nope"
export OPENAI_API_BASE="http://localhost:30087/v1"
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_MemoryOS_longmemeval_oracle.jsonl --dataset_type longmemeval