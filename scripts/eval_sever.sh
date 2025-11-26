MODEL_PATH="/mnt/data/models/zhipu-glm-4-9b-chat-1m"
# "/mnt/data/models/Llama-3.2-3B-Instruct"
# /mnt/data/models/glm-4-9b/
<<<<<<< HEAD
PORT=30082
CUDA_DEVICES=2,3

mkdir -p "$LOG_DIR"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/sglang_${MODEL_NAME}_${DATASETS}_$PORT_$NOW.log"
=======
PORT=30050
export CUDA_VISIBLE_DEVICES=2,3
>>>>>>> main

echo "Starting SGLang server..."

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model-path $MODEL_PATH \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 500000 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
  --enable-mixed-chunk \
  --enable-metrics \

export OPENAI_API_KEY="nope"
<<<<<<< HEAD
export OPENAI_API_BASE="http://localhost:30086/v1"
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_MemoryOS_longmemeval_m_results.jsonl --dataset_type longmemeval
=======
export OPENAI_API_BASE="http://localhost:30050/v1"
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_langmem_longmemeval_s_mem/langmem_longmemeval_s_generation_results.jsonl  --dataset_type longmemeval
>>>>>>> main
