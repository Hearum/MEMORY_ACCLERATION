MODEL_PATH="/mnt/data/models/zhipu-glm-4-9b-chat-1m"
# "/mnt/data/models/Llama-3.2-3B-Instruct"
# /mnt/data/models/glm-4-9b/
PORT=30052
export CUDA_VISIBLE_DEVICES=7

echo "Starting SGLang server..."

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model-path $MODEL_PATH \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 20000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
  --enable-mixed-chunk \
  --enable-metrics \

export OPENAI_API_KEY="nope"
export OPENAI_API_BASE="http://localhost:30052/v1"
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/EverMemOS-main/evaluation/results/longmemeval-evermemos_bf/answer_results.json  --dataset_type longmemeval
# /home/shm/document/MEMORY_ACCLERATION/evaluators/new_evaluator.py