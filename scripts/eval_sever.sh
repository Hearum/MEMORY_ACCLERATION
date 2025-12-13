MODEL_PATH="/mnt/data/models/zhipu-glm-4-9b-chat-1m"
# "/mnt/data/models/Qwen3-Next-80B-A3B-Instruct"
# Qwen3-30B-A3B-KT-2 zhipu-glm-4-9b-chat-1m
# "/mnt/data/models/Llama-3.2-3B-Instruct"
# /mnt/data/models/glm-4-9b/
PORT=30082
# CUDA_DEVICES=0,1,2,3,4,5,6,7
CUDA_DEVICES=0


echo "Starting SGLang server..."

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port $PORT \
  --model-path $MODEL_PATH \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 30000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
  --enable-mixed-chunk \
  --enable-metrics \

export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY="nope"
export OPENAI_API_BASE="http://localhost:30082/v1"
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_MemoryOS_locomo10_mem/MemoryOS_locomo10_generation_results.jsonl --dataset_type longmemeval
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_MemoryOS_longmemeval_oracle_mem/MemoryOS_longmemeval_oracle_generation_results.jsonl --dataset_type longmemeval

# curl -X POST "$OPENAI_API_BASE/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "model": "LLAMA", 
#     "messages": [
#       {
#         "role": "user",
#         "content": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: Do Jon and Gina start businesses out of what they love?\n\nExplanation: Yes\n\nModel Response: \nYes, Jon and Gina start businesses out of what they love.\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
#       }
#     ],
#     "n": 1,
#     "temperature": 0,
#     "max_tokens": 10
#   }'
