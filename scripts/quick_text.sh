#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY="nope"
export OPENAI_API_BASE="http://localhost:30144/v1"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sglang

EVAL_SCRIPT="/home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py"

INPUT_FILES=(
  "/home/shm/document/MEMORY_ACCLERATION/model/Nemori/nemori/evaluation/locomo/results_cleaned.json"

  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_HippoRAG_locomo10_mem/HippoRAG_locomo10_generation_results1.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_HippoRAG_longmemeval_oracle_mem/HippoRAG_longmemeval_oracle_generation_results1.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_HippoRAG_longmemeval_s_mem/HippoRAG_longmemeval_s_generation_results.jsonl"

  "/home/shm/document/MEMORY_ACCLERATION/model/Nemori/nemori/evaluation/locomo/results_cleaned.json"
  "/home/shm/document/MEMORY_ACCLERATION/model/Nemori/nemori/evaluation/longmemeval_results.json"

  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_EverMemOS_longmemeval_s_50_mem/answer_results.eval-results-locomo.json"
  "/home/shm/document/EverMemOS-main/evaluation/results/locomo-evermemos/answer_results.json"

  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_AMEM_locomo10_mem/realtime_results.jsonl"

  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_QA_longmemeval_s_mem/QA_longmemeval_s_generation_results.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_QA_locomo10_mem_bf/QA_locomo10_generation_results.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_QA_longmemeval_oracle_mem_bf/QA_longmemeval_oracle_generation_results.jsonl"

  "/home/shm/document/MEMORY_ACCLERATION/results/kaiwen/MemGAS_locomo10_generation_results.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/kaiwen/MemGAS_longmemeval_oracle_generation_results.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/kaiwen/MemGAS_longmemeval_s_generation_results.jsonl"

  "/home/shm/document/MEMORY_ACCLERATION/results/kaiwen/simplerag_locomo10_generation_results.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/kaiwen/simplerag_longmemeval_oracle_generation_results.jsonl"
  "/home/shm/document/MEMORY_ACCLERATION/results/kaiwen/simplerag_longmemeval_s_generation_results.jsonl"
)

for INPUT in "${INPUT_FILES[@]}"; do
  echo "=================================================="
  echo "Evaluating: $INPUT"
  echo "=================================================="

  python "$EVAL_SCRIPT" --input_file "$INPUT"

  echo
done

echo "All evaluations finished."
