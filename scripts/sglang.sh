#!/bin/bash

MODEL_PATH="/mnt/data/models/Llama-3.2-3B-Instruct"
MODEL_NAME="LLAMA"
BASE_PORT=30000
NUM_GPUS=4

mkdir -p logs

for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  echo ">>> Starting SGLang on GPU $i (port $PORT) ..."

  CUDA_VISIBLE_DEVICES=$i PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port $PORT \
    --model-path $MODEL_PATH \
    --served-model-name ${MODEL_NAME}_gpu${i} \
    --attention-backend triton \
    --chunked-prefill-size 4096 \
    --max-total-tokens 64000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --mem-fraction-static 0.98 \
    --disable-shared-experts-fusion \
    --max-running-requests 50 \
    --enable-mixed-chunk \
    > logs/sglang_gpu${i}.log 2>&1 &

  sleep 3
done

echo "All SGLang instances started."
