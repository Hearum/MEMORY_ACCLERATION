#!/bin/bash

CUDA_DEVICES=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
python -m vllm.entrypoints.openai.api_server \
    --model "/mnt/data3/models/Qwen3-Embedding-4B" \
    --host 0.0.0.0 \
    --port 8000 &   # 放到后台运行

OUTPUT_FILE="/home/shm/document/EverMemOS-main/evaluation/script/log/vllm_embedding_metrics.log"
INTERVAL=15
while true
do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "===== $TIMESTAMP =====" > "$OUTPUT_FILE"
    curl -s http://0.0.0.0:8000/metrics >> "$OUTPUT_FILE"
    echo -e "\n" >> "$OUTPUT_FILE"
    sleep $INTERVAL
done

# uv venv myenv --python 3.12 --seed
# source ~/myenv/bin/activate
# uv pip install vllm