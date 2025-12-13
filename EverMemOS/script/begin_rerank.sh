CUDA_DEVICES=3
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    python -m vllm.entrypoints.openai.api_server \
        --model "/home/shm/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/f16fc5d5d2b9b1d0db8280929242745d79794ef5" \
        --host 0.0.0.0 \
        --port 12000  &   # 放到后台运行

OUTPUT_FILE="/home/shm/document/EverMemOS-main/evaluation/script/log/vllm_rerank_metrics.log"
INTERVAL=15
while true
do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "===== $TIMESTAMP =====" > "$OUTPUT_FILE"
    curl -s http://0.0.0.0:12000/metrics >> "$OUTPUT_FILE"
    echo -e "\n" >> "$OUTPUT_FILE"
    sleep $INTERVAL
done


# source myenv/bin/activate