CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30099 \
  --model-path /mnt/data/models/Llama-3.2-3B-Instruct  \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 64000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
  --enable-mixed-chunk \
  --is-embedding