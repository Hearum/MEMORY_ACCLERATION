

# Readme

This repository provides a unified evaluation pipeline for memory-based systems.

### Project Structure

```
memory_eval_pipeline/
├── README.md
├── requirements.txt
├── configs/                 
│   ├── datasets.yaml        
│   ├── models.yaml         
│   └── evaluation.yaml      
├── datasets/                
│   ├── longmemeval/
│   └── locomo10/
├── models/                  
│   ├── MemoryOS
│   ├── Memo
│   └── Rag_and_others
├── evaluators/            
│   ├── base_evaluator.py    
│   ├── longmemeval_eval.py
│   └── locomo_eval.py
├── utils/                   
├── run_pipeline.py                 
└── results/                 
    ├── memoryos_longmemeval.jsonl
    └── rag_locomo10.jsonl

```

## dataset

### LOGOMOMO
Download from github [https://github.com/BAI-LAB/MemoryOS/blob/main/eval/locomo10.json]

### longmemeval

Download from Google Drive [longmemeval_data.tar.gz - Google](https://drive.google.com/file/d/1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80/view)

- `longmemeval_s.json`: The LongMemEval_S introduced in the paper. Concatenating all the chat history roughly consumes 115k tokens (~40 history sessions) for Llama 3.
- `longmemeval_m.json`: The LongMemEval_M introduced in the paper. Each chat history contains roughly 500 sessions.
- `longmemeval_oracle.json`: LongMemEval with oracle retrieval. Only the evidence sessions are included in the history.

## Model Serving (SGLang)

Launch local LLM server:

```
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path /mnt/data/models/Llama-3.2-3B-Instruct   \
  --served-model-name LLAMA \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --max-total-tokens 64000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --disable-shared-experts-fusion \
  --max-running-requests 50 \
  --enable-mixed-chunk
```

log KV cache success rate

```
  --enable-metrics
```
## baseline
to be continue.

## Generation

All models use the **OpenAI-compatible API**.

```
# Set global environment variables:
export OPENAI_API_KEY="nope"
export OPENAI_API_BASE="http://localhost:30000/v1"
export HF_ENDPOINT=https://hf-mirror.com 

```

We provide a helper script begin.sh to automatically:

- Launch the SGLang local LLM server.

- Set the environment variables for OpenAI-compatible API.

- Run the generation pipeline on the specified models and datasets.

```
./begin.sh
```
Configuration inside begin.sh:

- MODEL_PATH: path to the local model weights.

- PORT: port for the LLM server.

- CUDA_DEVICES: which GPU(s) to use.

- LOG_DIR: directory to save server logs.

- DATASETS: dataset(s) to evaluate, e.g., locomo10, longmemeval_m.




# evaluate
## Cache hit rate
caculate the cache hit rate from your generation sglang log:
```
./scripts/cal_kv_cache.py
```

## Metric
Our evaluation pipeline supports multiple metrics to comprehensively assess memory-based systems, including F1, BLEU, coverage, accuracy, and LLM-based judgment.
```
python ./evaluators/base_evaluator.py --input_file ./results/results.jsonl --dataset_type locomo 
python ./evaluators/base_evaluator.py --input_file ./results/results.jsonl --dataset_type longmemeval
```


## Perforom
### LOCOMO10_Llama-3.2-3B-Instruct_top_k_10

| f1_score/bleu_score/llm_score | Single Hop      | Multi-Hop      | Open Domain    | Temporal       |
| ----------------------------- | --------------- | -------------- | -------------- | -------------- |
| MemoryOS                      | 25.3/27.3/60.0  | 26.9/34.5/45.6 | 11.8/17.0/62.0 | 29.8/45.4/63.0 |
| Memo0                         | 12.3/15.8/32.27 | 11.7/13.5/33.4 | 8.0/7.2/45.6   | 13.3/12.5/32.9 |
| RAG                           | 6.4/13.4/36.5   | 2.9/8.9/34.2   | 13.2/17.8/54.2 | 8.8/13.4/38.2  |

### longmemeval_m_Llama-3.2-3B-Instruct_top_k_10

|                               | f1_score | bleu_score | llm_score | accuracy |
| ----------------------------- | -------- | ---------- | --------- | -------- |
| MemoryOS (longmemeval_m)      | 51.6     | 47.5       | 51.6      | 0.516    |
| MemoryOS (longmemeval_s)      | 30.9     | 28.5       | 30.9      | 0.41     |
| MemoryOS (longmemeval_oracle) | 32.5     | 29.5       | 32.5      | 0.442    |
| RAG (longmemeval_m)           | 15.8     | 17.9       | 21.2      | 0.25     |
| RAG (longmemeval_s)           | 6.4      | 4.8        | 6.06      | 0.07     |
| RAG (longmemeval_oracle)      | 13.0     | 9.86       | 13.2      | 0.19     |
|                               |          |            |           |          |



### KVcache hit rate

| KVcache hit rate/ Total token | longmemeval_m | longmemeval_s | longmemeval_oracle | LOCOMO10 |
| ----------------------------- | ------------- | ------------- | ------------------ | -------- |
| MemoryOS                      | 44.26%        | 45.94%        | 46.32%             | 55.69%   |
| Memo0                         | -             | -             | -                  | 75.15%   |
| RAG                           |               |   14.38%     | 16.31%             | 21.26%   |


