

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

Already included in repo under `datasets/locomo10`.

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

```
python scripts/run_pipeline.py
```

Example config inside `run_pipeline.py`:

```
models_to_run = ["MemoryOS", "MeMo0", "RAG"] ...
datasets_to_run = [
{"name": "locomo10", "path": os.path.join(os.path.join(DATA_DIR, "locomo10"),"locomo10.json")},
# {"name": "longmemeval_s", "path":  os.path.join(DATA_DIR, "longmemeval_s.json")},
# {"name": "longmemeval_m", "path":  os.path.join(DATA_DIR, "longmemeval_m.json")},
# {"name": "longmemeval_oracle", "path":  os.path.join(DATA_DIR, "longmemeval_oracle.json")},
]
```



# evaluate

**locomomo**

```
python evals.py --input_file  --output_file 
```

**longmemeval**

```
python3 evaluate_qa.py gpt-4o your_hypothesis_file ../../data/longmemeval_oracle.json
```

