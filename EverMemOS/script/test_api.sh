#!/bin/bash
curl -v -X POST http://127.0.0.1:12001/v1/rerank \
    -H "Content-Type: application/json" \
    -d '{
    "query": "Memory systems help LLMs store context.",
    "documents": [
        "Memory systems help LLMs store context.",
        "FastAPI is used to build APIs."
    ]
    }'
