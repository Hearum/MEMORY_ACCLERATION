from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import FlagReranker
# uvicorn rerank_sever:app --host 0.0.0.0 --port 12000
app = FastAPI()
model = FlagReranker("Qwen/Qwen3-Reranker-4B")  # 或 Qwen3-Reranker-4B

class RerankRequest(BaseModel):
    query: str
    documents: list[str]

@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    scores = model.compute_score([(req.query, d) for d in req.documents]).tolist()
    # scores = model.compute_score([("query", "doc")])
    return {"scores": scores}
