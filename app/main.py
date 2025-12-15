from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="E-commerce Recommender API", version="0.1.0")


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    model_version: Optional[str] = "baseline"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest):
    # TODO: replace stub with real model inference
    dummy_recs = list(range(1000, 1000 + payload.k))
    return RecommendResponse(user_id=payload.user_id, recommendations=dummy_recs)
