from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from src.models.recommend import recommend_top_n

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
    recs = recommend_top_n(user_id=payload.user_id, k=payload.k)
    return RecommendResponse(user_id=payload.user_id, recommendations=recs, model_version="svd")

