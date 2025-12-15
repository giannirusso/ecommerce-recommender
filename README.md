# ecommerce-recommender

Top-N product recommendations using collaborative filtering (TensorFlow/Keras) + FastAPI + Docker.

## Overview
This repository demonstrates an end-to-end recommender system workflow:
- training a collaborative filtering model on user–item interactions
- evaluating Top-N ranking metrics (Precision@K, Recall@K)
- serving recommendations through a production-style REST API

## API
### Endpoints
- `GET /health`
- `POST /recommend`

Example:
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 10}'
```

### Run (Docker)
```bash
docker build -t recommender-api .
docker run -p 8000:8000 recommender-api
```
