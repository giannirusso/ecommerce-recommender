# ecommerce-recommender

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-success)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Model](https://img.shields.io/badge/Model-SVD%20(Matrix%20Factorization)-brightgreen)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

Top-N product recommendations using collaborative filtering (Matrix Factorization) served via FastAPI and Docker.

---

## Overview

This repository demonstrates an end-to-end recommender system workflow:
- training a collaborative filtering model on user–item interaction data
- evaluating model quality using offline metrics
- serving Top-N recommendations through a production-style REST API
- packaging the service with Docker for reproducible deployment

The project is designed to reflect a real-world applied ML scenario rather than a notebook-only experiment.

---

## Dataset & Model

**Dataset:** MovieLens 100K  
- 100,000 explicit ratings  
- 943 users, 1,682 items  
- Ratings scale: 1–5  
- Standard benchmark dataset for recommender systems

**Model:** Matrix Factorization (SVD)  
- Implemented using `scikit-surprise`
- Latent factor collaborative filtering
- Trained on explicit feedback
- Optimized for Top-N recommendation generation

**Model Artifact:**  
- Trained model is serialized and stored as `artifacts/svd_model.pkl`

---
## Architecture Overview

The system follows a standard offline training + online inference architecture:

1. Offline training pipeline trains a Matrix Factorization (SVD) model on historical user–item interactions.
2. The trained model is serialized and stored as an artifact.
3. At runtime, the FastAPI service loads the model and generates Top-N recommendations on demand.
4. The inference service is containerized with Docker for reproducible deployment.

---
## Project Structure

```text
ecommerce-recommender/
├── app/                    # FastAPI application
├── src/
│   ├── data/               # Dataset download & preprocessing
│   └── models/             # Training and recommendation logic
├── artifacts/              # Trained model artifacts
├── Dockerfile              # Containerized API service
├── requirements.txt        # Python dependencies
└── README.md
```
---

## Training the Model
**1. Download the dataset**

The MovieLens 100K dataset is downloaded automatically from the official GroupLens repository.
```bash
python src/data/download_movielens.py
```

**2. Train the recommender model**

Train the Matrix Factorization (SVD) model and persist it as a reusable artifact.
```bash
python src/models/train.py
```

Example output:
```bash
RMSE: 0.93
Saved model to: artifacts/svd_model.pkl
```
---

## Running the API
**Run locally**

**1. After training the model, start the inference service:**
```bash
uvicorn app.main:app --reload
```

**2. Open the interactive API documentation:**
```bash
http://localhost:8000/docs
```

**Run with Docker**
```bash
docker build -t recommender-api .
docker run -p 8000:8000 recommender-api
```
Then open:
```bash
http://localhost:8000/docs
```
