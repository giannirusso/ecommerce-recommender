from pathlib import Path
import pickle
import pandas as pd

MODEL_PATH = Path("artifacts/svd_model.pkl")
RATINGS_PATH = Path("data/ml-100k/u.data")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_ratings():
    df = pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return df

def recommend_top_n(user_id: int, k: int = 10):
    model = load_model()
    ratings = load_ratings()

    all_items = ratings["item_id"].unique()
    rated_items = set(ratings.loc[ratings["user_id"] == user_id, "item_id"].tolist())

    candidates = [iid for iid in all_items if iid not in rated_items]
    scored = [(iid, model.predict(user_id, int(iid)).est) for iid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [int(iid) for iid, _ in scored[:k]]
