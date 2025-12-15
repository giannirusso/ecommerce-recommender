from pathlib import Path
import pickle

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

DATA_FILE = Path("data/ml-100k/u.data")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    reader = Reader(line_format="user item rating timestamp", sep="\t")
    data = Dataset.load_from_file(str(DATA_FILE), reader=reader)
    return data

def train():
    data = load_data()
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(
        n_factors=80,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )
    model.fit(trainset)

    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)

    model_path = ARTIFACTS_DIR / "svd_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to: {model_path}")
    print(f"RMSE: {rmse:.4f}")

    return rmse

if __name__ == "__main__":
    train()
