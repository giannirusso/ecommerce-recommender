from pathlib import Path
from urllib.request import urlretrieve
import zipfile

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_movielens_100k(data_dir: str = "data") -> Path:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / "ml-100k.zip"
    extract_path = data_path / "ml-100k"

    if not extract_path.exists():
        if not zip_path.exists():
            print(f"Downloading: {ML_100K_URL}")
            urlretrieve(ML_100K_URL, zip_path)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)

        # the archive extracts into "ml-100k" folder directly under data/
        # keep consistent path name
        if (data_path / "ml-100k").exists():
            extract_path = data_path / "ml-100k"
    else:
        print("Dataset already exists, skipping download.")

    return extract_path

if __name__ == "__main__":
    p = download_movielens_100k()
    print(f"Dataset ready at: {p}")
