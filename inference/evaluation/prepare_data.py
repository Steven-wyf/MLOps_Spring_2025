import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Reuse the llara_train.py data-loading functions
from train.llara_train import (
    load_embeddings_from_mlflow,
    load_playlist_data,
    create_training_pairs
)

def main():
    # 1) Fetch embeddings (MLP+LightGCN+BERT) and playlist data from MLflow
    emb_data = load_embeddings_from_mlflow()
    df       = load_playlist_data()

    # 2) Build context–target pairs for playlist continuation
    samples = create_training_pairs(df, emb_data['track_to_idx'])
    X = emb_data['embeddings'][
        [emb_data['track_to_idx'][ctx] for ctx, _ in samples]
    ]
    y = np.array([tgt for _, tgt in samples])

    # 3) Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Save numpy arrays under outputs/llara/
    out_dir = Path(__file__).parents[2] / "outputs" / "llara"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_test.npy",  y_test)

    print("✅ Test data saved to:", out_dir)

if __name__ == "__main__":
    main()

