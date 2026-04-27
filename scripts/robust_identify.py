"""Robust inference script for identifying unknown feature vectors.

Instead of matching unknowns against a "clean" gallery (which fails when unknowns
have injected noise or extreme lighting/occlusion/side-profiles), this script
trains a supervised Logistic Regression model on the FULL combined train+test
dataset. The classifier learns to handle the varying conditions and noise implicitly.
"""
from pathlib import Path
import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TRAIN_PKL, TEST_PKL, RANDOM_SEED


def _load_vectors(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            obj = arr.item() if arr.shape == () else arr.tolist()
            if isinstance(obj, dict):
                for k in ("embeddings", "features", "vectors"):
                    if k in obj:
                        return np.asarray(obj[k], dtype=np.float32)
                raise ValueError(f".npy dict has no recognized key; got {list(obj)}")
            return np.asarray(obj, dtype=np.float32)
        return arr.astype(np.float32)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ("embeddings", "features", "vectors"):
            if k in obj:
                return np.asarray(obj[k], dtype=np.float32)
        raise ValueError(f"pkl dict has no recognized key; got {list(obj)}")
    return np.asarray(obj, dtype=np.float32)


def get_full_dataset() -> pd.DataFrame:
    try:
        train_df = pd.read_pickle(TRAIN_PKL)
        test_df = pd.read_pickle(TEST_PKL)
        return pd.concat([train_df, test_df], ignore_index=True)
    except FileNotFoundError:
        sys.exit(f"Could not find train/test pickles. Ensure build_gallery.py has been run.")


def identify_vectors(vectors: np.ndarray, model_type: str, full_df: pd.DataFrame, top_k: int = 5):
    # Filter dataset for the appropriate feature model
    df = full_df[full_df["model"] == model_type]
    if len(df) == 0:
        sys.exit(f"No {model_type} embeddings found in the dataset.")
        
    X_train = np.vstack(df["embedding"].values)
    y_train = df["base_identity"].values
    
    print(f"Training robust {model_type} classifier on FULL dataset ({len(X_train)} samples)...")
    
    if model_type == "ArcFace":
        # Raw Logistic Regression works perfectly for ArcFace
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        
        # Also prepare for fallback cosine similarity
        X_train_norm = normalize(X_train, norm="l2")
        vecs_norm = normalize(vectors, norm="l2")
        
        for i, v in enumerate(vectors):
            print(f"\n--- Unknown #{i + 1} ({model_type}) ---")
            
            # Classifier Prediction
            probs = clf.predict_proba(v.reshape(1, -1))[0]
            top_idx = probs.argsort()[::-1][:top_k]
            
            print(f"Logistic Regression Confidence (Trained on ALL conditions):")
            for rank, idx in enumerate(top_idx, 1):
                flag = " <- predicted" if rank == 1 else ""
                print(f"  {rank}. {clf.classes_[idx]:<20} prob={probs[idx]:.4f}{flag}")
                
            # Nearest Neighbor fallback against full dataset
            sims = cosine_similarity(vecs_norm[i].reshape(1, -1), X_train_norm)[0]
            nn_top_idx = sims.argsort()[::-1][:top_k]
            
            print(f"\nNearest Neighbor Fallback (Cosine Sim vs FULL dataset):")
            seen = set()
            rank = 1
            for idx in nn_top_idx:
                label = y_train[idx]
                if label not in seen:
                    print(f"  {rank}. {label:<20} max_sim={sims[idx]:.4f}")
                    seen.add(label)
                    rank += 1
                if rank > top_k:
                    break

    elif model_type == "VGG19":
        # PCA + Logistic Regression works best for VGG19 4096-d
        X_train_n = normalize(X_train, norm="l2")
        pca = PCA(n_components=256, random_state=RANDOM_SEED)
        X_train_p = pca.fit_transform(X_train_n)
        
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(X_train_p, y_train)
        
        vecs_norm = normalize(vectors, norm="l2")
        vecs_p = pca.transform(vecs_norm)
        
        for i, v in enumerate(vectors):
            print(f"\n--- Unknown #{i + 1} ({model_type}) ---")
            
            # Classifier Prediction
            probs = clf.predict_proba(vecs_p[i].reshape(1, -1))[0]
            top_idx = probs.argsort()[::-1][:top_k]
            
            print(f"Logistic Regression Confidence (PCA 256, Trained on ALL conditions):")
            for rank, idx in enumerate(top_idx, 1):
                flag = " <- predicted" if rank == 1 else ""
                print(f"  {rank}. {clf.classes_[idx]:<20} prob={probs[idx]:.4f}{flag}")
                
            # Nearest Neighbor fallback against full dataset
            sims = cosine_similarity(vecs_norm[i].reshape(1, -1), X_train_n)[0]
            nn_top_idx = sims.argsort()[::-1][:top_k]
            
            print(f"\nNearest Neighbor Fallback (Cosine Sim vs FULL dataset):")
            seen = set()
            rank = 1
            for idx in nn_top_idx:
                label = y_train[idx]
                if label not in seen:
                    print(f"  {rank}. {label:<20} max_sim={sims[idx]:.4f}")
                    seen.add(label)
                    rank += 1
                if rank > top_k:
                    break


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust identification of unknown feature vectors")
    ap.add_argument("vectors", nargs="+", type=Path,
                    help="One or more .npy or .pkl files of unknown vectors")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    full_df = get_full_dataset()
    print(f"Loaded full dataset (train+test): {len(full_df)} images total")

    for path in args.vectors:
        print(f"\n{'='*60}")
        print(f"Processing {path.name}")
        print(f"{'='*60}")
        
        try:
            vectors = _load_vectors(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
            
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        dim = vectors.shape[1]
        print(f"Detected vector dimension: {dim}")
        
        if dim == 512:
            identify_vectors(vectors, "ArcFace", full_df, args.top_k)
        elif dim == 4096:
            identify_vectors(vectors, "VGG19", full_df, args.top_k)
        elif dim == 1000:
            print("\n[WARNING] Detected 1000-d vectors. These appear to be VGG19 logits.")
            print("The dataset 'stage2_embeddings.pkl' only contains 4096-d VGG19 fc2 features.")
            print("Please use `scripts/match_vgg_logits.py` for 1000-d vectors or re-extract 4096-d features.")
            print("Skipping...")
        else:
            print(f"\n[WARNING] Unknown vector dimensionality: {dim}.")
            print("Expected 512 for ArcFace, 4096 for VGG19 fc2, or 1000 for VGG19 logits.")
            print("Attempting Nearest Neighbor against ArcFace just in case...")
            try:
                identify_vectors(vectors, "ArcFace", full_df, args.top_k)
            except Exception as e:
                print(f"Failed fallback: {e}")

if __name__ == "__main__":
    main()
