"""Robust matcher for 1000-d VGG19 pre-softmax logits against the FULL dataset.

Unlike match_vgg_logits.py which only extracts from the clean Gallery, this script
extracts 1000-d logits from BOTH Gallery and Probe folders (including extreme conditions
like Lighting, Occlusion, Side profiles) and trains a Logistic Regression classifier.
This makes it significantly more robust to noise and varying conditions.
"""
from pathlib import Path
import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_ROOT, RANDOM_SEED
from src.feature_extraction import init_arcface, walk_image_tasks
from src.metadata import parse_base_identity


def _load_unknown(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        obj = arr.item() if arr.shape == () else arr
        if isinstance(obj, dict):
            for k in ("features", "embeddings", "vectors"):
                if k in obj:
                    return np.asarray(obj[k], dtype=np.float32).reshape(1, -1)
            raise ValueError(f"no recognized key in {path}; got {list(obj)}")
    return np.asarray(arr, dtype=np.float32).reshape(1, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors", nargs="+", type=Path,
                    help="One or more .npy files of 1000-d VGG19 logits")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--classmates-only", action="store_true",
                    help="Drop the 10 celebrity identities from the training set")
    ap.add_argument("--center", action="store_true", default=True,
                    help="Subtract mean logit before L2 norm (Default: True)")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    from tensorflow.keras.models import Model

    base = VGG19(weights="imagenet", include_top=True)
    fc2_model = Model(inputs=base.input, outputs=base.get_layer("fc2").output)
    W, b = base.get_layer("predictions").get_weights()

    app = init_arcface(use_gpu=args.gpu)
    
    # NEW: Extract from ALL images (Gallery + Probe) for maximum robustness
    tasks = list(walk_image_tasks(DATA_ROOT))
    print(f"Extracting 1000-d features from ALL {len(tasks)} images (Gallery + Probe)...")

    feats, labels = [], []
    for split, root, name in tasks:
        img = cv2.imread(os.path.join(root, name))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        batch = preprocess_input(np.expand_dims(resized, axis=0))
        fc2 = fc2_model.predict(batch, verbose=0)
        feats.append((fc2 @ W + b)[0].astype(np.float32))
        labels.append(parse_base_identity(os.path.basename(root), name))

    if not feats:
        sys.exit("No features extracted. Ensure DATA_ROOT is correct and contains images.")

    X_raw = np.vstack(feats)
    mean_vec = X_raw.mean(axis=0, keepdims=True) if args.center else 0.0
    X = normalize(X_raw - mean_vec, norm="l2")
    labels = np.array(labels)

    if args.classmates_only:
        keep = np.array([n.isdigit() for n in labels])
        X, labels = X[keep], labels[keep]

    print(f"Training robust Logistic Regression on {len(X)} images ({len(set(labels))} identities)...")
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    clf.fit(X, labels)

    def prep_unknown(path):
        return normalize(_load_unknown(path) - mean_vec, norm="l2")

    for path in args.vectors:
        u = prep_unknown(path)
        probs = clf.predict_proba(u)[0]
        order = probs.argsort()[::-1][:args.top_k]
        
        print(f"\n=== {path} ===")
        print("Logistic Regression Confidence:")
        for rank, j in enumerate(order, 1):
            tag = " <- predicted" if rank == 1 else ""
            print(f"  {rank}. {clf.classes_[j]:<20} prob={probs[j]:.4f}{tag}")
            
        # Also print nearest neighbor (cosine similarity) as fallback
        sims = (u @ X.T)[0]
        nn_order = sims.argsort()[::-1][:args.top_k]
        print("\nNearest Neighbor Fallback (Cosine Sim vs FULL dataset):")
        seen = set()
        rank = 1
        for idx in nn_order:
            label = labels[idx]
            if label not in seen:
                print(f"  {rank}. {label:<20} max_sim={sims[idx]:.4f}")
                seen.add(label)
                rank += 1
            if rank > args.top_k:
                break


if __name__ == "__main__":
    main()
