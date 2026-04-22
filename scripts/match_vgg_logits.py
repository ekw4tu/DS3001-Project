"""Match 1000-d VGG19 pre-softmax logits against a classmate gallery.

Used for the in-class challenge when the TA hands out VGG19 feature vectors
that are the 1000-d predictions-layer output (pre-softmax logits), not the
4096-d fc2 features our pipeline normally uses.

The script walks the on-disk Gallery folders, extracts 1000-d logits from
every image using VGG19 + ImageNet weights (same preprocessing as the TA:
InsightFace bbox crop -> 224x224 -> preprocess_input), builds per-identity
centroids on the unit sphere, and cosine-matches each unknown against them.

--classmates-only drops the 10 original celebrity identities so the sparse
classmate centroids aren't drowned out by the dense celebrity ones.
"""
from pathlib import Path
import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_ROOT
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
                    help="Drop the 10 celebrity identities from the centroid set")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    from tensorflow.keras.models import Model

    base = VGG19(weights="imagenet", include_top=True)
    fc2_model = Model(inputs=base.input, outputs=base.get_layer("fc2").output)
    W, b = base.get_layer("predictions").get_weights()

    app = init_arcface(use_gpu=args.gpu)
    tasks = [t for t in walk_image_tasks(DATA_ROOT) if t[0] == "Gallery"]
    print(f"gallery images to extract: {len(tasks)}")

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

    X = normalize(np.vstack(feats), norm="l2")
    labels = np.array(labels)

    if args.classmates_only:
        identities = sorted([n for n in set(labels) if n.isdigit()], key=int)
    else:
        identities = sorted(set(labels) - {"Unknown"})
    centroids = np.stack([
        normalize(X[labels == n].mean(axis=0, keepdims=True), norm="l2")[0]
        for n in identities
    ])
    print(f"gallery: {len(identities)} identities, {len(X)} vectors")

    for path in args.vectors:
        u = normalize(_load_unknown(path), norm="l2")
        sims = (u @ centroids.T)[0]
        order = sims.argsort()[::-1][:args.top_k]
        print(f"\n=== {path} ===")
        for rank, j in enumerate(order, 1):
            tag = " <- predicted" if rank == 1 else ""
            print(f"  {rank}. {identities[j]:<20} similarity={sims[j]:.4f}{tag}")


if __name__ == "__main__":
    main()
