"""Score unknown ArcFace feature vectors against the gallery.

Accepts a .npy (shape [N, 512]) or .pkl (list of vectors / dict with 'embeddings')
and prints the top-k match for each one.

Usage:
  python scripts/identify_unknowns.py unknowns.npy
  python scripts/identify_unknowns.py unknowns.pkl --top-k 5
  python scripts/identify_unknowns.py --image path/to/some.jpg    # extract + identify
"""
from pathlib import Path
import argparse
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GALLERY_PKL
from src.identify import load_gallery, identify


def _load_vectors(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ("embeddings", "features", "vectors"):
            if k in obj:
                return np.asarray(obj[k])
        raise ValueError(f"pkl dict has no recognized key; got {list(obj)}")
    return np.asarray(obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors", nargs="?", type=Path,
                    help=".npy or .pkl of ArcFace vectors (shape [N, 512])")
    ap.add_argument("--image", type=Path,
                    help="Alternative: run ArcFace on this image and identify it")
    ap.add_argument("--gallery", type=Path, default=GALLERY_PKL)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    emb, lbl = load_gallery(args.gallery)
    print(f"Loaded gallery: {len(lbl)} vectors, {len(set(lbl))} identities")

    if args.image:
        import cv2
        from src.feature_extraction import init_arcface
        app = init_arcface(use_gpu=args.gpu)
        img = cv2.imread(str(args.image))
        if img is None:
            sys.exit(f"Could not read {args.image}")
        faces = app.get(img)
        if not faces:
            sys.exit("No face detected in the image.")
        vectors = np.asarray([faces[0].embedding])
    elif args.vectors:
        vectors = _load_vectors(args.vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
    else:
        sys.exit("Provide either a vectors file or --image <path>.")

    print(f"Scoring {len(vectors)} unknown vector(s) against gallery ...\n")
    for i, v in enumerate(vectors):
        print(f"--- Unknown #{i + 1} ---")
        identify(v, emb, lbl, top_k=args.top_k, verbose=True)
        print()


if __name__ == "__main__":
    main()
