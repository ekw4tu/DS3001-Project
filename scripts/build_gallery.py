"""Walk data/ and build the full embeddings + gallery artifacts.

Run ONCE after the first clone, or any time you want to rebuild from scratch.
Outputs under artifacts/:
  - stage2_embeddings.pkl    (all ArcFace + VGG19 rows, with metadata)
  - train_embeddings.pkl     (80% split, stratified by identity+condition)
  - test_embeddings.pkl      (20% held-out split)
  - gallery_embeddings.pkl   (ArcFace gallery vectors + labels)
"""
from pathlib import Path
import argparse
import pickle
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    DATA_ROOT, EMBEDDINGS_PKL, TRAIN_PKL, TEST_PKL, RANDOM_SEED,
)
from src.feature_extraction import (
    init_arcface, init_vgg19, walk_image_tasks, extract_arcface, extract_vgg19,
)
from src.identify import build_gallery, save_gallery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=Path, default=DATA_ROOT,
                    help="Root containing Gallery/ and Probe/ subfolders")
    ap.add_argument("--gpu", action="store_true", help="Use CUDA for InsightFace if available")
    ap.add_argument("--skip-vgg", action="store_true",
                    help="Skip VGG19 extraction (useful if you only need the gallery)")
    args = ap.parse_args()

    tasks = walk_image_tasks(args.data_path)
    print(f"Found {len(tasks)} images under {args.data_path}")
    if not tasks:
        sys.exit("No images found. Populate data/Gallery/ and data/Probe/ first.")

    print("\n[1/3] Loading ArcFace (buffalo_l)")
    app = init_arcface(use_gpu=args.gpu)
    arc_records = extract_arcface(tasks, app)
    print(f"  extracted {len(arc_records)} ArcFace embeddings")

    vgg_records = []
    if not args.skip_vgg:
        print("\n[2/3] Loading VGG19")
        vgg_model = init_vgg19()
        vgg_records = extract_vgg19(tasks, app, vgg_model)
        print(f"  extracted {len(vgg_records)} VGG19 embeddings")

    df = pd.DataFrame(arc_records + vgg_records)
    df.to_pickle(EMBEDDINGS_PKL)
    print(f"\nSaved {len(df)} rows -> {EMBEDDINGS_PKL}")

    df["stratify_col"] = df["identity"] + "_" + df["condition"]
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["stratify_col"],
    )
    train_df.drop(columns=["stratify_col"]).to_pickle(TRAIN_PKL)
    test_df.drop(columns=["stratify_col"]).to_pickle(TEST_PKL)
    print(f"Train/test split saved: {TRAIN_PKL.name} ({len(train_df)}) / {TEST_PKL.name} ({len(test_df)})")

    print("\n[3/3] Building ArcFace gallery")
    emb, lbl, _ = build_gallery(df.drop(columns=["stratify_col"]))
    save_gallery(emb, lbl)
    print(f"  gallery: {len(lbl)} vectors, {len(set(lbl))} unique identities")


if __name__ == "__main__":
    main()
