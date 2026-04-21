"""Reproduce the Stage 3 evaluation table on the held-out test set.

This is the single command a TA runs to verify the system. It reads the
artifacts produced by build_gallery.py and runs every classifier reported
in Stage 3 cell 10 (the unified comparison table).
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TRAIN_PKL, TEST_PKL, VGG_HEAD_PTH, VGG_CONTRAST_PTH
from src.identify import build_gallery, identify
from src.train import (
    train_condition_classifier, train_identity_classifier,
    train_vgg_pca_pipeline, train_vgg_mlp_head, train_vgg_contrastive_head,
    per_condition_accuracy,
)
from src.metadata import merge_clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=TRAIN_PKL)
    ap.add_argument("--test", type=Path, default=TEST_PKL)
    ap.add_argument("--skip-mlp", action="store_true",
                    help="Skip the VGG MLP head (faster; keeps LR/PCA results)")
    args = ap.parse_args()

    train_df = pd.read_pickle(args.train)
    test_df = pd.read_pickle(args.test)
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")

    results = {}

    print("\n[1] ArcFace -> Condition (LR, balanced, clean merged)")
    r = train_condition_classifier(train_df, test_df, "ArcFace")
    results["ArcFace cond LR"] = r.accuracy
    print(f"  accuracy = {r.accuracy:.3f}")

    print("\n[2] VGG19 -> Condition (LR, balanced, clean merged)")
    r = train_condition_classifier(train_df, test_df, "VGG19")
    results["VGG19 cond LR"] = r.accuracy
    print(f"  accuracy = {r.accuracy:.3f}")

    print("\n[3] ArcFace -> Identity (LR)")
    arc_id = train_identity_classifier(train_df, test_df, "ArcFace")
    results["ArcFace id LR"] = arc_id.accuracy
    print(f"  accuracy = {arc_id.accuracy:.3f}")

    print("\n[4] VGG19 -> Identity (raw LR)")
    vgg_id = train_identity_classifier(train_df, test_df, "VGG19")
    results["VGG19 id LR"] = vgg_id.accuracy
    print(f"  accuracy = {vgg_id.accuracy:.3f}")

    print("\n[5] VGG19 -> Identity (L2 + PCA + LR)")
    vgg_pca = train_vgg_pca_pipeline(train_df, test_df)
    results["VGG19 id L2+PCA+LR"] = vgg_pca.accuracy
    print(f"  accuracy = {vgg_pca.accuracy:.3f}")

    if not args.skip_mlp:
        print("\n[6] VGG19 -> Identity (MLP head)")
        vgg_mlp = train_vgg_mlp_head(train_df, test_df)
        results["VGG19 id MLP head"] = vgg_mlp.accuracy
        print(f"  best accuracy = {vgg_mlp.accuracy:.3f}")
        import torch
        torch.save(vgg_mlp.model[2], VGG_HEAD_PTH)
        print(f"  saved best head -> {VGG_HEAD_PTH}")

        print("\n[7] VGG19 -> Identity (SupCon contrastive + nearest-centroid)")
        vgg_con = train_vgg_contrastive_head(train_df, test_df)
        results["VGG19 id SupCon+NN"] = vgg_con.accuracy
        print(f"  accuracy = {vgg_con.accuracy:.3f}")
        torch.save(vgg_con.model[0].state_dict(), VGG_CONTRAST_PTH)
        print(f"  saved contrastive head -> {VGG_CONTRAST_PTH}")

    print("\n[8] identify() validation (ArcFace cosine)")
    full = pd.concat([train_df, test_df], ignore_index=True)
    emb, lbl, seeded = build_gallery(full, verbose=False)
    test_arc = test_df[(test_df["model"] == "ArcFace") &
                       (~test_df["filename"].isin(seeded))]
    if len(test_arc):
        X = np.vstack(test_arc["embedding"].values)
        y = test_arc["base_identity"].values
        correct = sum(identify(v, emb, lbl) == t for v, t in zip(X, y))
        acc = correct / len(y)
        results["identify() cosine"] = acc
        print(f"  accuracy = {acc:.3f} ({correct}/{len(y)})")

    print("\n[9] Per-condition identity accuracy breakdown")
    cond_test = np.array([merge_clean(c) for c in test_df[test_df.model == "ArcFace"].condition.values])
    pivot = pd.concat([
        per_condition_accuracy(arc_id.y_true, arc_id.y_pred, cond_test, "ArcFace"),
        per_condition_accuracy(vgg_id.y_true, vgg_id.y_pred,
                               np.array([merge_clean(c) for c in test_df[test_df.model == "VGG19"].condition.values]),
                               "VGG19"),
    ]).pivot(index="Condition", columns="Model", values="Accuracy").round(3)
    pivot["Gap (Arc - VGG)"] = (pivot["ArcFace"] - pivot["VGG19"]).round(3)
    print(pivot)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:<28} {acc:.3f}")


if __name__ == "__main__":
    main()
